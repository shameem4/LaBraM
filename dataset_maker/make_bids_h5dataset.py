"""BIDS → LaBraM HDF5 converter using existing preprocessing utilities.

This script walks one or more BIDS-compliant EEG datasets (default location:
`D:/neuro_datasets`), applies the same filtering / resampling stack used by the
shipping LaBraM scripts, and stores the results in the HDF5 layout consumed by
`data_processor.dataset` and `utils.build_pretraining_dataset`. It inspects the
subject/session `scans.tsv` metadata to automatically resolve the underlying EEG
file extension, allowing ingestion of binary formats such as EDF/BDF and
tabular exports (`*.tsv`, `*.csv`, and their `.gz` variants). When the supplied
`--bids-root` contains multiple datasets (for example, several OpenNeuro downloads
in sibling folders), the script will traverse each dataset automatically.

Usage example (Windows PowerShell):

```
python dataset_maker/make_bids_h5dataset.py \
    --bids-root D:/neuro_datasets \
    --output-dir ./checkpoints/bids_h5 \
    --dataset-name bids_labram \
    --channel-template dataset_maker/channel_order_62.json

Dry-run example (no output written):

```
python dataset_maker/make_bids_h5dataset.py \
    --bids-root D:/neuro_datasets \
    --log-level INFO \
    --dry-run
```
```

The script requires `mne-bids` (already installed per user note) and only adds a
thin orchestration layer around the existing `shock.utils` helpers to avoid
duplicating preprocessing logic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import mne
from mne_bids import BIDSPath, get_bids_path_from_fname, read_raw_bids

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_maker.shock.utils.h5 import h5Dataset
from dataset_maker.shock.utils.eegUtils import preprocessing_edf

SUPPORTED_HELPER_EXTS = {".edf", ".bdf"}
TABULAR_EXTENSIONS = {".tsv": "\t", ".csv": ","}
LOGGER = logging.getLogger("labram.bids")

EEGArray = NDArray[np.floating]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BIDS EEG datasets into LaBraM-style HDF5 shards"
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path("D:/neuro_datasets"),
        help="Root directory that contains the BIDS datasets (default: D:/neuro_datasets)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./labram_bids_h5"),
        help="Destination folder for generated .hdf5 files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="labram_bids",
        help="Name used for the output HDF5 file (creates <name>.hdf5)",
    )
    parser.add_argument(
        "--l-freq",
        type=float,
        default=0.1,
        help="High-pass cutoff in Hz (match LaBraM defaults)",
    )
    parser.add_argument(
        "--h-freq",
        type=float,
        default=75.0,
        help="Low-pass cutoff in Hz",
    )
    parser.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Notch frequency in Hz (set to 60 for US mains)",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=200,
        help="Target sampling rate in Hz",
    )
    parser.add_argument(
        "--drop-channels",
        nargs="*",
        default=["M1", "M2", "VEO", "HEO", "ECG"],
        help="Channels to drop if present (case-sensitive per mne)",
    )
    parser.add_argument(
        "--channel-template",
        type=Path,
        default=None,
        help="Optional JSON/TXT file listing the desired channel ordering",
    )
    parser.add_argument(
        "--trim-seconds",
        type=int,
        default=0,
        help="Drop this many seconds from the tail of each recording (matches make_h5dataset_for_pretrain trimming)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the datasets and report estimated HDF5 sizes, but do not write output",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List discovered BIDS dataset roots under --bids-root and exit",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Only process a single discovered dataset (0-based index from --list-datasets)",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=1,
        help="Stop after processing this many discovered datasets (debug helper)",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Stop after processing this many recordings total (debug helper)",
    )
    parser.add_argument(
        "--resample-n-jobs",
        type=str,
        default="1",
        help="Value passed to mne Raw.resample n_jobs (int or 'cuda'); use 1 if unsure",
    )
    parser.add_argument(
        "--mne-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for MNE's internal logger",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_channel_template(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Channel template not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(ch).upper() for ch in data]
    except json.JSONDecodeError:
        pass
    return [line.strip().upper() for line in text.splitlines() if line.strip()]


def iter_bids_paths(bids_root: Path) -> Iterable[BIDSPath]:
    for dataset_root in discover_bids_roots(bids_root):
        yield from _iter_dataset_paths(dataset_root)


def discover_bids_roots(root: Path) -> List[Path]:
    root = root.resolve()
    descriptor = root / "dataset_description.json"
    if descriptor.exists():
        return [root]

    children = []
    try:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "dataset_description.json").exists():
                children.append(child)
    except FileNotFoundError:
        LOGGER.error("BIDS root %s cannot be read", root)
        return []

    if children:
        LOGGER.info("Discovered %d BIDS dataset(s) under %s", len(children), root)
        return children

    LOGGER.warning("No dataset_description.json found under %s; treating it as a single BIDS dataset", root)
    return [root]


def select_bids_roots(
    root: Path, dataset_index: Optional[int] = None, max_datasets: Optional[int] = None
) -> List[Path]:
    roots = discover_bids_roots(root)
    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(roots):
            raise IndexError(
                f"--dataset-index {dataset_index} is out of range; discovered {len(roots)} dataset(s)"
            )
        roots = [roots[dataset_index]]
    if max_datasets is not None:
        if max_datasets < 1:
            raise ValueError("--max-datasets must be >= 1")
        roots = roots[:max_datasets]
    return roots


def iter_bids_paths_selected(
    bids_root: Path,
    dataset_index: Optional[int] = None,
    max_datasets: Optional[int] = None,
) -> Iterable[BIDSPath]:
    for dataset_root in select_bids_roots(
        bids_root, dataset_index=dataset_index, max_datasets=max_datasets
    ):
        yield from _iter_dataset_paths(dataset_root)


def _iter_dataset_paths(dataset_root: Path) -> Iterable[BIDSPath]:
    discovered_any = False
    for meta_path in _iter_metadata_bids_paths(dataset_root):
        discovered_any = True
        yield meta_path

    if discovered_any:
        return

    template = BIDSPath(root=dataset_root, datatype="eeg", suffix="eeg")
    matches = template.match()
    if not matches:
        LOGGER.warning("No EEG files found under %s", dataset_root)
        return []
    for match in matches:
        yield match


def _iter_metadata_bids_paths(bids_root: Path) -> Iterable[BIDSPath]:
    seen = set()
    for scans_file in bids_root.rglob("scans.tsv"):
        if any(part == "derivatives" for part in scans_file.parts):
            continue
        for bids_path in _paths_from_scans_file(scans_file, bids_root):
            if bids_path.fpath in seen:
                continue
            seen.add(bids_path.fpath)
            yield bids_path


def _paths_from_scans_file(scans_path: Path, bids_root: Path) -> Iterable[BIDSPath]:
    if not scans_path.exists():
        return []

    base_dir = scans_path.parent
    try:
        with scans_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                return []
            filename_key = next((name for name in reader.fieldnames if name and name.lower() == "filename"), None)
            if filename_key is None:
                return []
            for row in reader:
                rel_path = (row.get(filename_key) or "").strip()
                if not rel_path or "_eeg" not in rel_path:
                    continue
                data_path = (base_dir / rel_path).resolve()
                try:
                    data_path.relative_to(bids_root.resolve())
                except ValueError:
                    LOGGER.debug("Skipping %s because it is outside the BIDS root", data_path)
                    continue
                try:
                    yield get_bids_path_from_fname(data_path, check=False)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Could not build BIDSPath from %s: %s", data_path, exc)
    except FileNotFoundError:
        return []


def _canonical_extension(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return ""
    if suffixes[-1] == ".gz" and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1]


def _open_text_file(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_tabular_matrix(path: Path, delimiter: str) -> Tuple[EEGArray, List[str]]:
    with _open_text_file(path) as handle:
        header_line = handle.readline().strip()
        if not header_line:
            raise ValueError(f"Missing header row in {path}")
        headers = [col.strip() or f"CH{idx+1}" for idx, col in enumerate(header_line.split(delimiter))]
        data = np.loadtxt(handle, delimiter=delimiter, dtype=np.float64, ndmin=2)
    data = data.T  # channels × samples
    return cast(EEGArray, data), [name.upper() for name in headers]


def _load_sampling_frequency(bids_path: BIDSPath) -> float:
    sidecar = bids_path.copy().update(suffix="eeg", extension=".json", check=False)
    if not sidecar.fpath.exists():
        raise FileNotFoundError(f"Sidecar JSON not found for {bids_path.basename}")
    metadata = json.loads(sidecar.fpath.read_text(encoding="utf-8"))
    if "SamplingFrequency" not in metadata:
        raise KeyError(f"SamplingFrequency missing in {sidecar.fpath}")
    return float(metadata["SamplingFrequency"])


def apply_standard_preprocessing(
    raw,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    resample: int,
    resample_n_jobs,
    mne_verbose: str,
) -> Tuple[EEGArray, List[str]]:
    raw.load_data(verbose=mne_verbose)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=mne_verbose)
    raw.notch_filter(notch_freq, verbose=mne_verbose)
    raw.resample(resample, n_jobs=resample_n_jobs, verbose=mne_verbose)
    data = cast(EEGArray, raw.get_data(units="uV"))
    return data, list(raw.ch_names)


def preprocess_recording(
    bids_path: BIDSPath,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    resample: int,
    resample_n_jobs,
    mne_verbose: str,
    drop_channels: Optional[Sequence[str]],
    standard_channels: Optional[Sequence[str]],
) -> Tuple[Optional[EEGArray], List[str]]:
    ext = _canonical_extension(bids_path.fpath)
    if ext == ".edf":
        data, ch_names = preprocessing_edf(
            bids_path.fpath,
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=resample,
            drop_channels=list(drop_channels) if drop_channels else [],
            standard_channels=list(standard_channels) if standard_channels else [],
        )
        if data is None:
            return None, []
        return cast(EEGArray, data), [str(name) for name in ch_names]

    if ext == ".bdf":
        data, ch_names = preprocess_bdf_recording(
            bids_path,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            resample=resample,
            resample_n_jobs=resample_n_jobs,
            mne_verbose=mne_verbose,
            drop_channels=drop_channels,
            standard_channels=standard_channels,
        )
        return data, ch_names

    if ext in TABULAR_EXTENSIONS:
        data, ch_names = preprocess_tabular_recording(
            bids_path,
            delimiter=TABULAR_EXTENSIONS[ext],
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            resample=resample,
            resample_n_jobs=resample_n_jobs,
            mne_verbose=mne_verbose,
            drop_channels=drop_channels,
            standard_channels=standard_channels,
        )
        return data, ch_names

    raw = read_raw_bids(bids_path, verbose="ERROR")
    if drop_channels:
        usable = [ch for ch in raw.ch_names if ch not in drop_channels]
        raw.pick_channels(usable, ordered=True)
    if standard_channels and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(list(standard_channels))
        except ValueError:
            LOGGER.warning("Channel reorder failed for %s; keeping native order", bids_path.basename)
    data, ch_names = apply_standard_preprocessing(
        raw,
        l_freq,
        h_freq,
        notch_freq,
        resample,
        resample_n_jobs=resample_n_jobs,
        mne_verbose=mne_verbose,
    )
    return data, ch_names


def preprocess_tabular_recording(
    bids_path: BIDSPath,
    delimiter: str,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    resample: int,
    resample_n_jobs,
    mne_verbose: str,
    drop_channels: Optional[Sequence[str]],
    standard_channels: Optional[Sequence[str]],
) -> Tuple[EEGArray, List[str]]:
    data_matrix, ch_names = _load_tabular_matrix(bids_path.fpath, delimiter)
    sfreq = _load_sampling_frequency(bids_path)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_matrix, info, verbose="ERROR")

    if drop_channels:
        drop = [ch for ch in drop_channels if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)
    if standard_channels and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(list(standard_channels))
        except ValueError:
            LOGGER.warning("Channel reorder failed for %s; keeping native order", bids_path.basename)

    data, ch_names = apply_standard_preprocessing(
        raw,
        l_freq,
        h_freq,
        notch_freq,
        resample,
        resample_n_jobs=resample_n_jobs,
        mne_verbose=mne_verbose,
    )
    return data, ch_names


def preprocess_bdf_recording(
    bids_path: BIDSPath,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    resample: int,
    resample_n_jobs,
    mne_verbose: str,
    drop_channels: Optional[Sequence[str]],
    standard_channels: Optional[Sequence[str]],
) -> Tuple[EEGArray, List[str]]:
    raw = mne.io.read_raw_bdf(bids_path.fpath, preload=True, verbose="ERROR")
    if drop_channels:
        drop = [ch for ch in drop_channels if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)
    if standard_channels and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(list(standard_channels))
        except ValueError:
            LOGGER.warning("Channel reorder failed for %s; keeping native order", bids_path.basename)
    data, ch_names = apply_standard_preprocessing(
        raw,
        l_freq,
        h_freq,
        notch_freq,
        resample,
        resample_n_jobs=resample_n_jobs,
        mne_verbose=mne_verbose,
    )
    return data, ch_names


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    mne.set_log_level(args.mne_log_level)

    resample_n_jobs_raw = (args.resample_n_jobs or "").strip()
    if resample_n_jobs_raw.lower() == "cuda":
        resample_n_jobs = "cuda"
    else:
        try:
            resample_n_jobs = int(resample_n_jobs_raw)
        except ValueError as exc:
            raise ValueError("--resample-n-jobs must be an int or 'cuda'") from exc

    if args.list_datasets:
        roots = discover_bids_roots(args.bids_root)
        if not roots:
            print(f"No BIDS datasets discovered under {args.bids_root.resolve()}")
            return
        for idx, root in enumerate(roots):
            print(f"[{idx}] {root}")
        return

    output_file = args.output_dir / f"{args.dataset_name}.hdf5"
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(f"{output_file} already exists. Use --overwrite to replace it.")

    channel_template = load_channel_template(args.channel_template)
    dataset = None if args.dry_run else h5Dataset(args.output_dir, args.dataset_name)

    processed = 0
    chunks = None
    total_bytes = 0
    for bids_path in iter_bids_paths_selected(
        args.bids_root, dataset_index=args.dataset_index, max_datasets=args.max_datasets
    ):
        if args.max_recordings is not None and processed >= args.max_recordings:
            LOGGER.info("Reached --max-recordings=%d, stopping early", args.max_recordings)
            break
        LOGGER.info("Processing %s", bids_path.basename)
        if not bids_path.fpath.exists():
            LOGGER.warning("Skipping %s because file is missing (%s)", bids_path.basename, bids_path.fpath)
            continue
        try:
            eeg_data, ch_names = preprocess_recording(
                bids_path,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch,
                resample=args.resample,
                resample_n_jobs=resample_n_jobs,
                mne_verbose=args.mne_log_level,
                drop_channels=args.drop_channels,
                standard_channels=channel_template,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process %s: %s", bids_path.basename, exc)
            continue

        if eeg_data is None or len(ch_names) == 0:
            LOGGER.warning("Skipping %s due to empty data", bids_path.basename)
            continue

        if args.trim_seconds > 0:
            trim = args.trim_seconds * args.resample
            if eeg_data.shape[1] > trim:
                eeg_data = eeg_data[:, :-trim]

        ch_order = [name.upper() for name in ch_names]
        chunks = chunks or (len(ch_order), args.resample)

        group_name = bids_path.basename.replace(".eeg", "")
        bytes_estimate = eeg_data.size * 4  # float32 storage
        total_bytes += bytes_estimate
        LOGGER.info(
            "Estimated size for %s: %.2f MB (%d channels × %d samples)",
            group_name,
            bytes_estimate / 1e6,
            eeg_data.shape[0],
            eeg_data.shape[1],
        )

        if dataset is None:
            processed += 1
            continue

        grp = dataset.addGroup(group_name)
        dset = dataset.addDataset(grp, "eeg", eeg_data.astype(np.float32), chunks)
        dataset.addAttributes(dset, "lFreq", args.l_freq)
        dataset.addAttributes(dset, "hFreq", args.h_freq)
        dataset.addAttributes(dset, "rsFreq", args.resample)
        dataset.addAttributes(dset, "chOrder", ch_order)
        dataset.addAttributes(dset, "source_bids_path", str(bids_path.fpath))

        processed += 1

    if dataset is not None:
        dataset.save()
        LOGGER.info(
            "Finished processing %d recordings into %s (%.2f GB)",
            processed,
            output_file,
            (total_bytes / 1e9),
        )
    else:
        LOGGER.info(
            "Dry run complete: %d recordings discovered. Estimated output size %.2f GB across %s",
            processed,
            total_bytes / 1e9,
            output_file,
        )


if __name__ == "__main__":
    main()
