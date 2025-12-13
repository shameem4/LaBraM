"""BIDS → LaBraM HDF5 converter using existing preprocessing utilities.

This script walks one or more BIDS-compliant EEG datasets (default location:
`D:/neuro_datasets`), applies the same filtering / resampling stack used by the
shipping LaBraM scripts, and stores the results in the HDF5 layout consumed by
`data_processor.dataset` and `utils.build_pretraining_dataset`.

Usage example (Windows PowerShell):

```
python dataset_maker/make_bids_h5dataset.py \
    --bids-root D:/neuro_datasets \
    --output-dir ./checkpoints/bids_h5 \
    --dataset-name bids_labram \
    --channel-template dataset_maker/channel_order_62.json
```

The script requires `mne-bids` (already installed per user note) and only adds a
thin orchestration layer around the existing `shock.utils` helpers to avoid
duplicating preprocessing logic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence

import numpy as np
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_maker.shock.utils.h5 import h5Dataset
from dataset_maker.shock.utils.eegUtils import preprocessing_edf

SUPPORTED_HELPER_EXTS = {".edf", ".bdf"}
LOGGER = logging.getLogger("labram.bids")


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
        help="Walk the datasets, report estimated HDF5 sizes, but do not write output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
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
    template = BIDSPath(root=bids_root, datatype="eeg", suffix="eeg")
    matches = template.match()
    if not matches:
        LOGGER.warning("No EEG files found under %s", bids_root)
        return []
    for match in matches:
        yield match


def apply_standard_preprocessing(raw, l_freq: float, h_freq: float, notch_freq: float, resample: int):
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.notch_filter(notch_freq)
    raw.resample(resample, n_jobs="auto")
    return raw.get_data(units="uV"), raw.ch_names


def preprocess_recording(
    bids_path: BIDSPath,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    resample: int,
    drop_channels: Optional[Sequence[str]],
    standard_channels: Optional[Sequence[str]],
):
    ext = bids_path.extension or ""
    if ext in SUPPORTED_HELPER_EXTS:
        return preprocessing_edf(
            bids_path.fpath,
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=resample,
            drop_channels=list(drop_channels) if drop_channels else None,
            standard_channels=list(standard_channels) if standard_channels else None,
        )

    raw = read_raw_bids(bids_path, verbose="ERROR")
    if drop_channels:
        usable = [ch for ch in raw.ch_names if ch not in drop_channels]
        raw.pick_channels(usable, ordered=True)
    if standard_channels and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(list(standard_channels))
        except ValueError:
            LOGGER.warning("Channel reorder failed for %s; keeping native order", bids_path.basename)
    data, ch_names = apply_standard_preprocessing(raw, l_freq, h_freq, notch_freq, resample)
    return data, ch_names


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

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
    for bids_path in iter_bids_paths(args.bids_root):
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
