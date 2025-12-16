"""Run LaBraM inference on a BIDS HDF5 dataset and save outputs for downstream training.

This script loads a preprocessed HDF5 dataset (created by make_bids_h5dataset.py),
runs inference through the pretrained LaBraM model, and saves both the input windows
and corresponding LaBraM embeddings to a new HDF5 file for training custom models.

Usage:
    python run_labram_inference.py \
        --input d:/neuro_datasets/derivatives/labram_bids.h5 \
        --output labram_bids_outputs.h5 \
        --checkpoint checkpoints/labram-base.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from timm.models import create_model
import modeling_finetune  # Register LaBraM models with timm

# Import channel mappings from separate module
from channel_maps import (
    STANDARD_1020,
    CHANNEL_MAPS,
    get_channel_map,
    list_channel_maps,
)

LOGGER = logging.getLogger("labram.inference")


_DATASET_ID_RE = re.compile(r"(ds\d{6})(?:-download)?", re.IGNORECASE)


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def infer_dataset_id_from_h5(
    rec_group: h5py.Group,
    eeg_dset: h5py.Dataset,
    input_path: Path,
) -> Optional[str]:
    """Infer dataset id (e.g., ds004460) for a given recording.

    Primary signal is the `source_bids_path` attribute written by
    `dataset_maker/make_bids_h5dataset.py`. Falls back to other metadata.
    """
    candidates: list[str] = []

    # Prefer explicit dataset id if present.
    if "dataset_id" in eeg_dset.attrs:
        candidates.append(_as_str(eeg_dset.attrs.get("dataset_id")))
    if "dataset_id" in rec_group.attrs:
        candidates.append(_as_str(rec_group.attrs.get("dataset_id")))

    if "source_bids_path" in eeg_dset.attrs:
        candidates.append(_as_str(eeg_dset.attrs.get("source_bids_path")))
    if "source_bids_path" in rec_group.attrs:
        candidates.append(_as_str(rec_group.attrs.get("source_bids_path")))

    # Last resort: input filename may include the dataset id.
    candidates.append(_as_str(input_path))

    for raw in candidates:
        if not raw:
            continue
        m = _DATASET_ID_RE.search(raw)
        if m:
            return m.group(1).lower()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LaBraM inference on BIDS HDF5 dataset")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("d:/neuro_datasets/derivatives/labram_bids.hdf5"),
        help="Input HDF5 file created by make_bids_h5dataset.py",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("d:/neuro_datasets/derivatives/labram_bids_outputs.hdf5"),
        help="Output HDF5 file for storing LaBraM features",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=Path("checkpoints/labram-base.pth"),
        help="Path to pretrained LaBraM checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="labram_base_patch200_200",
        choices=["labram_base_patch200_200", "labram_large_patch200_200", "labram_huge_patch200_200"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of 200-sample patches per window (4 = 4 seconds at 200Hz)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride in patches between windows (2 = 50%% overlap)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        default=False,
        help="Also save the input EEG windows (default: off)",
    )
    parser.add_argument(
        "--no-save-inputs",
        dest="save_inputs",
        action="store_false",
        help="Don't save input EEG windows, only embeddings",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--channel-map",
        type=str,
        nargs="+",
        default=None,
        help=f"Channel map(s) to use for electrode name mapping. "
             f"Available: {list_channel_maps()}. Multiple maps can be specified "
             f"(later maps override earlier). If omitted, the script will try to infer "
             f"the dataset id (e.g., ds004460) from each recording's source metadata "
             f"and apply that dataset's map automatically; otherwise it falls back to "
             f"the common default maps.",
    )
    parser.add_argument(
        "--list-channel-maps",
        action="store_true",
        help="List available channel maps and exit",
    )
    return parser.parse_args()


def get_input_chans(
    ch_names: List[str],
    channel_aliases: Optional[dict[str, str]] = None
) -> tuple[List[int], List[int], List[str]]:
    """Map channel names to standard 10-20 positions for LaBraM.

    Args:
        ch_names: List of channel names from the EEG recording.
        channel_aliases: Optional dict mapping non-standard names to 10-20 positions.
            If None, uses default mappings from get_channel_map().

    Handles:
    - Standard 10-20 names (FP1, CZ, O2, etc.)
    - Reference notation (C4:A1 -> C4, O2-A1 -> O2)
    - Non-standard systems via channel_aliases (cEEGrid, earEEG, etc.)

    Returns:
        input_chans: Position embedding indices (starts with 0 for CLS token)
        valid_eeg_indices: Indices into the original EEG array for valid channels
        valid_ch_names: Names of valid channels
    """
    if channel_aliases is None:
        channel_aliases = get_channel_map()

    def _candidate_labels(label: str) -> List[str]:
        """Generate progressively-simplified candidate labels for matching.

        This helps handle vendor prefixes like "BRAINVISION RDA_G01" by trying
        "RDA_G01" -> "G01" -> "G1".
        """
        candidates: List[str] = []
        if label:
            candidates.append(label)

        # Last whitespace token (e.g., "BRAINVISION RDA_G01" -> "RDA_G01")
        if " " in label:
            candidates.append(label.split()[-1])

        # Last underscore token (e.g., "RDA_G01" -> "G01")
        last = candidates[-1] if candidates else label
        if "_" in last:
            candidates.append(last.split("_")[-1])

        # Normalize zero-padded numeric suffix (e.g., "G01" -> "G1")
        normalized: List[str] = []
        for cand in candidates:
            m = re.match(r"^([A-Z]+)0+(\d+)$", cand)
            if m:
                normalized.append(f"{m.group(1)}{m.group(2)}")
        candidates.extend(normalized)

        # De-duplicate while preserving order
        out: List[str] = []
        seen = set()
        for cand in candidates:
            if cand and cand not in seen:
                out.append(cand)
                seen.add(cand)
        return out

    input_chans = [0]  # CLS token position
    valid_eeg_indices: List[int] = []
    valid_ch_names: List[str] = []

    for i, ch_name in enumerate(ch_names):
        ch_upper = ch_name.upper()
        # Strip reference notation (e.g., "C4:A1" -> "C4", "O2-A1" -> "O2")
        ch_base = ch_upper.split(':')[0].split('-')[0].strip()

        matched = False
        for cand in _candidate_labels(ch_base):
            # If an explicit alias is provided for this channel name, prefer it.
            # This allows dataset-specific maps to override ambiguous labels that
            # may also appear in STANDARD_1020 (e.g., BioSemi "A1" is not mastoid A1).
            if cand in channel_aliases:
                mapped_name = channel_aliases[cand]
                if mapped_name in STANDARD_1020:
                    idx = STANDARD_1020.index(mapped_name) + 1
                    input_chans.append(idx)
                    valid_eeg_indices.append(i)
                    valid_ch_names.append(ch_name)
                    LOGGER.debug(f"Channel '{ch_name}' mapped via alias: {cand} -> {mapped_name}")
                    matched = True
                    break
                LOGGER.debug(
                    f"Channel '{ch_name}' alias target '{mapped_name}' is not in standard 10-20 list, skipping"
                )
                matched = True
                break

            # Direct standard match
            if cand in STANDARD_1020:
                idx = STANDARD_1020.index(cand) + 1
                input_chans.append(idx)
                valid_eeg_indices.append(i)
                valid_ch_names.append(ch_name)
                matched = True
                break

        if matched:
            continue

        LOGGER.debug(f"Channel '{ch_name}' (base: '{ch_base}') not in standard 10-20 or aliases, skipping")

    return input_chans, valid_eeg_indices, valid_ch_names


def load_model(
    model_name: str,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    """Load pretrained LaBraM model for feature extraction."""
    LOGGER.info(f"Loading model {model_name} from {checkpoint_path}")

    # Create model without classification head (num_classes=0 makes head Identity)
    # init_values=0.1 enables LayerScale (gamma_1/gamma_2) to match pretrained checkpoint
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=0,  # Remove classification head for feature extraction
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_mean_pooling=True,
        init_values=0.1,  # Enable LayerScale to match pretrained weights
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip 'student.' prefix if present (from pretraining)
    # Also remap keys for architecture differences
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student."):
            k = k[8:]  # Remove 'student.' prefix
        # Skip pretraining-specific weights
        if k.startswith(("head.", "lm_head.", "projection_head.")):
            continue
        if k in ("mask_token", "logit_scale"):
            continue
        # Remap norm -> fc_norm for mean pooling mode
        if k in ("norm.weight", "norm.bias"):
            k = k.replace("norm.", "fc_norm.")
        new_state_dict[k] = v

    # Load weights (allow missing keys for removed head)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (head removed for feature extraction)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing:
            LOGGER.warning(f"Missing keys (may affect model quality): {missing}")
    if unexpected:
        LOGGER.warning(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    LOGGER.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def extract_windows(
    eeg_data: np.ndarray,
    window_size: int,
    stride: int,
    patch_size: int = 200,
) -> np.ndarray:
    """Extract sliding windows from continuous EEG data.

    Args:
        eeg_data: EEG data of shape [num_channels, time_samples]
        window_size: Number of patches per window
        stride: Stride in patches between windows
        patch_size: Samples per patch (200 for LaBraM)

    Returns:
        Windows of shape [num_windows, num_channels, window_size, patch_size]
    """
    num_channels, total_samples = eeg_data.shape
    window_samples = window_size * patch_size
    stride_samples = stride * patch_size

    # Calculate number of complete windows
    if total_samples < window_samples:
        LOGGER.warning(f"Recording too short ({total_samples} samples < {window_samples} window size)")
        return np.array([])

    num_windows = (total_samples - window_samples) // stride_samples + 1

    windows = []
    for i in range(num_windows):
        start = i * stride_samples
        end = start + window_samples
        window = eeg_data[:, start:end]
        # Reshape to [num_channels, window_size, patch_size]
        window = window.reshape(num_channels, window_size, patch_size)
        windows.append(window)

    return np.array(windows, dtype=np.float32)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    windows: np.ndarray,
    input_chans: List[int],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run LaBraM inference on EEG windows.

    Args:
        model: LaBraM model
        windows: EEG windows of shape [num_windows, num_channels, window_size, patch_size]
        input_chans: Channel mapping for position embeddings
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        Features of shape [num_windows, embed_dim]
    """
    num_windows = windows.shape[0]
    all_features = []

    for start_idx in range(0, num_windows, batch_size):
        end_idx = min(start_idx + batch_size, num_windows)
        batch = windows[start_idx:end_idx]

        # Convert to tensor and normalize (divide by 100 as per LaBraM convention)
        batch_tensor = torch.from_numpy(batch).float().to(device) / 100.0

        # Forward pass - model returns features (no head since num_classes=0)
        features = model(batch_tensor, input_chans=input_chans)

        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handle --list-channel-maps
    if args.list_channel_maps:
        print("Available channel maps:")
        for name in list_channel_maps():
            map_data = CHANNEL_MAPS[name]
            print(f"  {name}: {len(map_data)} mappings")
        print("\nUse --channel-map <name> [<name2> ...] to select maps.")
        print("Example: --channel-map ds004460  (for EASYCAP equidistant cap)")
        print("Example: --channel-map ceegrid eareeg  (for cEEGrid + earEEG)")
        return

    # If --channel-map is provided, we use it globally.
    # Otherwise we infer dataset per-recording and select the map at runtime.
    explicit_channel_aliases: Optional[dict[str, str]] = None
    if args.channel_map is not None:
        explicit_channel_aliases = get_channel_map(args.channel_map)
        LOGGER.info(
            f"Using explicit channel maps: {args.channel_map} ({len(explicit_channel_aliases)} aliases)"
        )
    else:
        LOGGER.info("Auto-selecting channel map per recording based on dataset id")

    channel_alias_cache: dict[str, dict[str, str]] = {}

    # Validate paths
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {args.output}. Use --overwrite to replace.")

    # Load model
    model = load_model(args.model, args.checkpoint, args.device)
    embed_dim = model.embed_dim

    # Open input and output HDF5 files
    LOGGER.info(f"Opening input file: {args.input}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5_in:
        with h5py.File(args.output, "w") as h5_out:
            # Store metadata
            h5_out.attrs["source_file"] = str(args.input)
            h5_out.attrs["model"] = args.model
            h5_out.attrs["checkpoint"] = str(args.checkpoint)
            h5_out.attrs["window_size"] = args.window_size
            h5_out.attrs["stride"] = args.stride
            h5_out.attrs["patch_size"] = 200
            h5_out.attrs["embed_dim"] = embed_dim

            # Process each recording
            recording_names = list(h5_in.keys())
            LOGGER.info(f"Found {len(recording_names)} recordings in input file")

            total_windows = 0
            processed = 0
            skipped = 0

            for rec_name in tqdm(recording_names, desc="Processing recordings"):
                grp = h5_in[rec_name]
                if not isinstance(grp, h5py.Group):
                    LOGGER.warning(f"Skipping {rec_name}: not a group")
                    skipped += 1
                    continue

                if "eeg" not in grp:
                    LOGGER.warning(f"Skipping {rec_name}: no 'eeg' dataset")
                    skipped += 1
                    continue

                eeg_dset = grp["eeg"]
                if not isinstance(eeg_dset, h5py.Dataset):
                    LOGGER.warning(f"Skipping {rec_name}: 'eeg' is not a dataset")
                    skipped += 1
                    continue

                eeg_data: np.ndarray = eeg_dset[:]

                # Get channel order
                ch_order: List[str]
                if "chOrder" in eeg_dset.attrs:
                    ch_order_raw = eeg_dset.attrs["chOrder"]
                    try:
                        # h5py attrs can return Empty type, handled by except
                        ch_order_list = list(ch_order_raw)  # pyright: ignore[reportArgumentType]
                        ch_order = [ch.decode() if isinstance(ch, bytes) else str(ch)
                                   for ch in ch_order_list]
                    except TypeError:
                        LOGGER.warning(f"Invalid channel order for {rec_name}, using default indices")
                        ch_order = [f"CH{i}" for i in range(eeg_data.shape[0])]
                else:
                    LOGGER.warning(f"No channel order for {rec_name}, using default indices")
                    ch_order = [f"CH{i}" for i in range(eeg_data.shape[0])]

                # Pick the right channel map.
                if explicit_channel_aliases is not None:
                    channel_aliases = explicit_channel_aliases
                else:
                    dataset_id = infer_dataset_id_from_h5(grp, eeg_dset, args.input)
                    cache_key = dataset_id or "__default__"
                    if cache_key not in channel_alias_cache:
                        if dataset_id is None:
                            channel_alias_cache[cache_key] = get_channel_map()
                            LOGGER.info(
                                "No dataset id found for %s; using common default channel maps",
                                rec_name,
                            )
                        else:
                            channel_alias_cache[cache_key] = get_channel_map([dataset_id], include_default=True)
                            LOGGER.info(
                                "Detected dataset %s for %s; using channel map '%s'",
                                dataset_id,
                                rec_name,
                                dataset_id,
                            )
                    channel_aliases = channel_alias_cache[cache_key]

                # Get input channel mapping and filter to valid channels
                input_chans, valid_eeg_indices, valid_ch_names = get_input_chans(ch_order, channel_aliases)

                # Check if we have enough valid channels
                if len(input_chans) < 2:  # Only CLS token
                    LOGGER.warning(f"Skipping {rec_name}: no valid channels found")
                    skipped += 1
                    continue

                # Filter EEG data to only include valid channels
                eeg_data_filtered = eeg_data[valid_eeg_indices, :]

                # Extract windows from filtered data
                windows = extract_windows(
                    eeg_data_filtered,
                    window_size=args.window_size,
                    stride=args.stride,
                )

                if len(windows) == 0:
                    LOGGER.warning(f"Skipping {rec_name}: recording too short")
                    skipped += 1
                    continue

                # Run inference
                features = run_inference(
                    model,
                    windows,
                    input_chans,
                    args.batch_size,
                    args.device,
                )

                # Create output group
                out_grp = h5_out.create_group(rec_name)

                # Save embeddings
                out_grp.create_dataset(
                    "embeddings",
                    data=features,
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                )

                # Optionally save input windows
                if args.save_inputs:
                    out_grp.create_dataset(
                        "inputs",
                        data=windows,
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=4,
                    )

                # Save metadata
                out_grp.attrs["num_windows"] = len(windows)
                out_grp.attrs["num_channels"] = len(valid_ch_names)
                out_grp.attrs["ch_order"] = valid_ch_names
                out_grp.attrs["input_chans"] = input_chans
                out_grp.attrs["original_samples"] = int(eeg_data.shape[1])
                out_grp.attrs["original_num_channels"] = len(ch_order)

                # Copy over source attributes if present
                if "source_bids_path" in eeg_dset.attrs:
                    out_grp.attrs["source_bids_path"] = eeg_dset.attrs["source_bids_path"]

                total_windows += len(windows)
                processed += 1

    LOGGER.info(f"Inference complete!")
    LOGGER.info(f"  Processed: {processed} recordings")
    LOGGER.info(f"  Skipped: {skipped} recordings")
    LOGGER.info(f"  Total windows: {total_windows:,}")
    LOGGER.info(f"  Output file: {args.output}")
    LOGGER.info(f"  Output size: {args.output.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
