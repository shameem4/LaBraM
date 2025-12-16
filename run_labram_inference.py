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

# Standard 10-20 electrode positions used by LaBraM
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

LOGGER = logging.getLogger("labram.inference")


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
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def get_input_chans(ch_names: List[str]) -> tuple[List[int], List[int], List[str]]:
    """Map channel names to standard 10-20 positions for LaBraM.

    Returns:
        input_chans: Position embedding indices (starts with 0 for CLS token)
        valid_eeg_indices: Indices into the original EEG array for valid channels
        valid_ch_names: Names of valid channels
    """
    input_chans = [0]  # CLS token position
    valid_eeg_indices: List[int] = []
    valid_ch_names: List[str] = []

    for i, ch_name in enumerate(ch_names):
        ch_upper = ch_name.upper()
        try:
            idx = STANDARD_1020.index(ch_upper) + 1
            input_chans.append(idx)
            valid_eeg_indices.append(i)
            valid_ch_names.append(ch_name)
        except ValueError:
            LOGGER.debug(f"Channel '{ch_name}' not in standard 10-20 montage, skipping")

    return input_chans, valid_eeg_indices, valid_ch_names


def load_model(
    model_name: str,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    """Load pretrained LaBraM model for feature extraction."""
    LOGGER.info(f"Loading model {model_name} from {checkpoint_path}")

    # Create model without classification head (num_classes=0 makes head Identity)
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=0,  # Remove classification head for feature extraction
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_mean_pooling=True,
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
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student."):
            k = k[8:]  # Remove 'student.' prefix
        # Skip classification head weights
        if k.startswith("head."):
            continue
        new_state_dict[k] = v

    # Load weights (allow missing keys for removed head)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        LOGGER.debug(f"Missing keys: {missing}")
    if unexpected:
        LOGGER.debug(f"Unexpected keys: {unexpected}")

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

                # Get input channel mapping and filter to valid channels
                input_chans, valid_eeg_indices, valid_ch_names = get_input_chans(ch_order)

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
