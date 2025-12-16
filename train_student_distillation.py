"""Train a lightweight student model using knowledge distillation from LaBraM.

This script trains the student model using teacher outputs saved in HDF5 format
by run_labram_inference.py. No LaBraM model is needed at training time - we
distill from the pre-computed embeddings.

Usage:
    python train_student_distillation.py \
        --input d:/neuro_datasets/derivatives/labram_bids_outputs.hdf5 \
        --model small \
        --output checkpoints/student_small.pth \
        --epochs 50 \
        --batch-size 64

For multi-GPU training:
    torchrun --nproc_per_node=2 train_student_distillation.py ...
"""

from __future__ import annotations

import argparse
import logging
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from student_model_design import (
    LaBraMStudent,
    labram_student_tiny,
    labram_student_small,
    labram_student_base,
    DistillationLoss,
)

LOGGER = logging.getLogger("distillation")


class LazyH5DistillationDataset(Dataset[Dict[str, Any]]):
    """Memory-efficient dataset that lazily loads from HDF5.

    Indexes all valid windows at init time but loads data on-demand.
    This allows training on datasets larger than RAM.
    """

    def __init__(
        self,
        h5_path: str,
        normalize_inputs: bool = True,
        load_tokens: bool = True,
        filter_nan: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.h5_path = h5_path
        self.normalize_inputs = normalize_inputs
        self.load_tokens = load_tokens

        # Build index of all valid windows
        self.index: List[Tuple[str, int]] = []
        self.recording_meta: Dict[str, Dict[str, Any]] = {}

        LOGGER.info(f"Indexing dataset: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            for rec_name in tqdm(list(f.keys()), desc="Indexing recordings"):
                item = f[rec_name]
                if not isinstance(item, h5py.Group):
                    continue
                grp: h5py.Group = item

                if 'embeddings' not in grp or 'inputs' not in grp:
                    continue

                emb_dset = grp['embeddings']
                if not isinstance(emb_dset, h5py.Dataset):
                    continue

                num_windows: int = emb_dset.shape[0]
                input_chans = list(grp.attrs.get('input_chans', [0]))

                # Store metadata
                self.recording_meta[rec_name] = {
                    'input_chans': input_chans,
                    'num_channels': grp.attrs.get('num_channels', len(input_chans) - 1),
                    'has_tokens': 'token_embeddings' in grp,
                }

                if filter_nan:
                    # Check which windows have valid embeddings
                    embeddings: np.ndarray = emb_dset[:]
                    valid_mask = ~np.isnan(embeddings).any(axis=1)
                    valid_indices = np.where(valid_mask)[0]

                    for idx in valid_indices:
                        self.index.append((rec_name, int(idx)))
                else:
                    for idx in range(num_windows):
                        self.index.append((rec_name, idx))

        LOGGER.info(f"Indexed {len(self.index)} valid windows from {len(self.recording_meta)} recordings")

        # Optionally limit dataset size (for debugging/testing)
        if max_samples is not None and len(self.index) > max_samples:
            random.shuffle(self.index)
            self.index = self.index[:max_samples]
            LOGGER.info(f"Limited to {max_samples} samples")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_name, window_idx = self.index[idx]
        meta = self.recording_meta[rec_name]

        with h5py.File(self.h5_path, 'r') as f:
            item = f[rec_name]
            assert isinstance(item, h5py.Group)
            grp: h5py.Group = item

            # Load input window [N, A, T]
            inputs_dset = grp['inputs']
            assert isinstance(inputs_dset, h5py.Dataset)
            inputs: np.ndarray = inputs_dset[window_idx].astype(np.float32)
            if self.normalize_inputs:
                inputs = inputs / 100.0

            # Load teacher embedding [D]
            emb_dset = grp['embeddings']
            assert isinstance(emb_dset, h5py.Dataset)
            embedding: np.ndarray = emb_dset[window_idx].astype(np.float32)

            result: Dict[str, Any] = {
                'inputs': torch.from_numpy(inputs),
                'embedding': torch.from_numpy(embedding),
                'input_chans': torch.tensor(meta['input_chans'], dtype=torch.long),
                'rec_name': rec_name,
            }

            # Load token embeddings if requested and available
            if self.load_tokens and meta['has_tokens']:
                tok_dset = grp['token_embeddings']
                assert isinstance(tok_dset, h5py.Dataset)
                tokens: np.ndarray = tok_dset[window_idx].astype(np.float32)
                result['tokens'] = torch.from_numpy(tokens)

            return result


def collate_variable_channels(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable channel counts.

    Pads inputs to the maximum channel count in the batch.
    """
    max_channels = max(b['inputs'].shape[0] for b in batch)
    max_seq_len = max_channels * batch[0]['inputs'].shape[1]  # N * A

    # Pad inputs
    padded_inputs = []
    padded_chans = []

    for b in batch:
        n_ch, n_patches, patch_size = b['inputs'].shape

        # Pad channels dimension
        if n_ch < max_channels:
            pad = torch.zeros(max_channels - n_ch, n_patches, patch_size)
            inputs = torch.cat([b['inputs'], pad], dim=0)
        else:
            inputs = b['inputs']

        padded_inputs.append(inputs)

        # Pad input_chans (add zeros for padding)
        chans = b['input_chans']
        if len(chans) < max_channels + 1:  # +1 for CLS
            chans = F.pad(chans, (0, max_channels + 1 - len(chans)), value=0)
        padded_chans.append(chans)

    result = {
        'inputs': torch.stack(padded_inputs),
        'embedding': torch.stack([b['embedding'] for b in batch]),
        'input_chans': torch.stack(padded_chans),
    }

    # Handle tokens if present
    if 'tokens' in batch[0]:
        # Pad token sequences
        max_tok_len = max(b['tokens'].shape[0] for b in batch)
        embed_dim = batch[0]['tokens'].shape[1]

        padded_tokens = []
        for b in batch:
            tok = b['tokens']
            if tok.shape[0] < max_tok_len:
                pad = torch.zeros(max_tok_len - tok.shape[0], embed_dim)
                tok = torch.cat([tok, pad], dim=0)
            padded_tokens.append(tok)

        result['tokens'] = torch.stack(padded_tokens)

    return result


def create_model(model_type: str, **kwargs) -> LaBraMStudent:
    """Create student model by type name."""
    if model_type == 'tiny':
        return labram_student_tiny(**kwargs)
    elif model_type == 'small':
        return labram_student_small(**kwargs)
    elif model_type == 'base':
        return labram_student_base(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss,
    device: str,
    use_token_loss: bool = True,
    grad_clip: float = 1.0,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_emb_loss = 0.0
    total_tok_loss = 0.0
    total_cos_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")

    for batch in pbar:
        inputs = batch['inputs'].to(device)
        teacher_emb = batch['embedding'].to(device)
        input_chans = batch['input_chans'][0].to(device)  # Use first sample's channels

        teacher_tokens = batch.get('tokens')
        if teacher_tokens is not None:
            teacher_tokens = teacher_tokens.to(device)

        # Forward
        if use_token_loss and teacher_tokens is not None:
            student_emb, student_tokens = model(
                inputs, input_chans=input_chans, return_all_tokens=True
            )
        else:
            student_emb = model(inputs, input_chans=input_chans)
            student_tokens = None

        # Loss
        loss, losses = criterion(
            student_emb, teacher_emb,
            student_tokens, teacher_tokens if use_token_loss else None
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track losses
        total_loss += losses['total']
        total_emb_loss += losses.get('emb_mse', 0)
        total_tok_loss += losses.get('tok_mse', 0)
        total_cos_loss += losses.get('emb_cos', 0)
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total']:.4f}",
            'emb': f"{losses.get('emb_mse', 0):.4f}",
        })

    return {
        'loss': total_loss / num_batches,
        'emb_mse': total_emb_loss / num_batches,
        'tok_mse': total_tok_loss / num_batches,
        'emb_cos': total_cos_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: DistillationLoss,
    device: str,
    use_token_loss: bool = True,
) -> Dict[str, float]:
    """Validate model on held-out data."""
    model.eval()

    total_loss = 0.0
    total_emb_loss = 0.0
    total_tok_loss = 0.0
    num_batches = 0

    # Also compute correlation metrics
    all_student_emb = []
    all_teacher_emb = []

    for batch in tqdm(dataloader, desc="Validating"):
        inputs = batch['inputs'].to(device)
        teacher_emb = batch['embedding'].to(device)
        input_chans = batch['input_chans'][0].to(device)

        teacher_tokens = batch.get('tokens')
        if teacher_tokens is not None:
            teacher_tokens = teacher_tokens.to(device)

        # Forward
        if use_token_loss and teacher_tokens is not None:
            student_emb, student_tokens = model(
                inputs, input_chans=input_chans, return_all_tokens=True
            )
        else:
            student_emb = model(inputs, input_chans=input_chans)
            student_tokens = None

        # Loss
        loss, losses = criterion(
            student_emb, teacher_emb,
            student_tokens, teacher_tokens if use_token_loss else None
        )

        total_loss += losses['total']
        total_emb_loss += losses.get('emb_mse', 0)
        total_tok_loss += losses.get('tok_mse', 0)
        num_batches += 1

        all_student_emb.append(student_emb.cpu())
        all_teacher_emb.append(teacher_emb.cpu())

    # Compute correlation
    all_student = torch.cat(all_student_emb, dim=0)
    all_teacher = torch.cat(all_teacher_emb, dim=0)

    # Cosine similarity
    cos_sim = F.cosine_similarity(all_student, all_teacher, dim=1).mean().item()

    # Per-dimension correlation (average Pearson r across embedding dims)
    correlations: list[float] = []
    for d in range(all_student.shape[1]):
        s = all_student[:, d]
        t = all_teacher[:, d]
        r = torch.corrcoef(torch.stack([s, t]))[0, 1].item()
        if not np.isnan(r):
            correlations.append(float(r))
    avg_corr = float(np.mean(correlations)) if correlations else 0.0

    return {
        'loss': total_loss / num_batches,
        'emb_mse': total_emb_loss / num_batches,
        'tok_mse': total_tok_loss / num_batches,
        'cos_sim': float(cos_sim),
        'avg_corr': avg_corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Train student model via distillation")
    parser.add_argument('--input', '-i', type=Path, default="d:/neuro_datasets/derivatives/labram_bids_outputs.hdf5",
                       help="HDF5 file with LaBraM outputs")
    parser.add_argument('--output', '-o', type=Path, default=Path('checkpoints/student_small.pth'),
                       help="Output checkpoint path")
    parser.add_argument('--model', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help="Student model size")
    parser.add_argument('--epochs', type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64,
                       help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument('--val-split', type=float, default=0.1,
                       help="Fraction of data for validation")
    parser.add_argument('--emb-weight', type=float, default=1.0,
                       help="Weight for embedding MSE loss")
    parser.add_argument('--tok-weight', type=float, default=0.5,
                       help="Weight for token MSE loss")
    parser.add_argument('--cos-weight', type=float, default=0.1,
                       help="Weight for cosine similarity loss")
    parser.add_argument('--no-token-loss', action='store_true',
                       help="Disable token-level distillation")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=None,
                       help="Limit training samples (for debugging)")
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--save-every', type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument('--resume', type=Path, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    LOGGER.info(f"Loading dataset from {args.input}")
    dataset = LazyH5DistillationDataset(
        args.input,
        normalize_inputs=True,
        load_tokens=not args.no_token_loss,
        max_samples=args.max_samples,
    )

    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    LOGGER.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_variable_channels,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_variable_channels,
        pin_memory=True,
    )

    # Create model
    LOGGER.info(f"Creating {args.model} student model")
    model = create_model(args.model)
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model has {n_params:,} parameters ({n_params/1e6:.2f}M)")

    # Create loss and optimizer
    criterion = DistillationLoss(
        embedding_weight=args.emb_weight,
        token_weight=args.tok_weight,
        cosine_weight=args.cos_weight,
        use_cosine=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume is not None and args.resume.exists():
        LOGGER.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    LOGGER.info("Starting training")
    history = {'train': [], 'val': []}

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            use_token_loss=not args.no_token_loss,
            grad_clip=args.grad_clip,
            epoch=epoch,
        )
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, args.device,
            use_token_loss=not args.no_token_loss,
        )
        history['val'].append(val_metrics)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        LOGGER.info(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"cos_sim={val_metrics['cos_sim']:.4f}, "
            f"corr={val_metrics['avg_corr']:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'model': model.state_dict(),
                'model_type': args.model,
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_cos_sim': val_metrics['cos_sim'],
                'args': vars(args),
            }, args.output.with_name(f"{args.output.stem}_best.pth"))
            LOGGER.info(f"Saved best model (val_loss={val_metrics['loss']:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'model_type': args.model,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, args.output.with_name(f"{args.output.stem}_epoch{epoch}.pth"))

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'model_type': args.model,
        'epoch': args.epochs - 1,
        'val_metrics': history['val'][-1],
        'args': vars(args),
        'history': history,
    }, args.output)

    LOGGER.info(f"Training complete. Final model saved to {args.output}")
    LOGGER.info(f"Best val_loss: {best_val_loss:.4f}")

    # Save training history
    with open(args.output.with_suffix('.json'), 'w') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'best_val_loss': best_val_loss,
        }, f, indent=2)


if __name__ == '__main__':
    main()
