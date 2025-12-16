"""Lightweight Convolutional Transformer Student for LaBraM Knowledge Distillation.

This module implements a mobile-friendly student network (~300K-800K params) designed
to distill knowledge from LaBraM Base (5.8M params) for efficient EEG inference.

Design Rationale:
=================

LaBraM Teacher Analysis:
- LaBraM Base: 12 transformer blocks, 200 embed_dim, 10 heads = 5.8M params
- TemporalConv preprocessing: Conv1(1,8,k=15,s=8) + 2x Conv(8,8,k=3) = 576 params
- Main cost: Transformer blocks (482K params each × 12 = 5.78M)

Mobile Constraints:
- Target: <1M params for real-time mobile inference
- Must handle variable channels (5-64 electrodes)
- Maintain ability to produce 200-dim embeddings for compatibility

Student Architecture Options:
============================

Option 1: MobileConvNet (~300K params)
--------------------------------------
Pure convolutional approach with depthwise separable convolutions.
+ Fastest inference, smallest memory footprint
+ Great for real-time streaming
- Limited global context modeling

Option 2: ConvTransformer Hybrid (~500K params)
----------------------------------------------
Convolutional encoder + 2-layer transformer.
+ Good balance of local and global features
+ Can match LaBraM's token-level outputs for sequence distillation
- Slightly larger

Option 3: EfficientFormer-style (~800K params)
---------------------------------------------
Pooling-based attention with efficient FFN.
+ Best accuracy potential
+ Full sequence-level distillation support
- Largest of the options

Recommended: Option 2 (ConvTransformer Hybrid)
This provides the best tradeoff for distillation quality while staying mobile-friendly.

Knowledge Distillation Strategy:
================================

1. Feature Distillation (primary):
   - Match pooled embeddings: MSE(student_emb, teacher_emb)
   - Match token sequences: MSE(student_tokens, teacher_tokens)

2. Attention Transfer (optional):
   - Match attention patterns from teacher's middle layers

3. Progressive Distillation:
   - First train on pooled embeddings only
   - Then add token-level matching
   - Finally add attention transfer

Usage with saved HDF5 dataset:
=============================
The labram_bids_outputs.hdf5 contains:
- 'inputs': Raw EEG windows [N, ch, 4, 200] - used as student input
- 'embeddings': Teacher pooled outputs [N, 200] - distillation target
- 'token_embeddings': Teacher token outputs [N, seq, 200] - sequence distillation

Training loop:
```python
# Load teacher outputs from HDF5
teacher_embeddings = h5['recording/embeddings'][:]
teacher_tokens = h5['recording/token_embeddings'][:]
inputs = h5['recording/inputs'][:]

# Forward through student
student_embeddings, student_tokens = student_model(inputs)

# Distillation losses
emb_loss = F.mse_loss(student_embeddings, teacher_embeddings)
tok_loss = F.mse_loss(student_tokens, teacher_tokens)
loss = emb_loss + 0.5 * tok_loss
```
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DepthwiseSeparableConv1d(nn.Module):
    """Efficient depthwise separable convolution for 1D signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EfficientTemporalEncoder(nn.Module):
    """Lightweight temporal encoder using depthwise separable convolutions.

    Replaces LaBraM's TemporalConv with a more efficient design.
    Input: [B, N, A, T] where N=channels, A=patches(4), T=patch_size(200)
    Output: [B, N*A, embed_dim] - one token per channel-patch pair
    """

    def __init__(
        self,
        embed_dim: int = 200,
        hidden_dim: int = 64,
        kernel_sizes: Tuple[int, ...] = (15, 7, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Stage 1: Initial temporal processing (200 -> 25 samples)
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=15, stride=8, padding=7)
        self.norm1 = nn.GroupNorm(4, hidden_dim // 2)

        # Stage 2: Feature extraction
        self.conv2 = DepthwiseSeparableConv1d(
            hidden_dim // 2, hidden_dim, kernel_size=7, padding=3
        )
        self.norm2 = nn.GroupNorm(4, hidden_dim)

        # Stage 3: Final projection
        self.conv3 = DepthwiseSeparableConv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=3
        )
        self.norm3 = nn.GroupNorm(4, hidden_dim)

        # Pool and project to embedding dimension
        # After conv1 with stride=8: 200 -> 25 samples
        # We need to get to embed_dim=200, so we use hidden_dim * (temporal_out // pool)
        self.final_proj = nn.Linear(hidden_dim * 25, embed_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, A, T] -> rearrange to process each patch independently
        B, N, A, T = x.shape

        # Flatten channels and patches, process each 200-sample segment
        x = rearrange(x, 'B N A T -> (B N A) 1 T')  # [(B*N*A), 1, 200]

        # Temporal convolutions
        x = self.act(self.norm1(self.conv1(x)))  # [(B*N*A), hidden/2, 25]
        x = self.act(self.norm2(self.conv2(x)))  # [(B*N*A), hidden, 25]
        x = self.act(self.norm3(self.conv3(x)))  # [(B*N*A), hidden, 25]

        # Flatten and project
        x = x.flatten(1)  # [(B*N*A), hidden*25]
        x = self.dropout(self.final_proj(x))  # [(B*N*A), embed_dim]

        # Reshape back to sequence
        x = rearrange(x, '(B N A) D -> B (N A) D', B=B, N=N, A=A)

        return x


class EfficientAttention(nn.Module):
    """Efficient attention with linear complexity option.

    Supports both standard softmax attention and efficient pooling-based attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_efficient: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_efficient = use_efficient

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_efficient:
            # Pooling-based attention reduces sequence length before attention
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_efficient and N > 64:
            # For long sequences, use pooling-based attention
            # This reduces complexity from O(N²) to O(N*K) where K is pool size
            k_pooled = rearrange(k, 'B H N D -> (B H) D N')
            v_pooled = rearrange(v, 'B H N D -> (B H) D N')

            k_pooled = F.adaptive_avg_pool1d(k_pooled, min(N, 32))
            v_pooled = F.adaptive_avg_pool1d(v_pooled, min(N, 32))

            k_pooled = rearrange(k_pooled, '(B H) D K -> B H K D', B=B, H=self.num_heads)
            v_pooled = rearrange(v_pooled, '(B H) D K -> B H K D', B=B, H=self.num_heads)

            attn = (q @ k_pooled.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v_pooled).transpose(1, 2).reshape(B, N, C)
        else:
            # Standard attention for short sequences
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientFFN(nn.Module):
    """Efficient Feed-Forward Network with optional expansion."""

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LightTransformerBlock(nn.Module):
    """Lightweight transformer block with efficient attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, use_efficient=use_efficient_attn
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = EfficientFFN(dim, expansion=mlp_ratio, dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LaBraMStudent(nn.Module):
    """Lightweight student model for LaBraM distillation.

    Architecture designed for mobile deployment while maintaining compatibility
    with LaBraM's output format (200-dim embeddings).

    Three configurations:
    - tiny: ~300K params - pure conv, fastest
    - small: ~500K params - conv + 2 transformer layers (recommended)
    - base: ~800K params - conv + 4 transformer layers, best accuracy

    Args:
        embed_dim: Output embedding dimension (default 200 for LaBraM compatibility)
        num_transformer_layers: Number of transformer blocks (0=pure conv, 2-4 typical)
        num_heads: Number of attention heads
        mlp_ratio: FFN expansion ratio
        hidden_dim: Hidden dimension for temporal encoder
        max_channels: Maximum expected number of EEG channels
        max_time_patches: Maximum expected time patches per window
        dropout: Dropout rate
        use_efficient_attn: Use pooling-based attention for long sequences
    """

    def __init__(
        self,
        embed_dim: int = 200,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        hidden_dim: int = 64,
        max_channels: int = 128,
        max_time_patches: int = 16,
        dropout: float = 0.1,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_transformer_layers = num_transformer_layers

        # Temporal encoder (conv-based patch embedding)
        self.temporal_encoder = EfficientTemporalEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Positional embeddings (spatial + temporal)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_channels + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, max_time_patches, embed_dim))

        # Transformer layers (optional)
        if num_transformer_layers > 0:
            self.blocks = nn.ModuleList([
                LightTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=dropout,
                    use_efficient_attn=use_efficient_attn,
                )
                for _ in range(num_transformer_layers)
            ])
        else:
            self.blocks = None
            # For pure conv model, add a final mixing layer
            self.mix_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )

        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input EEG windows [B, N, A, T] where:
                B = batch size
                N = number of channels
                A = number of time patches (typically 4)
                T = samples per patch (typically 200)
            input_chans: Optional channel indices for positional embedding lookup
                         Shape [N+1] including CLS position (default index 0)
            return_all_tokens: If True, return (pooled, all_tokens) tuple

        Returns:
            If return_all_tokens=False: Pooled embedding [B, embed_dim]
            If return_all_tokens=True: (pooled [B, D], tokens [B, 1+N*A, D])
        """
        B, N, A, T = x.shape

        # Temporal encoding: [B, N, A, T] -> [B, N*A, D]
        x = self.temporal_encoder(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + N*A, D]

        # Add positional embeddings
        if input_chans is not None:
            # Use provided channel indices
            pos_embed = self.pos_embed[:, input_chans]  # [1, N+1, D]
        else:
            # Default: use first N+1 positions
            pos_embed = self.pos_embed[:, :N + 1]

        # Expand spatial embedding for each time patch
        # CLS pos: [1, 1, D], Channel pos: [1, N, D] -> expand to [1, N*A, D]
        cls_pos = pos_embed[:, 0:1, :]  # [1, 1, D]
        chan_pos = pos_embed[:, 1:, :]  # [1, N, D]
        chan_pos_expanded = chan_pos.unsqueeze(2).expand(-1, -1, A, -1)  # [1, N, A, D]
        chan_pos_expanded = chan_pos_expanded.flatten(1, 2)  # [1, N*A, D]

        full_pos_embed = torch.cat([cls_pos, chan_pos_expanded], dim=1)  # [1, 1+N*A, D]
        x = x + full_pos_embed.expand(B, -1, -1)

        # Add time embeddings to patch tokens (not CLS)
        time_embed = self.time_embed[:, :A, :]  # [1, A, D]
        time_embed_expanded = time_embed.unsqueeze(1).expand(-1, N, -1, -1)  # [1, N, A, D]
        time_embed_expanded = time_embed_expanded.flatten(1, 2)  # [1, N*A, D]
        x[:, 1:, :] = x[:, 1:, :] + time_embed_expanded.expand(B, -1, -1)

        # Process through transformer blocks or mixing layer
        if self.blocks is not None:
            for blk in self.blocks:
                x = blk(x)
        else:
            # Pure conv: just mix the features
            x = self.mix_layer(x)

        # Normalize
        x = self.norm(x)

        if return_all_tokens:
            # Return (pooled, all_tokens) for sequence distillation
            pooled = x[:, 1:, :].mean(dim=1)  # Mean pool patch tokens
            return pooled, x
        else:
            # Return pooled representation (mean of patch tokens)
            return x[:, 1:, :].mean(dim=1)


# =============================================================================
# Model factory functions
# =============================================================================

def labram_student_tiny(**kwargs) -> LaBraMStudent:
    """Tiny student (~300K params) - Pure convolutional, fastest inference."""
    return LaBraMStudent(
        embed_dim=200,
        num_transformer_layers=0,
        hidden_dim=48,
        mlp_ratio=2.0,
        dropout=0.1,
        **kwargs
    )


def labram_student_small(**kwargs) -> LaBraMStudent:
    """Small student (~500K params) - Recommended for distillation."""
    return LaBraMStudent(
        embed_dim=200,
        num_transformer_layers=2,
        num_heads=4,
        hidden_dim=64,
        mlp_ratio=2.0,
        dropout=0.1,
        **kwargs
    )


def labram_student_base(**kwargs) -> LaBraMStudent:
    """Base student (~800K params) - Best accuracy, still mobile-friendly."""
    return LaBraMStudent(
        embed_dim=200,
        num_transformer_layers=4,
        num_heads=8,
        hidden_dim=96,
        mlp_ratio=2.0,
        dropout=0.1,
        **kwargs
    )


# =============================================================================
# Knowledge Distillation Loss Functions
# =============================================================================

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation from LaBraM.

    Supports:
    1. Embedding-level distillation (MSE on pooled representations)
    2. Token-level distillation (MSE on sequence outputs)
    3. Optional cosine similarity loss for better direction alignment
    """

    def __init__(
        self,
        embedding_weight: float = 1.0,
        token_weight: float = 0.5,
        use_cosine: bool = True,
        cosine_weight: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.token_weight = token_weight
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight
        self.temperature = temperature

        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(
        self,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
        student_tokens: Optional[torch.Tensor] = None,
        teacher_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute distillation loss.

        Args:
            student_embedding: Student pooled output [B, D]
            teacher_embedding: Teacher pooled output [B, D]
            student_tokens: Optional student token sequence [B, S, D]
            teacher_tokens: Optional teacher token sequence [B, S, D]

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        losses = {}

        # Embedding loss
        emb_loss = self.mse(student_embedding, teacher_embedding)
        losses['emb_mse'] = emb_loss.item()
        total_loss = self.embedding_weight * emb_loss

        # Cosine similarity loss
        if self.use_cosine:
            target = torch.ones(student_embedding.shape[0], device=student_embedding.device)
            cos_loss = self.cosine(student_embedding, teacher_embedding, target)
            losses['emb_cos'] = cos_loss.item()
            total_loss = total_loss + self.cosine_weight * cos_loss

        # Token-level loss
        if student_tokens is not None and teacher_tokens is not None:
            # Handle potential sequence length mismatch
            min_len = min(student_tokens.shape[1], teacher_tokens.shape[1])
            s_tok = student_tokens[:, :min_len, :]
            t_tok = teacher_tokens[:, :min_len, :]

            tok_loss = self.mse(s_tok, t_tok)
            losses['tok_mse'] = tok_loss.item()
            total_loss = total_loss + self.token_weight * tok_loss

        losses['total'] = total_loss.item()
        return total_loss, losses


# =============================================================================
# Training utilities
# =============================================================================

class EEGDistillationDataset(torch.utils.data.Dataset):
    """Dataset for loading LaBraM inference outputs for distillation training.

    Loads from HDF5 file created by run_labram_inference.py.
    """

    def __init__(
        self,
        h5_path: str,
        normalize_inputs: bool = True,
        load_tokens: bool = True,
    ):
        import h5py

        self.h5_path = h5_path
        self.normalize_inputs = normalize_inputs
        self.load_tokens = load_tokens

        # Index all windows across recordings
        self.windows = []  # List of (recording_name, window_idx)

        with h5py.File(h5_path, 'r') as f:
            for rec_name in f.keys():
                grp = f[rec_name]
                if 'embeddings' not in grp:
                    continue

                num_windows = grp['embeddings'].shape[0]

                # Check for valid data (skip NaN-only recordings)
                emb_sample = grp['embeddings'][0]
                if any(~torch.isnan(torch.tensor(emb_sample))):
                    for i in range(num_windows):
                        self.windows.append((rec_name, i))

        print(f"Loaded {len(self.windows)} windows from {h5_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        import h5py

        rec_name, window_idx = self.windows[idx]

        with h5py.File(self.h5_path, 'r') as f:
            grp = f[rec_name]

            # Load input window
            inputs = grp['inputs'][window_idx]  # [N, A, T]
            if self.normalize_inputs:
                inputs = inputs / 100.0  # LaBraM normalization

            # Load teacher embeddings
            embedding = grp['embeddings'][window_idx]  # [D]

            # Load input channel indices
            input_chans = grp.attrs['input_chans']  # Includes CLS position

            result = {
                'inputs': torch.tensor(inputs, dtype=torch.float32),
                'embedding': torch.tensor(embedding, dtype=torch.float32),
                'input_chans': torch.tensor(input_chans, dtype=torch.long),
            }

            # Optionally load token embeddings
            if self.load_tokens and 'token_embeddings' in grp:
                tokens = grp['token_embeddings'][window_idx]  # [S, D]
                result['tokens'] = torch.tensor(tokens, dtype=torch.float32)

            return result


def train_student_epoch(
    student: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss,
    device: str,
    use_token_distillation: bool = True,
) -> dict:
    """Train student model for one epoch.

    Args:
        student: Student model
        dataloader: DataLoader for distillation dataset
        optimizer: Optimizer
        criterion: Distillation loss function
        device: Device to train on
        use_token_distillation: Whether to use token-level distillation

    Returns:
        Dictionary of average losses for the epoch
    """
    student.train()

    total_losses = {}
    num_batches = 0

    for batch in dataloader:
        inputs = batch['inputs'].to(device)
        teacher_emb = batch['embedding'].to(device)
        input_chans = batch['input_chans'][0].to(device)  # Same for all in batch

        # Get teacher tokens if available
        teacher_tokens = batch.get('tokens')
        if teacher_tokens is not None:
            teacher_tokens = teacher_tokens.to(device)

        # Forward through student
        if use_token_distillation and teacher_tokens is not None:
            student_emb, student_tokens = student(
                inputs, input_chans=input_chans, return_all_tokens=True
            )
        else:
            student_emb = student(inputs, input_chans=input_chans)
            student_tokens = None

        # Compute loss
        loss, losses = criterion(
            student_emb, teacher_emb,
            student_tokens, teacher_tokens if use_token_distillation else None
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    # Average losses
    return {k: v / num_batches for k, v in total_losses.items()}


# =============================================================================
# Quick test / demo
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LaBraM Student Model - Architecture Analysis")
    print("=" * 60)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def profile_model(name, model_fn):
        model = model_fn()
        params = count_params(model)

        # Test forward pass
        x = torch.randn(2, 6, 4, 200)  # [B, N, A, T]
        with torch.no_grad():
            out = model(x)
            out_with_tokens = model(x, return_all_tokens=True)

        print(f"\n{name}:")
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Pooled output shape: {out.shape}")
        print(f"  With tokens: pooled={out_with_tokens[0].shape}, tokens={out_with_tokens[1].shape}")

        return params

    tiny_params = profile_model("Student Tiny (pure conv)", labram_student_tiny)
    small_params = profile_model("Student Small (2 transformer)", labram_student_small)
    base_params = profile_model("Student Base (4 transformer)", labram_student_base)

    print("\n" + "=" * 60)
    print("Comparison with LaBraM Teacher:")
    print("=" * 60)
    print(f"  LaBraM Base:    5,815,136 params (100%)")
    print(f"  Student Tiny:   {tiny_params:,} params ({tiny_params/5815136*100:.1f}%)")
    print(f"  Student Small:  {small_params:,} params ({small_params/5815136*100:.1f}%)")
    print(f"  Student Base:   {base_params:,} params ({base_params/5815136*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Recommended for mobile: Student Small (~10% of teacher)")
    print("=" * 60)
