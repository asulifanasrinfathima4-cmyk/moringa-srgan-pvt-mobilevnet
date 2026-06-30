import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used for lightweight local feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class MobileResidualBlock(nn.Module):
    """MobileVNet-inspired residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_skip = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            return x + out
        return out


class MobileVNetBackbone(nn.Module):
    """Compact encoder that preserves local disease texture with low parameter cost."""

    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(MobileResidualBlock(32, 48, stride=1), MobileResidualBlock(48, 48, stride=1))
        self.stage2 = nn.Sequential(MobileResidualBlock(48, 96, stride=2), MobileResidualBlock(96, 96, stride=1))
        self.stage3 = nn.Sequential(MobileResidualBlock(96, 160, stride=2), MobileResidualBlock(160, 160, stride=1))
        self.stage4 = nn.Sequential(MobileResidualBlock(160, embed_dim, stride=2), MobileResidualBlock(embed_dim, embed_dim, stride=1))
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class SpatialReductionAttention(nn.Module):
    """PVT-style attention with spatial reduction for keys and values."""

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 2, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            feature_map = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr(feature_map).reshape(batch, channels, -1).transpose(1, 2)
            reduced = self.norm(reduced)
        else:
            reduced = x

        kv = self.kv(reduced).reshape(batch, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        out = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        return self.proj_drop(out)


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PVTBlock(nn.Module):
    """Pyramid Vision Transformer block with spatial-reduction attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, sr_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), height, width)
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_position_encoding(num_tokens: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create deterministic sinusoidal position encoding."""
    pe = torch.zeros(num_tokens, dim, device=device)
    position = torch.arange(0, num_tokens, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class MobileVNetPVTClassifier(nn.Module):
    """Hybrid MobileVNet and Pyramid Vision Transformer classifier."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 192,
        transformer_depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = MobileVNetBackbone(embed_dim=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                PVTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    sr_ratio=2 if i < transformer_depth - 1 else 1,
                )
                for i in range(transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.backbone(x))
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + sinusoidal_position_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model_from_config(config: dict) -> MobileVNetPVTClassifier:
    """Factory function used by training, evaluation, and prediction scripts."""
    ccfg = config["classifier"]
    return MobileVNetPVTClassifier(
        num_classes=len(config["classes"]),
        embed_dim=int(ccfg["embed_dim"]),
        transformer_depth=int(ccfg["transformer_depth"]),
        num_heads=int(ccfg["num_heads"]),
        mlp_ratio=float(ccfg["mlp_ratio"]),
        dropout=float(ccfg["dropout"]),
    )
