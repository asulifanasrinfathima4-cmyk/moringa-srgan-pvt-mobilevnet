from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block used in the SRGAN generator."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """PixelShuffle upsampling block."""

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRGenerator(nn.Module):
    """Compact SRGAN generator for 2x image super-resolution."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_residual_blocks: int = 8, scale_factor: int = 2) -> None:
        super().__init__()
        if scale_factor != 2:
            raise ValueError("This compact implementation is configured for scale_factor=2.")
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_residual_blocks)])
        self.mid = nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1), nn.BatchNorm2d(base_channels))
        self.upsample = UpsampleBlock(base_channels, scale=2)
        self.output = nn.Sequential(nn.Conv2d(base_channels, in_channels, kernel_size=9, padding=4), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.initial(x)
        x2 = self.residuals(x1)
        x3 = self.mid(x2)
        x4 = x1 + x3
        return self.output(self.upsample(x4))


class SRDiscriminator(nn.Module):
    """Patch discriminator for distinguishing real and generated HR leaf images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, stride: int, normalize: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(in_channels, base_channels, 1, normalize=False),
            block(base_channels, base_channels, 2),
            block(base_channels, base_channels * 2, 1),
            block(base_channels * 2, base_channels * 2, 2),
            block(base_channels * 2, base_channels * 4, 1),
            block(base_channels * 4, base_channels * 4, 2),
            block(base_channels * 4, base_channels * 8, 1),
            block(base_channels * 8, base_channels * 8, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for structure-preserving loss."""
    channels = x.shape[1]
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)


class SRGANLoss(nn.Module):
    """Hybrid SRGAN loss: content, edge-preservation, and adversarial terms."""

    def __init__(self, lambda_content: float = 1.0, lambda_edge: float = 0.05, lambda_adversarial: float = 0.001) -> None:
        super().__init__()
        self.lambda_content = lambda_content
        self.lambda_edge = lambda_edge
        self.lambda_adversarial = lambda_adversarial
        self.content_loss = nn.L1Loss()
        self.edge_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, fake_hr: torch.Tensor, real_hr: torch.Tensor, fake_logits: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        real_labels = torch.ones_like(fake_logits)
        content = self.content_loss(fake_hr, real_hr)
        edge = self.edge_loss(sobel_edges(fake_hr), sobel_edges(real_hr))
        adversarial = self.adversarial_loss(fake_logits, real_labels)
        total = self.lambda_content * content + self.lambda_edge * edge + self.lambda_adversarial * adversarial
        return total, {"content": content.item(), "edge": edge.item(), "adversarial": adversarial.item(), "total": total.item()}

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        real_loss = self.adversarial_loss(real_logits, real_labels)
        fake_loss = self.adversarial_loss(fake_logits, fake_labels)
        return 0.5 * (real_loss + fake_loss)
