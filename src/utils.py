import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    """Make training as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(config: Dict[str, Any]) -> torch.device:
    """Return the configured computation device."""
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory when it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Track streaming average for losses and metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a mini-batch."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a model/training checkpoint."""
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    """Load model weights from a checkpoint saved by this project."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint.get("generator_state", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def append_metrics_csv(path: str | Path, row: Dict[str, Any], header: Optional[Iterable[str]] = None) -> None:
    """Append one row of metrics to a CSV file."""
    path = Path(path)
    ensure_dir(path.parent)
    fields = list(header or row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
