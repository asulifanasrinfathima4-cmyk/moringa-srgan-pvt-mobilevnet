import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_classifier_dataloaders
from mobilevnet_pvt import build_model_from_config
from utils import AverageMeter, count_parameters, ensure_dir, get_device, load_config, save_checkpoint, seed_everything, top1_accuracy


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(loader, desc="train" if is_train else "validate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1_accuracy(logits.detach(), labels), images.size(0))

    return loss_meter.avg, acc_meter.avg


def train_classifier(config_path: str, use_sr: bool = False) -> None:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config)
    train_loader, val_loader, _, class_names = get_classifier_dataloaders(config, use_sr=use_sr)

    model = build_model_from_config(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["classifier"]["learning_rate"]),
        weight_decay=float(config["classifier"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["classifier"]["epochs"]))
    scaler = GradScaler(enabled=device.type == "cuda")

    checkpoint_dir = ensure_dir(config["classifier"]["checkpoint_dir"])
    best_path = checkpoint_dir / config["classifier"]["best_model_name"]
    best_val_acc = 0.0

    print(f"Classes: {class_names}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Training source: {'SRGAN-enhanced dataset' if use_sr else 'original dataset'}")

    epochs = int(config["classifier"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict(), "class_names": class_names, "config": config},
            checkpoint_dir / "latest_mobilevnet_pvt.pt",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "config": config,
                },
                best_path,
            )

    print(f"Classifier training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileVNet-PVT classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--use-sr", action="store_true", help="Use dataset_sr paths from config.yaml")
    args = parser.parse_args()
    train_classifier(args.config, use_sr=args.use_sr)


if __name__ == "__main__":
    main()
