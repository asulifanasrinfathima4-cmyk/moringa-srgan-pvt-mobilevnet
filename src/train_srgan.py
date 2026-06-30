import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_srgan_dataloader
from srgan import SRDiscriminator, SRGANLoss, SRGenerator
from utils import AverageMeter, ensure_dir, get_device, load_config, save_checkpoint, seed_everything


def train_srgan(config_path: str) -> None:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config)
    loader = get_srgan_dataloader(config)

    scfg = config["srgan"]
    checkpoint_dir = ensure_dir(scfg["checkpoint_dir"])

    generator = SRGenerator(scale_factor=int(scfg["scale_factor"])).to(device)
    discriminator = SRDiscriminator().to(device)
    criterion = SRGANLoss(
        lambda_content=float(scfg["lambda_content"]),
        lambda_edge=float(scfg["lambda_edge"]),
        lambda_adversarial=float(scfg["lambda_adversarial"]),
    )
    opt_g = Adam(generator.parameters(), lr=float(scfg["learning_rate_generator"]), betas=(0.9, 0.999))
    opt_d = Adam(discriminator.parameters(), lr=float(scfg["learning_rate_discriminator"]), betas=(0.9, 0.999))
    scaler = GradScaler(enabled=device.type == "cuda")

    best_g_loss = float("inf")
    epochs = int(scfg["epochs"])

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_meter = AverageMeter()
        d_meter = AverageMeter()

        progress = tqdm(loader, desc=f"SRGAN epoch {epoch}/{epochs}")
        for lr_image, hr_image in progress:
            lr_image = lr_image.to(device, non_blocking=True)
            hr_image = hr_image.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                fake_hr = generator(lr_image)
                real_logits = discriminator(hr_image)
                fake_logits_detached = discriminator(fake_hr.detach())
                d_loss = criterion.discriminator_loss(real_logits, fake_logits_detached)

            opt_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            with autocast(enabled=device.type == "cuda"):
                fake_hr = generator(lr_image)
                fake_logits = discriminator(fake_hr)
                g_loss, g_terms = criterion.generator_loss(fake_hr, hr_image, fake_logits)

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            g_meter.update(g_loss.item(), lr_image.size(0))
            d_meter.update(d_loss.item(), lr_image.size(0))
            progress.set_postfix(g_loss=f"{g_meter.avg:.4f}", d_loss=f"{d_meter.avg:.4f}")

        latest_path = checkpoint_dir / "generator_latest.pt"
        save_checkpoint({"epoch": epoch, "generator_state": generator.state_dict(), "config": config}, latest_path)
        save_checkpoint({"epoch": epoch, "discriminator_state": discriminator.state_dict(), "config": config}, checkpoint_dir / "discriminator_latest.pt")

        if g_meter.avg < best_g_loss:
            best_g_loss = g_meter.avg
            save_checkpoint({"epoch": epoch, "generator_state": generator.state_dict(), "best_g_loss": best_g_loss, "config": config}, checkpoint_dir / "generator_best.pt")

    print(f"SRGAN training completed. Best generator loss: {best_g_loss:.6f}")
    print(f"Best generator checkpoint: {checkpoint_dir / 'generator_best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SRGAN for offline Moringa leaf image enhancement.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    train_srgan(args.config)


if __name__ == "__main__":
    main()
