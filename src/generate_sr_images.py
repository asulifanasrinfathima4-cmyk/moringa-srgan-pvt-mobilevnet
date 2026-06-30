import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from srgan import SRGenerator
from utils import ensure_dir, get_device, load_config


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_generator(config: dict, checkpoint_path: str, device: torch.device) -> SRGenerator:
    generator = SRGenerator(scale_factor=int(config["srgan"]["scale_factor"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint.get("generator_state", checkpoint), strict=True)
    generator.eval()
    return generator


def enhance_split(generator: SRGenerator, input_dir: str, output_dir: str, lr_size: int, device: torch.device) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input split not found: {input_dir}")

    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    lr_transform = transforms.Compose([transforms.Resize((lr_size, lr_size)), transforms.ToTensor()])
    to_pil = transforms.ToPILImage()

    for image_path in tqdm(image_paths, desc=f"Enhancing {input_dir.name}"):
        rel = image_path.relative_to(input_dir)
        out_path = output_dir / rel
        ensure_dir(out_path.parent)
        image = Image.open(image_path).convert("RGB")
        lr_tensor = lr_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = generator(lr_tensor).clamp(0, 1).squeeze(0).cpu()
        to_pil(sr_tensor).save(out_path)


def generate_sr_dataset(config_path: str, checkpoint_path: str) -> None:
    config = load_config(config_path)
    device = get_device(config)
    generator = load_generator(config, checkpoint_path, device)
    lr_size = int(config["image"]["low_resolution_size"])
    dcfg = config["dataset"]

    split_pairs = [
        (dcfg["train_dir"], dcfg["sr_train_dir"]),
        (dcfg["val_dir"], dcfg["sr_val_dir"]),
        (dcfg["test_dir"], dcfg["sr_test_dir"]),
    ]
    for input_dir, output_dir in split_pairs:
        enhance_split(generator, input_dir, output_dir, lr_size, device)

    print("Super-resolved dataset generation completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SRGAN-enhanced train/val/test images.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to trained SRGAN generator checkpoint")
    args = parser.parse_args()
    generate_sr_dataset(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
