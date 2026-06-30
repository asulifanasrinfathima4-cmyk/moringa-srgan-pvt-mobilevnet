import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

from utils import load_config, ensure_dir


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def preprocess_image(image_path: Path, image_size: int) -> Image.Image:
    """Resize, denoise, and contrast-normalize one leaf image."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.equalize(y)
    image = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
    return image


def preprocess_directory(input_root: str, output_root: str, image_size: int) -> None:
    """Apply preprocessing while preserving class-folder structure."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    image_paths = [p for p in input_root.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_root}")

    for path in tqdm(image_paths, desc="Preprocessing images"):
        rel = path.relative_to(input_root)
        out_path = output_root / rel
        ensure_dir(out_path.parent)
        processed = preprocess_image(path, image_size)
        processed.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Moringa leaf images.")
    parser.add_argument("--input", required=True, help="Input dataset root, e.g., dataset")
    parser.add_argument("--output", required=True, help="Output dataset root, e.g., dataset_preprocessed")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = int(config["image"]["image_size"])
    preprocess_directory(args.input, args.output, image_size)


if __name__ == "__main__":
    main()
