import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import single_image_transform
from mobilevnet_pvt import build_model_from_config
from srgan import SRGenerator
from utils import get_device, load_config


def optionally_enhance_with_srgan(image: Image.Image, config: dict, sr_checkpoint: str | None, device: torch.device) -> Image.Image:
    """Enhance a single image with SRGAN when a generator checkpoint is supplied."""
    if sr_checkpoint is None:
        return image

    lr_size = int(config["image"]["low_resolution_size"])
    generator = SRGenerator(scale_factor=int(config["srgan"]["scale_factor"])).to(device)
    checkpoint = torch.load(sr_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint.get("generator_state", checkpoint), strict=True)
    generator.eval()

    to_tensor = transforms.Compose([transforms.Resize((lr_size, lr_size)), transforms.ToTensor()])
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        sr = generator(to_tensor(image).unsqueeze(0).to(device)).clamp(0, 1).squeeze(0).cpu()
    return to_pil(sr)


def predict(config_path: str, checkpoint_path: str, image_path: str, sr_checkpoint: str | None = None) -> None:
    config = load_config(config_path)
    device = get_device(config)
    class_names = list(config["classes"])

    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = optionally_enhance_with_srgan(image, config, sr_checkpoint, device)
    transform = single_image_transform(config)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    print(f"Predicted class: {class_names[int(predicted_idx)]}")
    print(f"Confidence: {float(confidence):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Moringa leaf disease class for one image.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Classifier checkpoint path")
    parser.add_argument("--image", required=True, help="Input leaf image path")
    parser.add_argument("--sr-checkpoint", default=None, help="Optional SRGAN generator checkpoint for enhancement")
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.image, sr_checkpoint=args.sr_checkpoint)


if __name__ == "__main__":
    main()
