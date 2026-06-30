from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_transform(image_size: int, train: bool = False) -> transforms.Compose:
    """Return image transformations for classifier training/evaluation."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def srgan_pair_transform(hr_size: int, lr_size: int) -> transforms.Compose:
    """Create a transform placeholder for SRGAN datasets.

    The SRGAN dataset class performs separate low-resolution and high-resolution
    resizing internally, so this function is kept for API clarity.
    """
    return transforms.Compose([transforms.Resize((hr_size, hr_size)), transforms.ToTensor()])


class SRGANDataset(torch.utils.data.Dataset):
    """Dataset that returns low-resolution and high-resolution image pairs.

    High-resolution targets are resized to `hr_size`. Low-resolution inputs are
    obtained by downsampling the same image to `lr_size`. This is suitable when
    separate paired LR/HR acquisitions are unavailable.
    """

    def __init__(self, root: str, hr_size: int = 256, lr_size: int = 128) -> None:
        self.root = Path(root)
        self.hr_transform = transforms.Compose([transforms.Resize((hr_size, hr_size)), transforms.ToTensor()])
        self.lr_transform = transforms.Compose([transforms.Resize((lr_size, lr_size)), transforms.ToTensor()])
        self.samples = [p for p in self.root.rglob("*") if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        image = Image.open(self.samples[index]).convert("RGB")
        lr_image = self.lr_transform(image)
        hr_image = self.hr_transform(image)
        return lr_image, hr_image


def build_imagefolder(root: str, image_size: int, train: bool = False) -> datasets.ImageFolder:
    """Build a torchvision ImageFolder with validation checks."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")
    dataset = datasets.ImageFolder(root=str(root_path), transform=image_transform(image_size, train=train))
    if len(dataset) == 0:
        raise FileNotFoundError(f"No class images found in: {root_path}")
    return dataset


def get_classifier_dataloaders(config: Dict, use_sr: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test dataloaders."""
    image_size = int(config["image"]["image_size"])
    batch_size = int(config["classifier"]["batch_size"])
    num_workers = int(config.get("num_workers", 2))
    dcfg = config["dataset"]

    train_dir = dcfg["sr_train_dir"] if use_sr else dcfg["train_dir"]
    val_dir = dcfg["sr_val_dir"] if use_sr else dcfg["val_dir"]
    test_dir = dcfg["sr_test_dir"] if use_sr else dcfg["test_dir"]

    train_data = build_imagefolder(train_dir, image_size, train=True)
    val_data = build_imagefolder(val_dir, image_size, train=False)
    test_data = build_imagefolder(test_dir, image_size, train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_data.classes


def get_srgan_dataloader(config: Dict) -> DataLoader:
    """Create dataloader for SRGAN training."""
    hr_size = int(config["image"]["image_size"])
    lr_size = int(config["image"]["low_resolution_size"])
    batch_size = int(config["srgan"]["batch_size"])
    train_dir = config["dataset"]["train_dir"]
    dataset = SRGANDataset(train_dir, hr_size=hr_size, lr_size=lr_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(config.get("num_workers", 2)), pin_memory=True)


def single_image_transform(config: Dict) -> transforms.Compose:
    """Transform used by predict.py."""
    return image_transform(int(config["image"]["image_size"]), train=False)
