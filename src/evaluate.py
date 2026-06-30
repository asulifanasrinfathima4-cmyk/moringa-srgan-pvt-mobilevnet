import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from dataset import get_classifier_dataloaders
from mobilevnet_pvt import build_model_from_config
from utils import append_metrics_csv, count_parameters, ensure_dir, get_device, load_config


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(config_path: str, checkpoint_path: str, use_sr: bool = False) -> None:
    config = load_config(config_path)
    device = get_device(config)
    _, _, test_loader, class_names = get_classifier_dataloaders(config, use_sr=use_sr)

    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    y_true, y_pred = [], []
    elapsed_times = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            elapsed_times.append(elapsed / images.size(0))
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    parameters = count_parameters(model)
    inference_time_ms = float(np.mean(elapsed_times) * 1000.0)

    output_dir = ensure_dir(config["evaluation"]["output_dir"])
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, output_dir / config["evaluation"]["confusion_matrix_name"])

    metrics = {
        "model": "SRGAN-MobileVNet-PVT" if use_sr else "MobileVNet-PVT",
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "parameters": parameters,
        "inference_time_ms": round(inference_time_ms, 6),
    }
    append_metrics_csv(output_dir / "metrics_summary.csv", metrics)

    print("Evaluation completed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Classification report: {output_dir / 'classification_report.txt'}")
    print(f"Confusion matrix: {output_dir / config['evaluation']['confusion_matrix_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained MobileVNet-PVT classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Classifier checkpoint path")
    parser.add_argument("--use-sr", action="store_true", help="Use dataset_sr test split")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, use_sr=args.use_sr)


if __name__ == "__main__":
    main()
