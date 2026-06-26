from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))
sys.path.insert(0, str(LAB13_ROOT / "task133_train_transformer"))

from skeleton_transformer import SkeletonTransformer
from utils import LABELS, label_id_to_name, load_label_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Skeleton Transformer 测试集表现。")
    parser.add_argument("--processed_dir", type=Path, default=LAB13_ROOT / "data" / "processed")
    parser.add_argument("--model_path", type=Path, default=LAB13_ROOT / "task133_train_transformer" / "output" / "best_model.pth")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required = ["X_test.npy", "y_test.npy", "label_map.json"]
    missing = [name for name in required if not (args.processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError("缺少测试数据：" + ", ".join(missing))
    if not args.model_path.exists():
        raise FileNotFoundError(f"缺少模型文件：{args.model_path}")

    X_test = np.load(args.processed_dir / "X_test.npy")
    y_test = np.load(args.processed_dir / "y_test.npy")
    if len(y_test) == 0:
        raise RuntimeError("测试集为空，无法评估。")
    mapping = load_label_map(args.processed_dir / "label_map.json")
    id_to_name = label_id_to_name(mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(args.model_path, device)
    model = SkeletonTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(6)))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(6)),
        target_names=[id_to_name.get(i, LABELS[i]) for i in range(6)],
        zero_division=0,
    )

    plot_confusion_matrix(cm, args.out_dir / "confusion_matrix.png")
    (args.out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    summary = [
        "Lab13 测试集评估总结",
        "=" * 24,
        f"model_path: {args.model_path}",
        f"test_samples: {len(y_test)}",
        f"accuracy: {accuracy:.4f}",
        "",
        "classification report:",
        report,
    ]
    (args.out_dir / "evaluation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    with (args.out_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["index", "true_id", "true_name", "pred_id", "pred_name", "confidence"]
        fieldnames += [f"prob_{i}_{LABELS[i].replace(' ', '_')}" for i in range(6)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (true_id, pred_id, probs) in enumerate(zip(y_test, y_pred, probabilities)):
            row = {
                "index": i,
                "true_id": int(true_id),
                "true_name": id_to_name.get(int(true_id), str(true_id)),
                "pred_id": int(pred_id),
                "pred_name": id_to_name.get(int(pred_id), str(pred_id)),
                "confidence": f"{float(probs[pred_id]):.6f}",
            }
            for class_id in range(6):
                row[f"prob_{class_id}_{LABELS[class_id].replace(' ', '_')}"] = f"{float(probs[class_id]):.6f}"
            writer.writerow(row)

    print("\n".join(summary))


if __name__ == "__main__":
    main()
