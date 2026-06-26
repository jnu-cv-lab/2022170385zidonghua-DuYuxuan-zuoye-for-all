from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skeleton_transformer import SkeletonTransformer
from utils import LABELS, load_label_map


class SkeletonDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Skeleton Transformer 羽毛球动作分类模型。")
    parser.add_argument("--processed_dir", type=Path, default=LAB13_ROOT / "data" / "processed")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def load_processed_data(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "label_map.json"]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "缺少预处理输出文件："
            + ", ".join(missing)
            + "\n请先运行：python lab13/task132_preprocess_skeleton/preprocess_skeleton.py --demo_fast"
        )
    return (
        np.load(processed_dir / "X_train.npy"),
        np.load(processed_dir / "y_train.npy"),
        np.load(processed_dir / "X_test.npy"),
        np.load(processed_dir / "y_test.npy"),
        load_label_map(processed_dir / "label_map.json"),
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        total_count += len(y_batch)

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def save_training_curve(history: List[Dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["test_acc"] for row in history], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(__file__).resolve().parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, mapping = load_processed_data(args.processed_dir)
    if len(y_train) == 0:
        raise RuntimeError("训练集为空，无法训练模型。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SkeletonDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SkeletonDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model_config = {
        "input_dim": int(X_train.shape[2]),
        "target_frames": int(X_train.shape[1]),
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "num_classes": 6,
        "dropout": 0.1,
    }
    model = SkeletonTransformer(**model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    best_acc = -1.0
    best_epoch = 0

    print(f"device={device}, train={len(y_train)}, test={len(y_test)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "label_map": mapping,
            "epoch": epoch,
            "test_acc": test_acc,
        }
        torch.save(checkpoint, args.out_dir / "last_model.pth")
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(checkpoint, args.out_dir / "best_model.pth")

    with (args.out_dir / "training_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerows(history)
    save_training_curve(history, args.out_dir / "training_curve.png")

    summary_lines = [
        "Lab13 Skeleton Transformer 训练总结",
        "=" * 34,
        f"device: {device}",
        f"X_train shape: {list(X_train.shape)}",
        f"X_test shape: {list(X_test.shape)}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"learning_rate: {args.lr}",
        f"best_epoch: {best_epoch}",
        f"best_test_acc: {best_acc:.4f}",
        f"classes: {', '.join(LABELS)}",
    ]
    (args.out_dir / "train_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    notes = [
        "# Task133 Skeleton Transformer 训练说明",
        "",
        "## 模型结构",
        "输入 X 为 [B,T,132]，先经过 Linear Embedding 将 132 维骨架特征映射到 d_model=128，再加入可学习位置编码，经过 2 层 Transformer Encoder，最后 Mean Pooling 并输入 MLP 分类器输出 6 类 logits。",
        "",
        "## Transformer Encoder 作用",
        "Transformer Encoder 通过自注意力建模不同时间帧之间的关系，适合从击球动作的骨架序列中学习时序模式。",
        "",
        "## Mean Pooling 的作用",
        "Mean Pooling 将 T 个时间步的特征平均为一个视频级表示，用于最终分类。",
        "",
        "## 损失函数和优化器",
        f"损失函数为 CrossEntropyLoss，优化器为 Adam，学习率为 {args.lr}。",
        "",
        "## 最佳测试准确率",
        f"best_epoch={best_epoch}, best_test_acc={best_acc:.4f}",
    ]
    (docs_dir / "task133_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
