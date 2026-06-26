import copy
import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task102")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"
REPO_ROOT = TASK_DIR.parents[1]

# 优先复用已有 MNIST 数据，避免重复下载。
TASK101_DATA_ROOT = REPO_ROOT / "lab10" / "task101" / "data"
LAB09_DATA_ROOT = REPO_ROOT / "lab09" / "task95_to_97" / "data"
LOCAL_DATA_ROOT = TASK_DIR / "data"

BATCH_SIZE = 64
EPOCHS = 5
TRAIN_SIZE = 50000
SPLIT_SEED = 42
INIT_SEED = 42
LOADER_SEED = 2026


# 与 lab09/task93、lab10/task101 中的 SimpleCNN 完全一致。
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def choose_data_root():
    """选择 MNIST 数据目录。"""
    env_root = os.environ.get("MNIST_DATA_ROOT")
    if env_root:
        return Path(env_root), True

    if (TASK101_DATA_ROOT / "MNIST").exists():
        return TASK101_DATA_ROOT, False

    if (LAB09_DATA_ROOT / "MNIST").exists():
        return LAB09_DATA_ROOT, False

    if (LOCAL_DATA_ROOT / "MNIST").exists():
        return LOCAL_DATA_ROOT, False

    return LOCAL_DATA_ROOT, True


def load_datasets():
    """加载 MNIST，并按任务一方式划分训练集和验证集。"""
    data_root, download = choose_data_root()
    print(f"MNIST 数据目录: {data_root}")

    # 与 lab09 和 task101 保持一致：只转换为 Tensor。
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform,
    )

    val_size = len(full_train_dataset) - TRAIN_SIZE
    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [TRAIN_SIZE, val_size],
        generator=split_generator,
    )
    return train_dataset, val_dataset, test_dataset


def build_loaders(train_dataset, val_dataset, test_dataset):
    """每个优化器使用同样的 shuffle 随机种子，保证训练数据顺序一致。"""
    train_generator = torch.Generator().manual_seed(LOADER_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, criterion, device):
    """计算指定数据集上的平均 loss 和 accuracy。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def make_optimizer(optimizer_name, model):
    """根据名称创建优化器。"""
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=0.01)
    if optimizer_name == "SGD + Momentum":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001)
    raise ValueError(f"未知优化器: {optimizer_name}")


def train_one_optimizer(
    optimizer_name,
    model_file,
    initial_state,
    train_dataset,
    val_dataset,
    test_dataset,
    criterion,
    device,
):
    """使用一种优化器完成 5 个 epoch 的训练，并返回训练记录和最终结果。"""
    train_loader, val_loader, test_loader = build_loaders(train_dataset, val_dataset, test_dataset)

    model = SimpleCNN().to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = make_optimizer(optimizer_name, model)

    history = []
    print("-" * 60)
    print(f"开始训练优化器: {optimizer_name}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_size
            correct_train += (predicted == labels).sum().item()

        train_loss = total_loss / total_train
        train_acc = correct_train / total_train
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)

        row = {
            "optimizer": optimizer_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"{optimizer_name} | Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), OUTPUT_DIR / model_file)

    final_row = history[-1]
    summary = {
        "optimizer": optimizer_name,
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "final_train_acc": final_row["train_acc"],
        "final_val_acc": final_row["val_acc"],
        "test_acc": final_test_acc,
        "test_loss": final_test_loss,
    }

    return history, summary


def save_csv(rows, csv_path, fieldnames):
    """保存 CSV 文件。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_comparison(history):
    """绘制三种优化器的 loss 曲线。"""
    plt.figure(figsize=(9, 6))
    for optimizer_name in ["SGD", "SGD + Momentum", "Adam"]:
        rows = [row for row in history if row["optimizer"] == optimizer_name]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_loss"] for row in rows], linestyle="--", marker="o", label=f"{optimizer_name} Train")
        plt.plot(epochs, [row["val_loss"] for row in rows], linestyle="-", marker="x", label=f"{optimizer_name} Val")

    plt.title("Optimizer Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_comparison.png", dpi=300)
    plt.close()


def plot_accuracy_comparison(history):
    """绘制三种优化器的 accuracy 曲线。"""
    plt.figure(figsize=(9, 6))
    for optimizer_name in ["SGD", "SGD + Momentum", "Adam"]:
        rows = [row for row in history if row["optimizer"] == optimizer_name]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_acc"] for row in rows], linestyle="--", marker="o", label=f"{optimizer_name} Train")
        plt.plot(epochs, [row["val_acc"] for row in rows], linestyle="-", marker="x", label=f"{optimizer_name} Val")

    plt.title("Optimizer Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()


def plot_test_accuracy(summary):
    """绘制最终 test accuracy 柱状图。"""
    names = [row["optimizer"] for row in summary]
    accuracies = [row["test_acc"] for row in summary]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, accuracies, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(max(0.0, min(accuracies) - 0.02), 1.0)
    plt.title("Final Test Accuracy Comparison")
    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc,
            f"{acc * 100:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_accuracy_comparison.png", dpi=300)
    plt.close()


def write_test_result(summary):
    """保存中文测试结果说明。"""
    best = max(summary, key=lambda row: row["test_acc"])
    lines = [
        "Lab 10 任务二：优化器对比测试结果",
        "数据集：MNIST",
        "模型：复用 lab09/task101 的 SimpleCNN，模型结构未改变",
        f"训练轮数：每种优化器 {EPOCHS} 个 epoch",
        "",
    ]

    for row in summary:
        lines.append(
            f"{row['optimizer']} 最终 Test Accuracy: "
            f"{row['test_acc']:.4f} ({row['test_acc'] * 100:.2f}%)"
        )

    lines.extend(
        [
            "",
            f"表现最好的优化器：{best['optimizer']}",
            f"最高 Test Accuracy: {best['test_acc']:.4f} ({best['test_acc'] * 100:.2f}%)",
        ]
    )

    with open(OUTPUT_DIR / "test_result.txt", "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60)
    print("Lab 10 任务二：优化器对比")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    train_dataset, val_dataset, test_dataset = load_datasets()
    criterion = nn.CrossEntropyLoss()

    # 三种优化器从同一组初始权重开始训练，保证对比更公平。
    torch.manual_seed(INIT_SEED)
    initial_model = SimpleCNN()
    initial_state = copy.deepcopy(initial_model.state_dict())

    optimizer_jobs = [
        ("SGD", "model_sgd.pth"),
        ("SGD + Momentum", "model_sgd_momentum.pth"),
        ("Adam", "model_adam.pth"),
    ]

    all_history = []
    all_summary = []
    for optimizer_name, model_file in optimizer_jobs:
        history, summary = train_one_optimizer(
            optimizer_name,
            model_file,
            initial_state,
            train_dataset,
            val_dataset,
            test_dataset,
            criterion,
            device,
        )
        all_history.extend(history)
        all_summary.append(summary)

    save_csv(
        all_history,
        OUTPUT_DIR / "optimizer_history.csv",
        ["optimizer", "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "test_acc"],
    )
    save_csv(
        all_summary,
        OUTPUT_DIR / "optimizer_summary.csv",
        [
            "optimizer",
            "final_train_loss",
            "final_val_loss",
            "final_train_acc",
            "final_val_acc",
            "test_acc",
            "test_loss",
        ],
    )

    plot_loss_comparison(all_history)
    plot_accuracy_comparison(all_history)
    plot_test_accuracy(all_summary)
    write_test_result(all_summary)

    best = max(all_summary, key=lambda row: row["test_acc"])
    print("-" * 60)
    print("三种优化器最终测试结果：")
    for row in all_summary:
        print(f"{row['optimizer']}: Test Accuracy = {row['test_acc']:.4f} ({row['test_acc'] * 100:.2f}%)")
    print(f"表现最好的优化器: {best['optimizer']}")
    print("所有结果已保存到 outputs 文件夹。")


if __name__ == "__main__":
    main()
