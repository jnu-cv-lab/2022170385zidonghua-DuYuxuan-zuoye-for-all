import copy
import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task103")

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

# 优先复用已经下载好的 MNIST 数据，避免重复下载。
TASK101_DATA_ROOT = REPO_ROOT / "lab10" / "task101" / "data"
TASK102_DATA_ROOT = REPO_ROOT / "lab10" / "task102" / "data"
LAB09_DATA_ROOT = REPO_ROOT / "lab09" / "task95_to_97" / "data"
LOCAL_DATA_ROOT = TASK_DIR / "data"

BATCH_SIZE = 64
EPOCHS = 5
TRAIN_SIZE = 50000
SPLIT_SEED = 42
INIT_SEED = 42
LOADER_SEED = 2026


# 与 lab09、task101、task102 中的 SimpleCNN 完全一致，不改变模型结构。
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

    for data_root in [TASK101_DATA_ROOT, TASK102_DATA_ROOT, LAB09_DATA_ROOT, LOCAL_DATA_ROOT]:
        if (data_root / "MNIST").exists():
            return data_root, False

    return LOCAL_DATA_ROOT, True


def load_datasets():
    """加载 MNIST，并按任务一、任务二的方式划分训练集和验证集。"""
    data_root, download = choose_data_root()
    print(f"MNIST 数据目录: {data_root}", flush=True)

    # 与前两个任务保持一致：只将图像转换为 Tensor。
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
    """每个学习率使用同样的数据顺序，保证对比公平。"""
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


def lr_label(learning_rate):
    """生成适合文件名和图例使用的学习率标签。"""
    return f"{learning_rate:g}"


def train_one_learning_rate(
    learning_rate,
    model_file,
    initial_state,
    train_dataset,
    val_dataset,
    test_dataset,
    criterion,
    device,
):
    """使用一个学习率训练 Adam 模型，并返回训练记录和最终结果。"""
    train_loader, val_loader, test_loader = build_loaders(train_dataset, val_dataset, test_dataset)

    model = SimpleCNN().to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    print("-" * 60, flush=True)
    print(f"开始训练 Adam，learning rate = {learning_rate}", flush=True)

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
            "learning_rate": lr_label(learning_rate),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"Adam lr={learning_rate:g} | Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}",
            flush=True,
        )

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), OUTPUT_DIR / model_file)

    final_row = history[-1]
    summary = {
        "learning_rate": lr_label(learning_rate),
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
    """绘制三个学习率的 loss 曲线。"""
    plt.figure(figsize=(9, 6))
    for learning_rate in ["0.1", "0.01", "0.001"]:
        rows = [row for row in history if row["learning_rate"] == learning_rate]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_loss"] for row in rows], linestyle="--", marker="o", label=f"lr={learning_rate} Train")
        plt.plot(epochs, [row["val_loss"] for row in rows], linestyle="-", marker="x", label=f"lr={learning_rate} Val")

    plt.title("Adam Learning Rate Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_lr_comparison.png", dpi=300)
    plt.close()


def plot_accuracy_comparison(history):
    """绘制三个学习率的 accuracy 曲线。"""
    plt.figure(figsize=(9, 6))
    for learning_rate in ["0.1", "0.01", "0.001"]:
        rows = [row for row in history if row["learning_rate"] == learning_rate]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_acc"] for row in rows], linestyle="--", marker="o", label=f"lr={learning_rate} Train")
        plt.plot(epochs, [row["val_acc"] for row in rows], linestyle="-", marker="x", label=f"lr={learning_rate} Val")

    plt.title("Adam Learning Rate Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_lr_comparison.png", dpi=300)
    plt.close()


def plot_test_accuracy(summary):
    """绘制最终 test accuracy 柱状图。"""
    labels = [f"lr={row['learning_rate']}" for row in summary]
    accuracies = [row["test_acc"] for row in summary]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(max(0.0, min(accuracies) - 0.05), 1.0)
    plt.title("Adam Learning Rate Test Accuracy Comparison")
    plt.xlabel("Learning Rate")
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
    plt.savefig(OUTPUT_DIR / "test_accuracy_lr_comparison.png", dpi=300)
    plt.close()


def write_test_result(summary):
    """保存中文测试结果说明。"""
    best = max(summary, key=lambda row: row["test_acc"])
    lines = [
        "Lab 10 任务三：学习率对比测试结果",
        "数据集：MNIST",
        "模型：复用 lab09/task101/task102 的 SimpleCNN，模型结构未改变",
        "优化器：Adam",
        f"训练轮数：每个学习率 {EPOCHS} 个 epoch",
        "",
    ]

    for row in summary:
        lines.append(
            f"learning rate = {row['learning_rate']} 最终 Test Accuracy: "
            f"{row['test_acc']:.4f} ({row['test_acc'] * 100:.2f}%)"
        )

    lines.extend(
        [
            "",
            f"表现最好的学习率：{best['learning_rate']}",
            f"最高 Test Accuracy: {best['test_acc']:.4f} ({best['test_acc'] * 100:.2f}%)",
            "",
            "简单分析：",
            "学习率过大可能导致 loss 震荡或训练不稳定，模型难以收敛到较好结果。",
            "学习率过小可能导致收敛较慢，在较少 epoch 内准确率提升有限。",
            "合适的学习率通常能获得更稳定的训练过程和更高的准确率。",
        ]
    )

    with open(OUTPUT_DIR / "test_result.txt", "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60, flush=True)
    print("Lab 10 任务三：学习率对比", flush=True)
    print("=" * 60, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}", flush=True)

    train_dataset, val_dataset, test_dataset = load_datasets()
    criterion = nn.CrossEntropyLoss()

    # 三组实验从同一组初始权重开始训练，不在上一组结果基础上继续训练。
    torch.manual_seed(INIT_SEED)
    initial_model = SimpleCNN()
    initial_state = copy.deepcopy(initial_model.state_dict())

    lr_jobs = [
        (0.1, "model_adam_lr_0_1.pth"),
        (0.01, "model_adam_lr_0_01.pth"),
        (0.001, "model_adam_lr_0_001.pth"),
    ]

    all_history = []
    all_summary = []
    for learning_rate, model_file in lr_jobs:
        history, summary = train_one_learning_rate(
            learning_rate,
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
        OUTPUT_DIR / "lr_history.csv",
        ["learning_rate", "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "test_acc"],
    )
    save_csv(
        all_summary,
        OUTPUT_DIR / "lr_summary.csv",
        [
            "learning_rate",
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
    print("-" * 60, flush=True)
    print("三个学习率最终测试结果：", flush=True)
    for row in all_summary:
        print(
            f"learning rate = {row['learning_rate']}: "
            f"Test Accuracy = {row['test_acc']:.4f} ({row['test_acc'] * 100:.2f}%)",
            flush=True,
        )
    print(f"表现最好的学习率: {best['learning_rate']}", flush=True)
    print("所有结果已保存到 outputs 文件夹。", flush=True)


if __name__ == "__main__":
    main()
