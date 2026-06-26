import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task101")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# 当前任务目录：lab10/task101
TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"

# 项目根目录：/home/david/cv-course
REPO_ROOT = TASK_DIR.parents[1]

# 优先复用 lab09 已经下载好的 MNIST 数据，避免课堂运行时重复下载。
LAB09_DATA_ROOT = REPO_ROOT / "lab09" / "task95_to_97" / "data"
LOCAL_DATA_ROOT = TASK_DIR / "data"


# 模型结构必须与 lab09 中的 SimpleCNN 保持一致，不新增或修改网络层。
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

    if (LOCAL_DATA_ROOT / "MNIST").exists():
        return LOCAL_DATA_ROOT, False

    if (LAB09_DATA_ROOT / "MNIST").exists():
        return LAB09_DATA_ROOT, False

    return LOCAL_DATA_ROOT, True


def evaluate(model, data_loader, criterion, device):
    """在验证集或测试集上计算平均 loss 和 accuracy。"""
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

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_history_csv(history, csv_path):
    """保存每个 epoch 的训练记录。"""
    fieldnames = [
        "epoch",
        "training_loss",
        "validation_loss",
        "training_accuracy",
        "validation_accuracy",
        "test_accuracy",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history):
    """保存 loss 曲线图和 accuracy 曲线图。"""
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [item["training_loss"] for item in history], marker="o", label="Training Loss")
    plt.plot(epochs, [item["validation_loss"] for item in history], marker="x", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [item["training_accuracy"] for item in history], marker="o", label="Training Accuracy")
    plt.plot(epochs, [item["validation_accuracy"] for item in history], marker="x", label="Validation Accuracy")
    plt.plot(epochs, [item["test_accuracy"] for item in history], marker="s", label="Test Accuracy")
    plt.title("Training, Validation and Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("Lab 10 任务一：复用上次 CNN 模型")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    data_root, download = choose_data_root()
    print(f"MNIST 数据目录: {data_root}")

    # 与 lab09 保持一致：只把图像转换为 Tensor。
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

    # 按 lab09 的做法，将 60000 张训练图像拆分为 50000 训练集和 10000 验证集。
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    history = []

    print(f"\n开始训练，共 {epochs} 个 epoch。")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
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

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        training_loss = running_loss / len(train_loader)
        training_accuracy = correct_train / total_train
        validation_loss, validation_accuracy = evaluate(model, val_loader, criterion, device)
        _, test_accuracy = evaluate(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {training_loss:.4f}, Train Acc: {training_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    final_test_loss, final_test_accuracy = evaluate(model, test_loader, criterion, device)

    save_history_csv(history, OUTPUT_DIR / "training_history.csv")
    plot_curves(history)
    torch.save(model.state_dict(), OUTPUT_DIR / "model_task101.pth")

    result_text = (
        "Lab 10 任务一测试结果\n"
        "数据集: MNIST\n"
        "复用模型: lab09 SimpleCNN\n"
        f"训练轮数: {epochs}\n"
        f"Final Test Loss: {final_test_loss:.4f}\n"
        f"Final Test Accuracy: {final_test_accuracy:.4f}\n"
        f"Final Test Accuracy (%): {final_test_accuracy * 100:.2f}%\n"
    )
    with open(OUTPUT_DIR / "test_result.txt", "w", encoding="utf-8") as result_file:
        result_file.write(result_text)

    print("\n训练完成，输出文件已保存到 outputs 文件夹。")
    print(f"最终 Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
