import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task107")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"
REPO_ROOT = TASK_DIR.parents[1]

MODEL_CANDIDATES = [
    REPO_ROOT / "lab10" / "task102" / "outputs" / "model_sgd_momentum.pth",
    REPO_ROOT / "lab10" / "task103" / "outputs" / "model_adam_lr_0_001.pth",
    REPO_ROOT / "lab10" / "task101" / "outputs" / "model_task101.pth",
]

TASK101_DATA_ROOT = REPO_ROOT / "lab10" / "task101" / "data"
TASK102_DATA_ROOT = REPO_ROOT / "lab10" / "task102" / "data"
TASK103_DATA_ROOT = REPO_ROOT / "lab10" / "task103" / "data"
LAB09_DATA_ROOT = REPO_ROOT / "lab09" / "task95_to_97" / "data"
LOCAL_DATA_ROOT = TASK_DIR / "data"

CLASS_LABELS = list(range(10))


# 与前面任务中的 SimpleCNN 完全一致，不能修改模型结构。
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


def choose_model_path():
    """按任务要求依次选择可用的模型权重文件。"""
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path
    raise FileNotFoundError("没有找到可用的模型权重文件。")


def choose_data_root():
    """选择 MNIST 数据目录。"""
    env_root = os.environ.get("MNIST_DATA_ROOT")
    if env_root:
        return Path(env_root), True

    for data_root in [TASK101_DATA_ROOT, TASK102_DATA_ROOT, TASK103_DATA_ROOT, LAB09_DATA_ROOT, LOCAL_DATA_ROOT]:
        if (data_root / "MNIST").exists():
            return data_root, False

    return LOCAL_DATA_ROOT, True


def load_test_dataset():
    """加载 MNIST 测试集。"""
    data_root, download = choose_data_root()
    print(f"MNIST 数据目录: {data_root}")
    transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.MNIST(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform,
    )


def predict_test_dataset(model, test_dataset):
    """收集测试集所有真实标签和预测标签。"""
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for image, label in test_dataset:
            output = model(image.unsqueeze(0))
            predicted = int(output.argmax(dim=1).item())
            true_labels.append(int(label))
            pred_labels.append(predicted)

    return true_labels, pred_labels


def build_confusion_matrix(true_labels, pred_labels):
    """使用 numpy 计算 10x10 混淆矩阵。"""
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[true_label, pred_label] += 1
    return matrix


def normalize_confusion_matrix(matrix):
    """按真实类别归一化，即每一行除以该真实类别样本总数。"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def save_matrix_csv(matrix, csv_path, value_format=None):
    """保存混淆矩阵 CSV，行是真实类别，列是预测类别。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label\\predicted_label"] + CLASS_LABELS)
        for true_label, row in zip(CLASS_LABELS, matrix):
            if value_format:
                writer.writerow([true_label] + [value_format.format(value) for value in row])
            else:
                writer.writerow([true_label] + [int(value) for value in row])


def plot_confusion_matrix(matrix, output_path, title, normalized=False):
    """绘制带数字标注的混淆矩阵图片。"""
    plt.figure(figsize=(8, 7))
    cmap = "Blues"
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(CLASS_LABELS)
    plt.yticks(CLASS_LABELS)
    plt.colorbar()

    threshold = matrix.max() * 0.55 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalized else str(int(value))
            color = "white" if value > threshold else "black"
            plt.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_confusion_pairs_summary(matrix):
    """统计所有非对角线混淆组合，并按数量从大到小排序。"""
    rows = []
    for true_label in CLASS_LABELS:
        for pred_label in CLASS_LABELS:
            if true_label == pred_label:
                continue
            count = int(matrix[true_label, pred_label])
            if count > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                    }
                )
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows


def save_confusion_pairs_summary(rows, csv_path):
    """保存非对角线混淆组合统计。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["true_label", "predicted_label", "count"])
        writer.writeheader()
        writer.writerows(rows)


def get_worst_confusions(confusion_pairs):
    """找出数量最大的混淆组合；若并列，则全部返回。"""
    if not confusion_pairs:
        return []
    max_count = confusion_pairs[0]["count"]
    return [row for row in confusion_pairs if row["count"] == max_count]


def format_confusion_rows(rows):
    """格式化混淆组合，便于打印和写入分析文件。"""
    if not rows:
        return "无非对角线混淆"
    return "；".join(
        f"真实 {row['true_label']} 被预测为 {row['predicted_label']}：{row['count']} 次"
        for row in rows
    )


def write_analysis(model_path, total_count, accuracy, matrix, confusion_pairs):
    """生成中文混淆矩阵分析文件。"""
    worst_confusions = get_worst_confusions(confusion_pairs)
    worst_text = format_confusion_rows(worst_confusions)
    top_five_text = format_confusion_rows(confusion_pairs[:5])
    error_count = int(total_count - np.trace(matrix))

    lines = [
        "Lab 10 任务七：混淆矩阵分析",
        "",
        f"本任务加载的模型权重文件：{model_path}",
        f"测试集总样本数：{total_count}",
        f"测试准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"错误分类样本数：{error_count}",
        f"混淆矩阵大小：{matrix.shape[0]} x {matrix.shape[1]}",
        "",
        "对角线元素含义：",
        "混淆矩阵的对角线元素表示真实类别和预测类别相同的样本数量，即分类正确的样本数量。",
        "",
        "非对角线元素含义：",
        "非对角线元素表示真实类别和预测类别不同的样本数量，即分类错误或类别混淆的情况。",
        "",
        "最严重混淆类别组合：",
        worst_text,
        f"前几组混淆情况：{top_five_text}",
        "",
        "结合任务六的错误原因分析：",
        "这些混淆通常与手写数字形状相似、笔画模糊或断裂、书写倾斜、局部结构接近等因素有关。",
        "例如 5 和 3 都可能包含上部横线和下部弧形结构，4 和 9、9 和 3 也可能因为局部弧线或竖线结构接近而被混淆。",
        "",
        "改进方向：",
        "可以从数据增强、增加训练轮数、调整学习率、使用学习率衰减、优化模型结构等方面改进。",
        "例如加入旋转、平移、缩放增强样本，或在模型中增加卷积层、BatchNorm、Dropout，以提升模型泛化能力。",
    ]

    with open(OUTPUT_DIR / "confusion_matrix_analysis.txt", "w", encoding="utf-8") as analysis_file:
        analysis_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60)
    print("Lab 10 任务七：混淆矩阵")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path()
    print(f"使用的模型权重文件: {model_path}")

    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = load_test_dataset()
    true_labels, pred_labels = predict_test_dataset(model, test_dataset)

    matrix = build_confusion_matrix(true_labels, pred_labels)
    normalized_matrix = normalize_confusion_matrix(matrix)
    total_count = len(true_labels)
    correct_count = int(np.trace(matrix))
    accuracy = correct_count / total_count

    confusion_pairs = build_confusion_pairs_summary(matrix)
    worst_confusions = get_worst_confusions(confusion_pairs)

    save_matrix_csv(matrix, OUTPUT_DIR / "confusion_matrix.csv")
    save_matrix_csv(normalized_matrix, OUTPUT_DIR / "confusion_matrix_normalized.csv", value_format="{:.6f}")
    plot_confusion_matrix(matrix, OUTPUT_DIR / "confusion_matrix.png", "Confusion Matrix", normalized=False)
    plot_confusion_matrix(
        normalized_matrix,
        OUTPUT_DIR / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
        normalized=True,
    )
    save_confusion_pairs_summary(confusion_pairs, OUTPUT_DIR / "confusion_pairs_summary.csv")
    write_analysis(model_path, total_count, accuracy, matrix, confusion_pairs)

    print(f"测试集总样本数: {total_count}")
    print(f"测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"混淆矩阵形状: {matrix.shape}")
    print(f"最严重混淆类别组合: {format_confusion_rows(worst_confusions)}")
    print(f"原始混淆矩阵图片保存位置: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"归一化混淆矩阵图片保存位置: {OUTPUT_DIR / 'confusion_matrix_normalized.png'}")
    print(f"原始混淆矩阵 CSV 保存位置: {OUTPUT_DIR / 'confusion_matrix.csv'}")
    print(f"归一化混淆矩阵 CSV 保存位置: {OUTPUT_DIR / 'confusion_matrix_normalized.csv'}")
    print(f"混淆组合统计 CSV 保存位置: {OUTPUT_DIR / 'confusion_pairs_summary.csv'}")
    print(f"混淆矩阵分析文件保存位置: {OUTPUT_DIR / 'confusion_matrix_analysis.txt'}")


if __name__ == "__main__":
    main()
