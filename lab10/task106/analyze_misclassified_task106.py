import csv
import os
from collections import Counter
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task106")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# 与 lab09、task101、task102、task103、task104、task105 中的 SimpleCNN 完全一致。
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


def find_misclassified_samples(model, test_dataset):
    """遍历测试集，找出所有预测错误的样本。"""
    model.eval()
    misclassified = []
    correct_count = 0

    with torch.no_grad():
        for sample_index, (image, label) in enumerate(test_dataset):
            output = model(image.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            true_label = int(label)
            predicted_label = int(predicted.item())
            confidence_value = float(confidence.item())

            if predicted_label == true_label:
                correct_count += 1
            else:
                misclassified.append(
                    {
                        "sample_index": sample_index,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence_value,
                        "image": image,
                    }
                )

    total_count = len(test_dataset)
    accuracy = correct_count / total_count
    return misclassified, total_count, accuracy


def save_misclassified_grid(samples, output_path, cmap, title):
    """保存错误分类样本图片网格。"""
    cols = 4
    rows = (len(samples) + cols - 1) // cols

    plt.figure(figsize=(cols * 2.8, rows * 2.8))
    for index, sample in enumerate(samples):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(sample["image"].squeeze(0), cmap=cmap)
        plt.title(f"True: {sample['true_label']} | Pred: {sample['predicted_label']}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_misclassified_details(misclassified, csv_path):
    """保存所有错误分类样本的详细信息。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["sample_index", "true_label", "predicted_label", "confidence"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in misclassified:
            writer.writerow(
                {
                    "sample_index": sample["sample_index"],
                    "true_label": sample["true_label"],
                    "predicted_label": sample["predicted_label"],
                    "confidence": sample["confidence"],
                }
            )


def save_class_error_summary(misclassified, csv_path):
    """统计每个真实类别被错误分类的数量。"""
    class_counter = Counter(sample["true_label"] for sample in misclassified)
    rows = [{"true_label": label, "error_count": class_counter.get(label, 0)} for label in range(10)]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["true_label", "error_count"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def save_confused_pairs_summary(misclassified, csv_path):
    """统计真实类别和预测类别的混淆组合。"""
    pair_counter = Counter((sample["true_label"], sample["predicted_label"]) for sample in misclassified)
    rows = [
        {"true_label": true_label, "predicted_label": predicted_label, "count": count}
        for (true_label, predicted_label), count in pair_counter.most_common()
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["true_label", "predicted_label", "count"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def format_top_confusions(confused_pairs, top_n=3):
    """生成最容易混淆类别组合的中文描述。"""
    if not confused_pairs:
        return "无错误分类样本"

    parts = []
    for row in confused_pairs[:top_n]:
        parts.append(f"{row['true_label']} 被预测为 {row['predicted_label']}：{row['count']} 次")
    return "；".join(parts)


def write_error_analysis(model_path, total_count, error_count, accuracy, shown_count, class_summary, confused_pairs):
    """生成中文错误分类分析文件。"""
    top_class = max(class_summary, key=lambda row: row["error_count"])
    top_confusions = format_top_confusions(confused_pairs, top_n=5)

    if confused_pairs:
        easiest_pair = confused_pairs[0]
        easiest_pair_text = (
            f"{easiest_pair['true_label']} 被预测为 {easiest_pair['predicted_label']}，"
            f"共 {easiest_pair['count']} 次"
        )
    else:
        easiest_pair_text = "无错误分类样本"

    lines = [
        "Lab 10 任务六：错误分类样本分析",
        "",
        f"本任务加载的模型权重文件：{model_path}",
        f"测试集总样本数：{total_count}",
        f"错误分类数量：{error_count}",
        f"测试准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"本次显示的错误分类图片数量：{shown_count}",
        "",
        "最容易混淆的类别：",
        f"错误数量最多的真实类别是 {top_class['true_label']}，错误数量为 {top_class['error_count']}。",
        f"最常见的混淆组合是：{easiest_pair_text}。",
        f"前几组混淆情况：{top_confusions}。",
        "",
        "错误分类可能原因：",
        "1. 手写数字形状相似，例如部分 4、7、9 的局部结构可能接近。",
        "2. 部分样本笔画模糊、断裂或过细，导致关键结构不明显。",
        "3. 字体倾斜或书写习惯差异较大，模型提取到的局部特征容易偏离典型数字形态。",
        "4. 一些数字的局部结构接近，例如圆弧、竖线、斜线组合相似。",
        "5. 图像中有效笔画区域较小或位置偏移，可能使卷积特征响应不够稳定。",
        "",
        "提高准确率的改进方向：",
        "1. 数据方面：增加数据量，使用数据增强，加入旋转、平移、缩放等样本。",
        "2. 模型结构方面：可以增加卷积层，使用 BatchNorm、Dropout 等方法提升表达能力和泛化能力。",
        "3. 训练方法方面：调整学习率，增加训练轮数，使用更合适的优化器或学习率衰减策略。",
    ]

    with open(OUTPUT_DIR / "error_analysis.txt", "w", encoding="utf-8") as analysis_file:
        analysis_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60)
    print("Lab 10 任务六：错误分类样本分析")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path()
    print(f"使用的模型权重文件: {model_path}")

    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = load_test_dataset()
    misclassified, total_count, accuracy = find_misclassified_samples(model, test_dataset)
    error_count = len(misclassified)

    if error_count < 8:
        raise RuntimeError("错误分类样本数量少于 8 张，不满足任务要求。")

    shown_count = min(16, error_count)
    shown_samples = misclassified[:shown_count]

    save_misclassified_grid(
        shown_samples,
        OUTPUT_DIR / "misclassified_samples.png",
        cmap="viridis",
        title="Misclassified Samples",
    )
    save_misclassified_grid(
        shown_samples,
        OUTPUT_DIR / "misclassified_samples_gray.png",
        cmap="gray",
        title="Misclassified Samples Gray",
    )
    save_misclassified_details(misclassified, OUTPUT_DIR / "misclassified_details.csv")
    class_summary = save_class_error_summary(misclassified, OUTPUT_DIR / "class_error_summary.csv")
    confused_pairs = save_confused_pairs_summary(misclassified, OUTPUT_DIR / "confused_pairs_summary.csv")
    write_error_analysis(model_path, total_count, error_count, accuracy, shown_count, class_summary, confused_pairs)

    top_confusion = confused_pairs[0] if confused_pairs else None
    top_confusion_text = (
        f"{top_confusion['true_label']} -> {top_confusion['predicted_label']} ({top_confusion['count']} 次)"
        if top_confusion
        else "无"
    )

    print(f"测试集总样本数: {total_count}")
    print(f"错误分类样本数量: {error_count}")
    print(f"测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"实际显示错误分类图片数量: {shown_count}")
    print(f"最容易被混淆的类别组合: {top_confusion_text}")
    print(f"彩色错误样本图片保存位置: {OUTPUT_DIR / 'misclassified_samples.png'}")
    print(f"灰度错误样本图片保存位置: {OUTPUT_DIR / 'misclassified_samples_gray.png'}")
    print(f"错误样本详情 CSV 保存位置: {OUTPUT_DIR / 'misclassified_details.csv'}")
    print(f"类别错误统计 CSV 保存位置: {OUTPUT_DIR / 'class_error_summary.csv'}")
    print(f"混淆类别组合 CSV 保存位置: {OUTPUT_DIR / 'confused_pairs_summary.csv'}")
    print(f"错误分析文件保存位置: {OUTPUT_DIR / 'error_analysis.txt'}")


if __name__ == "__main__":
    main()
