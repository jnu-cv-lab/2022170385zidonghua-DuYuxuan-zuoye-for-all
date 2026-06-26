import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task104")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"
REPO_ROOT = TASK_DIR.parents[1]

MODEL_CANDIDATES = [
    REPO_ROOT / "lab10" / "task102" / "outputs" / "model_sgd_momentum.pth",
    REPO_ROOT / "lab10" / "task103" / "outputs" / "model_adam_lr_0_001.pth",
    REPO_ROOT / "lab10" / "task101" / "outputs" / "model_task101.pth",
]


# 与 lab09、task101、task102、task103 中的 SimpleCNN 完全一致。
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


def find_first_conv_layer(model):
    """自动找到模型中的第一层卷积层。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise RuntimeError("模型中没有找到 Conv2d 卷积层。")


def normalize_kernel(kernel):
    """将卷积核数值归一化到 0 到 1，便于显示。"""
    min_value = kernel.min()
    max_value = kernel.max()
    if max_value - min_value < 1e-12:
        return torch.zeros_like(kernel)
    return (kernel - min_value) / (max_value - min_value)


def save_kernel_grid(kernels, output_path, cmap, title):
    """将多个二维卷积核保存为网格图片。"""
    num_kernels = kernels.shape[0]
    cols = 4
    rows = (num_kernels + cols - 1) // cols

    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for index in range(num_kernels):
        kernel = kernels[index]
        plt.subplot(rows, cols, index + 1)
        plt.imshow(normalize_kernel(kernel), cmap=cmap, interpolation="nearest")
        plt.title(f"Kernel {index + 1}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_kernel_summary(kernels, csv_path):
    """保存每个卷积核的简单统计信息。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["kernel_index", "min_value", "max_value", "mean_value", "std_value"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, kernel in enumerate(kernels, start=1):
            writer.writerow(
                {
                    "kernel_index": index,
                    "min_value": float(kernel.min().item()),
                    "max_value": float(kernel.max().item()),
                    "mean_value": float(kernel.mean().item()),
                    "std_value": float(kernel.std(unbiased=False).item()),
                }
            )


def infer_kernel_features(kernels):
    """根据卷积核数值给出简短的可视化观察。"""
    positive_negative_count = 0
    direction_like_count = 0

    for kernel in kernels:
        has_positive = bool((kernel > 0).any().item())
        has_negative = bool((kernel < 0).any().item())
        if has_positive and has_negative:
            positive_negative_count += 1

        row_change = torch.mean(torch.abs(kernel[1:, :] - kernel[:-1, :])).item()
        col_change = torch.mean(torch.abs(kernel[:, 1:] - kernel[:, :-1])).item()
        if max(row_change, col_change) > min(row_change, col_change) * 1.2:
            direction_like_count += 1

    return positive_negative_count, direction_like_count


def write_analysis(model_path, conv_name, weight_shape, shown_kernels, kernels):
    """生成中文卷积核分析文件。"""
    out_channels, in_channels, kernel_height, kernel_width = weight_shape
    positive_negative_count, direction_like_count = infer_kernel_features(kernels)

    lines = [
        "Lab 10 任务四：卷积核可视化分析",
        "",
        f"本任务加载的模型权重文件：{model_path}",
        f"自动找到的第一层卷积层：{conv_name}",
        f"第一层卷积核权重形状：{tuple(weight_shape)}",
        f"第一层卷积核数量：{out_channels}",
        f"每个卷积核尺寸：{in_channels} x {kernel_height} x {kernel_width}",
        f"本次实际显示的卷积核数量：{shown_kernels}",
        "",
        "可视化结果观察：",
        f"显示的 {shown_kernels} 个卷积核中，有 {positive_negative_count} 个同时包含正值和负值，这类卷积核通常会对局部灰度变化更敏感。",
        f"根据行方向和列方向数值变化的粗略统计，有 {direction_like_count} 个卷积核呈现一定方向差异，可能对应水平、垂直或斜向边缘响应。",
        "不同卷积核之间的数值分布、正负区域和变化方向不同，因此它们会对 MNIST 图像中的不同局部笔画、边缘方向或小纹理模式产生不同响应。",
        "",
        "卷积核如何得到：",
        "这些卷积核不是人工手动设计的，而是在数字分类任务训练过程中自动学习得到的。",
        "训练时，模型先根据输入图像进行前向传播并计算分类损失，然后通过反向传播计算损失对卷积核权重的梯度。",
        "优化器根据梯度不断更新卷积核参数，使模型逐渐学到有助于区分 0 到 9 数字类别的局部特征。",
    ]

    with open(OUTPUT_DIR / "kernel_analysis.txt", "w", encoding="utf-8") as analysis_file:
        analysis_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60)
    print("Lab 10 任务四：卷积核可视化")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path()
    print(f"使用的模型权重文件: {model_path}")

    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    conv_name, conv_layer = find_first_conv_layer(model)
    weights = conv_layer.weight.detach().cpu()
    print(f"第一层卷积层名称: {conv_name}")
    print(f"第一层卷积核权重形状: {tuple(weights.shape)}")

    # MNIST 是单通道灰度图，所以第一层卷积核可直接转为二维图像显示。
    max_show = min(16, weights.shape[0])
    if max_show < 8:
        raise RuntimeError("第一层卷积核数量少于 8 个，不满足任务要求。")

    kernels_2d = weights[:max_show, 0, :, :]
    print(f"实际可视化卷积核数量: {max_show}")

    save_kernel_grid(
        kernels_2d,
        OUTPUT_DIR / "first_layer_kernels.png",
        cmap="viridis",
        title="First Layer Kernels",
    )
    save_kernel_grid(
        kernels_2d,
        OUTPUT_DIR / "first_layer_kernels_gray.png",
        cmap="gray",
        title="First Layer Kernels Gray",
    )
    save_kernel_summary(kernels_2d, OUTPUT_DIR / "kernel_values_summary.csv")
    write_analysis(model_path, conv_name, tuple(weights.shape), max_show, kernels_2d)

    print(f"彩色卷积核图片保存位置: {OUTPUT_DIR / 'first_layer_kernels.png'}")
    print(f"灰度卷积核图片保存位置: {OUTPUT_DIR / 'first_layer_kernels_gray.png'}")
    print(f"卷积核统计 CSV 保存位置: {OUTPUT_DIR / 'kernel_values_summary.csv'}")
    print(f"卷积核分析文件保存位置: {OUTPUT_DIR / 'kernel_analysis.txt'}")


if __name__ == "__main__":
    main()
