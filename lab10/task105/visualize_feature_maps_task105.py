import csv
import os
from pathlib import Path

# 避免部分 WSL 环境中 Matplotlib 尝试写入 ~/.config 时出现权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-task105")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


# 与 lab09、task101、task102、task103、task104 中的 SimpleCNN 完全一致。
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


def find_first_conv_layer(model):
    """自动找到模型中的第一层卷积层。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise RuntimeError("模型中没有找到 Conv2d 卷积层。")


def normalize_map(feature_map):
    """将 feature map 归一化到 0 到 1，便于显示响应强弱。"""
    min_value = feature_map.min()
    max_value = feature_map.max()
    if max_value - min_value < 1e-12:
        return torch.zeros_like(feature_map)
    return (feature_map - min_value) / (max_value - min_value)


def choose_test_sample(model, test_dataset):
    """优先选择一张模型能够正确分类的测试图像。"""
    model.eval()
    fallback = None

    with torch.no_grad():
        for index in range(min(200, len(test_dataset))):
            image, label = test_dataset[index]
            output = model(image.unsqueeze(0))
            predicted = int(output.argmax(dim=1).item())

            if fallback is None:
                fallback = (index, image, int(label), predicted)

            if predicted == int(label):
                return index, image, int(label), predicted

    return fallback


def save_selected_image(image, true_label, predicted_label, output_path):
    """保存被选择的 MNIST 测试图片。"""
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"True: {true_label} | Pred: {predicted_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def extract_first_layer_feature_maps(model, conv_layer, image):
    """使用 forward hook 提取第一层卷积层输出的 feature maps。"""
    captured = {}

    def hook_fn(module, input_tensor, output_tensor):
        captured["feature_maps"] = output_tensor.detach().cpu()

    hook = conv_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    hook.remove()

    if "feature_maps" not in captured:
        raise RuntimeError("没有成功提取第一层 feature maps。")

    return captured["feature_maps"]


def save_feature_maps_grid(feature_maps, output_path, cmap, title):
    """保存 feature maps 网格图。"""
    num_maps = feature_maps.shape[0]
    cols = 4
    rows = (num_maps + cols - 1) // cols

    plt.figure(figsize=(cols * 2.4, rows * 2.4))
    for index in range(num_maps):
        feature_map = feature_maps[index]
        plt.subplot(rows, cols, index + 1)
        plt.imshow(normalize_map(feature_map), cmap=cmap, interpolation="nearest")
        plt.title(f"Feature Map {index + 1}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_map_summary(feature_maps, csv_path):
    """保存每张 feature map 的统计信息。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["feature_map_index", "min_value", "max_value", "mean_value", "std_value"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, feature_map in enumerate(feature_maps, start=1):
            writer.writerow(
                {
                    "feature_map_index": index,
                    "min_value": float(feature_map.min().item()),
                    "max_value": float(feature_map.max().item()),
                    "mean_value": float(feature_map.mean().item()),
                    "std_value": float(feature_map.std(unbiased=False).item()),
                }
            )


def describe_strong_response_regions(feature_maps):
    """粗略统计高响应位置，生成中文观察描述。"""
    descriptions = []
    h, w = feature_maps.shape[1], feature_maps.shape[2]

    for index, feature_map in enumerate(feature_maps, start=1):
        max_position = torch.nonzero(feature_map == feature_map.max(), as_tuple=False)[0]
        row = int(max_position[0].item())
        col = int(max_position[1].item())

        if row < h / 3:
            vertical_region = "上部"
        elif row < h * 2 / 3:
            vertical_region = "中部"
        else:
            vertical_region = "下部"

        if col < w / 3:
            horizontal_region = "左侧"
        elif col < w * 2 / 3:
            horizontal_region = "中间"
        else:
            horizontal_region = "右侧"

        descriptions.append(f"Feature Map {index} 的最强响应位于图像{vertical_region}{horizontal_region}附近")

    return descriptions


def write_analysis(model_path, true_label, predicted_label, feature_shape, shown_maps, feature_maps):
    """生成中文 feature map 分析文件。"""
    response_descriptions = describe_strong_response_regions(feature_maps)
    high_std_count = sum(1 for fmap in feature_maps if float(fmap.std(unbiased=False).item()) > 0.15)

    lines = [
        "Lab 10 任务五：Feature map 可视化分析",
        "",
        f"本任务加载的模型权重文件：{model_path}",
        f"选择的测试图片真实类别：{true_label}",
        f"模型预测类别：{predicted_label}",
        f"第一层 feature maps 形状：{tuple(feature_shape)}",
        f"本次实际显示的 feature maps 数量：{shown_maps}",
        "",
        "不同 feature maps 的响应观察：",
        f"显示的 {shown_maps} 张 feature maps 中，有 {high_std_count} 张的响应标准差较大，说明它们对图像局部区域的响应差异更明显。",
    ]

    lines.extend(response_descriptions[:8])
    lines.extend(
        [
            "从可视化图中可以看到，不同 feature maps 会在数字笔画的不同位置出现较强响应，例如笔画边缘、转折处、局部亮度变化明显的位置。",
            "",
            "原理说明：",
            "不同卷积核会提取不同图像特征，例如边缘、笔画方向、局部纹理或局部亮度变化。",
            "feature map 是卷积核在输入图像上滑动卷积后得到的响应结果。",
            "响应较强的区域表示该卷积核在这些位置检测到了更符合自身权重模式的局部特征。",
        ]
    )

    with open(OUTPUT_DIR / "feature_map_analysis.txt", "w", encoding="utf-8") as analysis_file:
        analysis_file.write("\n".join(lines) + "\n")


def main():
    print("=" * 60)
    print("Lab 10 任务五：Feature map 可视化")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path()
    print(f"使用的模型权重文件: {model_path}")

    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    conv_name, conv_layer = find_first_conv_layer(model)
    test_dataset = load_test_dataset()
    sample_index, image, true_label, predicted_label = choose_test_sample(model, test_dataset)

    save_selected_image(
        image,
        true_label,
        predicted_label,
        OUTPUT_DIR / "selected_test_image.png",
    )

    feature_maps_batch = extract_first_layer_feature_maps(model, conv_layer, image)
    print(f"第一层卷积层名称: {conv_name}")
    print(f"选择的测试图片索引: {sample_index}")
    print(f"选择的测试图片真实类别: {true_label}")
    print(f"模型预测类别: {predicted_label}")
    print(f"第一层 feature maps 形状: {tuple(feature_maps_batch.shape)}")

    # 去掉 batch 维度，形状变为 channels x height x width。
    feature_maps = feature_maps_batch[0]
    shown_maps = min(16, feature_maps.shape[0])
    if shown_maps < 8:
        raise RuntimeError("第一层 feature maps 数量少于 8 张，不满足任务要求。")

    shown_feature_maps = feature_maps[:shown_maps]
    print(f"实际可视化 feature maps 数量: {shown_maps}")

    save_feature_maps_grid(
        shown_feature_maps,
        OUTPUT_DIR / "feature_maps_first_layer.png",
        cmap="viridis",
        title="First Layer Feature Maps",
    )
    save_feature_maps_grid(
        shown_feature_maps,
        OUTPUT_DIR / "feature_maps_first_layer_gray.png",
        cmap="gray",
        title="First Layer Feature Maps Gray",
    )
    save_feature_map_summary(shown_feature_maps, OUTPUT_DIR / "feature_maps_summary.csv")
    write_analysis(model_path, true_label, predicted_label, tuple(feature_maps_batch.shape), shown_maps, shown_feature_maps)

    print(f"原始测试图片保存位置: {OUTPUT_DIR / 'selected_test_image.png'}")
    print(f"彩色 feature maps 图片保存位置: {OUTPUT_DIR / 'feature_maps_first_layer.png'}")
    print(f"灰度 feature maps 图片保存位置: {OUTPUT_DIR / 'feature_maps_first_layer_gray.png'}")
    print(f"feature maps 统计 CSV 保存位置: {OUTPUT_DIR / 'feature_maps_summary.csv'}")
    print(f"feature map 分析文件保存位置: {OUTPUT_DIR / 'feature_map_analysis.txt'}")


if __name__ == "__main__":
    main()
