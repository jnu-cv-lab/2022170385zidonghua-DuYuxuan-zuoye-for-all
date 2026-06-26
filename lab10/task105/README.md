# 任务五：Feature map 可视化

## 实验目的

本任务加载已经训练好的 `SimpleCNN` 模型，选择一张 MNIST 测试图片，提取并显示第一层卷积输出的 feature maps。通过观察不同 feature maps 的响应区域，理解不同卷积核会提取不同的图像特征。

## 使用的数据集和模型

本实验使用 MNIST 手写数字测试集和前面任务中的 `SimpleCNN` 模型。

`SimpleCNN` 模型结构保持不变：

- 第 1 个卷积层：输入通道 1，输出通道 16，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 1 个最大池化层：2x2
- 第 2 个卷积层：输入通道 16，输出通道 32，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 2 个最大池化层：2x2
- 全连接层：`32 * 7 * 7 -> 128`
- ReLU 激活层
- 输出层：`128 -> 10`

## 使用的模型权重文件

脚本会按以下顺序自动选择可用权重：

1. `lab10/task102/outputs/model_sgd_momentum.pth`
2. `lab10/task103/outputs/model_adam_lr_0_001.pth`
3. `lab10/task101/outputs/model_task101.pth`

本任务优先使用任务二中表现最好的 `model_sgd_momentum.pth`。

## Feature map 提取方法

脚本会自动找到模型中的第一层 `Conv2d`，并使用 forward hook 捕获该层的输出。对于 MNIST 单通道输入，第一层卷积输出通常包含 16 张 feature maps。脚本会显示前 16 张 feature maps，并对每张图进行归一化处理，便于观察响应强弱。

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `selected_test_image.png`：被选择的 MNIST 测试图片，标题包含真实类别和预测类别
- `feature_maps_first_layer.png`：彩色 colormap 形式的第一层 feature maps
- `feature_maps_first_layer_gray.png`：灰度形式的第一层 feature maps
- `feature_maps_summary.csv`：每张 feature map 的最小值、最大值、均值和标准差
- `feature_map_analysis.txt`：中文 feature map 可视化分析

## 运行方法

```bash
cd /home/david/cv-course/lab10/task105
python visualize_feature_maps_task105.py
```

## 简要实验结论

本次运行加载的权重为 `lab10/task102/outputs/model_sgd_momentum.pth`，选择的是 MNIST 测试集第 0 张图片，真实类别为 7，模型预测类别为 7。第一层 feature maps 的形状为 `(1, 16, 28, 28)`，实际显示 16 张 feature maps。

第一层 feature maps 展示了不同卷积核对输入图像不同区域的响应。响应较强的位置通常对应数字笔画、边缘、转折处或局部亮度变化明显的区域。不同卷积核参数不同，因此会提取不同方向、纹理或局部形状特征。
