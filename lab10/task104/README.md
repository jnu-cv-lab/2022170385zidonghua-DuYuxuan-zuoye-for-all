# 任务四：卷积核可视化

## 实验目的

本任务加载前面任务中已经训练好的 CNN 模型，提取第一层卷积层的卷积核权重，并将其可视化。通过观察卷积核图像，理解 CNN 第一层如何学习局部边缘、方向和纹理等基础特征。

## 使用的数据集和模型

本实验延续前面任务使用的 MNIST 手写数字数据集和 `SimpleCNN` 模型。

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

## 第一层卷积核可视化方法

脚本会自动找到模型中的第一层 `Conv2d`，提取其 `weight`。第一层卷积核形状通常为：

```text
out_channels x in_channels x kernel_height x kernel_width
```

本实验中 MNIST 是单通道灰度图，因此每个第一层卷积核可以转换为二维图像显示。脚本会显示前 16 个卷积核，并对每个卷积核做归一化处理，使图像更容易观察。

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `first_layer_kernels.png`：彩色 colormap 形式的第一层卷积核可视化图
- `first_layer_kernels_gray.png`：灰度形式的第一层卷积核可视化图
- `kernel_values_summary.csv`：每个卷积核的最小值、最大值、均值、标准差
- `kernel_analysis.txt`：中文卷积核可视化分析

## 运行方法

```bash
cd /home/david/cv-course/lab10/task104
python visualize_kernels_task104.py
```

## 简要实验结论

训练后的第一层卷积核通常会呈现不同的局部响应模式。有些卷积核会对水平、垂直或斜向边缘更敏感，有些卷积核会关注局部灰度变化或纹理。不同卷积核之间的差异来自训练过程中的反向传播和梯度下降：模型为了降低数字分类损失，会自动调整卷积核权重，使其学习到有助于分类的局部特征。
