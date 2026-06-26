# 任务七：混淆矩阵

## 实验目的

本任务使用训练好的 `SimpleCNN` 模型对 MNIST 测试集进行预测，绘制 0 到 9 十个类别的混淆矩阵。通过观察对角线和非对角线元素，分析模型在哪些类别上分类正确，在哪些类别之间容易发生混淆。

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

## 混淆矩阵绘制方法

脚本会遍历 MNIST 测试集，收集所有样本的真实标签和预测标签，然后计算 10 x 10 混淆矩阵。矩阵的行表示真实类别，列表示预测类别。对角线表示分类正确的样本数量，非对角线表示分类错误或类别混淆的样本数量。

脚本同时保存原始数量矩阵和按真实类别归一化后的矩阵，并使用 matplotlib 绘制带数字标注的图片。

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `confusion_matrix.png`：原始数量混淆矩阵图片
- `confusion_matrix_normalized.png`：按真实类别归一化后的混淆矩阵图片
- `confusion_matrix.csv`：原始数量混淆矩阵
- `confusion_matrix_normalized.csv`：归一化混淆矩阵
- `confusion_pairs_summary.csv`：非对角线混淆组合统计
- `confusion_matrix_analysis.txt`：中文混淆矩阵分析

## 运行方法

```bash
cd /home/david/cv-course/lab10/task107
python confusion_matrix_task107.py
```

## 简要实验结论

本次运行加载的权重为 `lab10/task102/outputs/model_sgd_momentum.pth`。MNIST 测试集共有 10000 张图片，测试准确率为 98.84%，混淆矩阵大小为 10 x 10。

混淆矩阵可以直观看出模型整体分类效果。对角线数值越大，说明对应类别分类越准确；非对角线数值越大，说明真实类别和预测类别之间越容易混淆。本次最严重混淆组合有并列三组：`4 -> 9`、`5 -> 3`、`9 -> 3`，均为 6 次。这些错误通常与手写数字形状相似、笔画模糊、书写倾斜和局部结构接近有关。
