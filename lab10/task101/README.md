# 任务一：复用上次 CNN 模型

## 实验目的

本任务复用第 9 次实验中的 CNN 图像分类模型，对同一数据集重新训练，并完整记录训练过程。实验重点不是修改模型结构，而是整理训练代码、保存训练记录、绘制曲线图、保存模型权重和记录最终测试结果。

## 使用的数据集

本任务使用 MNIST 手写数字数据集。该数据集包含 0 到 9 共 10 个数字类别，图像为 28x28 的单通道灰度图。

脚本会优先读取 `lab09/task95_to_97/data` 中已经下载好的 MNIST 数据。如果没有找到该数据，会下载到 `lab10/task101/data`。

## 复用的模型结构说明

本任务复用 `lab09` 中的 `SimpleCNN` 模型结构，保持网络层不变：

- 第 1 个卷积层：输入通道 1，输出通道 16，卷积核 3x3，padding 为 1
- 第 1 个 ReLU 激活层
- 第 1 个最大池化层：2x2
- 第 2 个卷积层：输入通道 16，输出通道 32，卷积核 3x3，padding 为 1
- 第 2 个 ReLU 激活层
- 第 2 个最大池化层：2x2
- 全连接层：`32 * 7 * 7 -> 128`
- 第 3 个 ReLU 激活层
- 输出层：`128 -> 10`

## 训练过程记录说明

训练轮数设置为 5 个 epoch。每个 epoch 记录以下指标：

- training loss
- validation loss
- training accuracy
- validation accuracy
- test accuracy

训练集和验证集划分方式为：从 MNIST 原始训练集 60000 张图像中划分 50000 张作为训练集，10000 张作为验证集。测试集使用 MNIST 原始测试集。

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `training_history.csv`：每个 epoch 的 loss 和 accuracy 记录
- `loss_curve.png`：training loss 和 validation loss 曲线图
- `accuracy_curve.png`：training accuracy、validation accuracy 和 test accuracy 曲线图
- `model_task101.pth`：最终训练好的模型权重
- `test_result.txt`：最终测试集 loss 和 accuracy

## 运行方法

在 `lab10/task101` 目录下运行：

```bash
python train_task101.py
```

也可以在项目根目录运行：

```bash
python lab10/task101/train_task101.py
```

## 实验结果填写位置

最终测试准确率请查看：

```text
outputs/test_result.txt
```

填写示例：

```text
Final Test Accuracy: 读取 outputs/test_result.txt 中的结果
```
