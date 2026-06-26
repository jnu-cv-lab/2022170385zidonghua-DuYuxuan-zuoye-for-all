# 任务二：优化器对比

## 实验目的

本任务在不改变 CNN 模型结构的前提下，对比三种常见优化器在 MNIST 图像分类任务上的训练效果。实验通过记录训练集、验证集和测试集指标，观察不同优化器对模型收敛速度和最终准确率的影响。

## 使用的数据集

本任务使用 MNIST 手写数字数据集。该数据集包含 0 到 9 共 10 个类别，图像为 28x28 的单通道灰度图。

脚本会优先复用已有数据目录：

- `lab10/task101/data`
- `lab09/task95_to_97/data`

如果以上目录中没有 MNIST 数据，脚本会下载到 `lab10/task102/data`。

## 复用的 CNN 模型结构说明

本任务复用 `lab09` 和 `lab10/task101` 中的 `SimpleCNN`，模型结构保持不变：

- 第 1 个卷积层：输入通道 1，输出通道 16，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 1 个最大池化层：2x2
- 第 2 个卷积层：输入通道 16，输出通道 32，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 2 个最大池化层：2x2
- 全连接层：`32 * 7 * 7 -> 128`
- ReLU 激活层
- 输出层：`128 -> 10`

## 三种优化器设置说明

三种优化器使用相同的数据集划分、batch size、epoch 数、损失函数和模型初始权重。

- SGD：learning rate = 0.01
- SGD + Momentum：learning rate = 0.01，momentum = 0.9
- Adam：learning rate = 0.001

训练设置：

- batch size：64
- epoch：5
- loss function：CrossEntropyLoss
- 训练集：50000 张
- 验证集：10000 张
- 测试集：MNIST 原始测试集

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `optimizer_history.csv`：每种优化器每个 epoch 的训练记录
- `optimizer_summary.csv`：三种优化器的最终结果汇总
- `loss_comparison.png`：三种优化器的 loss 曲线对比图
- `accuracy_comparison.png`：三种优化器的 accuracy 曲线对比图
- `test_accuracy_comparison.png`：三种优化器最终 test accuracy 柱状图
- `model_sgd.pth`：SGD 训练后的模型权重
- `model_sgd_momentum.pth`：SGD + Momentum 训练后的模型权重
- `model_adam.pth`：Adam 训练后的模型权重
- `test_result.txt`：三种优化器最终测试准确率和最佳优化器说明

## 运行方法

```bash
cd /home/david/cv-course/lab10/task102
python train_task102.py
```

## 实验结果简要分析

本次运行结果如下：

- SGD：94.79%
- SGD + Momentum：98.84%
- Adam：98.81%

本次实验中，`SGD + Momentum` 的测试准确率最高，为 98.84%。从训练过程看，普通 SGD 收敛较慢，5 个 epoch 内准确率明显低于另外两种优化器；SGD + Momentum 通过动量项让参数更新方向更稳定，收敛速度明显提升；Adam 也表现很好，因为它能自适应调整参数学习率，前期收敛较快。本次结果中 SGD + Momentum 略高于 Adam，但二者差距很小。

完整结果请查看：

```text
outputs/test_result.txt
```
