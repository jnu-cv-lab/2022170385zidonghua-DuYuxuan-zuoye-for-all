# 任务三：学习率对比

## 实验目的

本任务在固定 CNN 模型结构和 Adam 优化器的前提下，对比不同 learning rate 对 MNIST 图像分类训练过程和最终测试准确率的影响。实验重点是观察 loss 曲线、accuracy 曲线以及最终 test accuracy 的差异。

## 使用的数据集

本任务使用 MNIST 手写数字数据集。该数据集包含 0 到 9 共 10 个类别，图像为 28x28 的单通道灰度图。

脚本会优先复用已有数据目录：

- `lab10/task101/data`
- `lab10/task102/data`
- `lab09/task95_to_97/data`

如果以上目录中没有 MNIST 数据，脚本会下载到 `lab10/task103/data`。

## 复用的 CNN 模型结构说明

本任务复用 `lab09`、`lab10/task101` 和 `lab10/task102` 中的 `SimpleCNN`，模型结构保持不变：

- 第 1 个卷积层：输入通道 1，输出通道 16，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 1 个最大池化层：2x2
- 第 2 个卷积层：输入通道 16，输出通道 32，卷积核 3x3，padding 为 1
- ReLU 激活层
- 第 2 个最大池化层：2x2
- 全连接层：`32 * 7 * 7 -> 128`
- ReLU 激活层
- 输出层：`128 -> 10`

## Adam 优化器和三个学习率设置说明

本任务固定优化器为 Adam，只改变 learning rate：

- learning rate = 0.1
- learning rate = 0.01
- learning rate = 0.001

三组实验使用相同的数据集划分、batch size、epoch 数、损失函数、模型结构和初始权重。

训练设置：

- batch size：64
- epoch：5
- loss function：CrossEntropyLoss
- 训练集：50000 张
- 验证集：10000 张
- 测试集：MNIST 原始测试集

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `lr_history.csv`：每个学习率每个 epoch 的训练记录
- `lr_summary.csv`：三个学习率的最终结果汇总
- `loss_lr_comparison.png`：三个学习率下的 loss 曲线对比图
- `accuracy_lr_comparison.png`：三个学习率下的 accuracy 曲线对比图
- `test_accuracy_lr_comparison.png`：三个学习率最终 test accuracy 柱状图
- `model_adam_lr_0_1.pth`：Adam lr=0.1 训练后的模型权重
- `model_adam_lr_0_01.pth`：Adam lr=0.01 训练后的模型权重
- `model_adam_lr_0_001.pth`：Adam lr=0.001 训练后的模型权重
- `test_result.txt`：三个学习率最终测试准确率和最佳学习率说明

## 运行方法

```bash
cd /home/david/cv-course/lab10/task103
python train_task103.py
```

## 实验结果简要分析

本次运行结果如下：

- learning rate = 0.1：10.28%
- learning rate = 0.01：98.26%
- learning rate = 0.001：98.81%

本次实验中，`learning rate = 0.001` 的测试准确率最高，为 98.81%。`0.1` 的学习率过大，训练 loss 基本维持在 2.31 左右，准确率接近随机猜测，说明训练过程不稳定且没有有效收敛。`0.01` 能够较快收敛，并达到较高准确率。`0.001` 的 loss 下降更稳定，最终 test accuracy 最高。

完整结果请查看：

```text
outputs/test_result.txt
```
