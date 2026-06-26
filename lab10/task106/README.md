# 任务六：错误分类样本分析

## 实验目的

本任务使用已经训练好的 `SimpleCNN` 模型对 MNIST 测试集进行预测，找出预测错误的样本并进行可视化和统计分析。实验目标是观察模型容易混淆哪些数字类别，并分析错误分类的可能原因和改进方向。

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

## 错误分类样本查找方法

脚本会加载 MNIST 测试集，逐张图片输入模型进行预测。若预测类别与真实类别不同，则记录该样本的测试集索引、真实类别、预测类别和预测置信度。随后统计每个真实类别的错误数量，以及真实类别到预测类别的混淆组合。

## 输出文件说明

运行脚本后，结果会保存到 `outputs` 文件夹：

- `misclassified_samples.png`：彩色 colormap 形式的错误分类样本图
- `misclassified_samples_gray.png`：灰度形式的错误分类样本图
- `misclassified_details.csv`：所有错误分类样本的索引、真实类别、预测类别和置信度
- `class_error_summary.csv`：每个真实类别被错误分类的数量
- `confused_pairs_summary.csv`：真实类别和预测类别的混淆组合统计
- `error_analysis.txt`：中文错误分类分析

## 运行方法

```bash
cd /home/david/cv-course/lab10/task106
python analyze_misclassified_task106.py
```

## 简要实验结论

本次运行加载的权重为 `lab10/task102/outputs/model_sgd_momentum.pth`。MNIST 测试集共有 10000 张图片，错误分类 116 张，测试准确率为 98.84%。本次显示前 16 张错误分类图片。

错误数量最多的真实类别是 9，共 25 张被错误分类。最常见的混淆组合是 `5 -> 3`，共 6 次；另外 `4 -> 9`、`9 -> 3` 也各出现 6 次。错误分类样本通常出现在手写数字形状相似、笔画模糊或断裂、数字倾斜、局部结构接近、有效笔画区域较小等情况下。通过分析混淆类别组合，可以更清楚地了解模型薄弱点。提高准确率可以从数据增强、改进模型结构和调整训练方法等方向入手。
