# 计算机视觉实验第十三课实验报告

## 1. 实验名称

Skeleton Transformer for Badminton Action Recognition：基于 MediaPipe Pose 与骨架序列 Transformer 的羽毛球击球动作识别。

## 2. 实验背景

传统视频动作识别常直接把原始视频帧输入 3D CNN 或 Video Transformer，计算量较大。本实验改用人体骨架序列作为动作表示：先从视频中提取人体关键点，再用轻量 Transformer Encoder 建模时间序列，从而完成 6 类羽毛球击球动作分类。

## 3. 数据集说明

本实验使用 Kaggle `badminton_storke_video` 数据集。原数据集名称中的 `storke` 是页面拼写，本实验按 `stroke` 击球动作理解。

标签顺序如下：

- 0: forehand drive
- 1: forehand lift
- 2: forehand net shot
- 3: forehand clear
- 4: backhand drive
- 5: backhand net shot

数据集检查结果保存在 `lab13/task131_dataset_check/output/dataset_overview.txt` 和 `dataset_overview.csv`。

本次检查共找到并映射 836 个视频，各类别数量如下：

- forehand drive: 159
- forehand lift: 174
- forehand net shot: 105
- forehand clear: 119
- backhand drive: 111
- backhand net shot: 168

## 4. 方法流程

整体流程为：

```text
Kaggle 羽毛球视频
-> OpenCV 逐帧读取
-> MediaPipe Pose 提取人体 33 个关键点
-> 每帧得到 33×4=132 维骨架特征
-> 每个视频重采样为 [30,132]
-> Skeleton Transformer 分类 6 类动作
-> 测试评估、单视频推理、结果可视化
```

## 5. MediaPipe Pose 骨架特征提取

MediaPipe Pose 对每帧图像输出 33 个人体关键点。每个关键点包含：

- `x`: 归一化横坐标
- `y`: 归一化纵坐标
- `z`: 相对深度
- `visibility`: 可见性置信度

因此单帧特征维度为 `33×4=132`。

## 6. 骨架序列数据格式：[30,132]

本实验统一把每个视频重采样为 30 帧，因此每个样本形状为 `[30,132]`。如果任务书其他位置出现 `[60,132]`，本实验仍按前面多次要求和模型参数统一使用 `[30,132]`，保证预处理、训练、测试和推理一致。

预处理时还进行了简单骨架归一化：以左右髋部中心为原点，以左右肩部距离为尺度，对 `x,y,z` 坐标进行平移和缩放。若肩宽接近 0，则使用 1.0 避免除零。未检测到人体的帧使用全 0 向量。

## 7. Skeleton Transformer 模型结构

模型输入为 `X: [B,T,132]`，结构如下：

```text
[B,T,132]
-> Linear Embedding: 132 -> d_model
-> Learnable Position Embedding
-> Transformer Encoder × 2
-> Mean Pooling
-> MLP Classifier
-> 6 类 logits
```

推荐参数在代码中实现为：

- `input_dim=132`
- `target_frames=30`
- `d_model=128`
- `nhead=4`
- `num_layers=2`
- `dim_feedforward=256`
- `num_classes=6`
- `dropout=0.1`

## 8. 训练设置

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`
- 学习率：`1e-3`
- batch size：默认 `16`，本次快速实验使用 `8`
- epochs：默认 `20`，本次快速实验使用 `5`
- 设备：自动检测 CUDA，没有 GPU 时使用 CPU

训练日志保存在 `lab13/task133_train_transformer/output/training_log.csv`，训练曲线保存在 `training_curve.png`。

## 9. 实验结果

本次为了课堂快速跑通，先使用 `--demo_fast` 模式，每类最多处理 5 个视频，共 30 个样本。预处理结果如下：

- `X_train.npy`: `[24,30,132]`
- `y_train.npy`: `[24]`
- `X_test.npy`: `[6,30,132]`
- `y_test.npy`: `[6]`
- 低检测率视频数量：4

短训练设置为 CPU、5 epochs、batch size 8。训练结果如下：

- best_epoch: 1
- best_test_acc: 0.3333
- 测试样本数: 6
- 测试 accuracy: 0.3333

单视频推理样例使用 `forehand_drive/001.mp4`：

- predicted class: backhand drive
- confidence: 0.201772
- total_frames: 56
- detected_frames: 55
- detection_rate: 0.9821

结果文件：

- 训练曲线：`lab13/task133_train_transformer/output/training_curve.png`
- 测试 accuracy：`lab13/task134_test_and_infer/output/evaluation_summary.txt`
- confusion matrix：`lab13/task134_test_and_infer/output/confusion_matrix.png`
- classification report：`lab13/task134_test_and_infer/output/classification_report.txt`
- 单样本推理结果：`lab13/task134_test_and_infer/output/inference_result.txt`
- 骨架可视化：`lab13/task135_report_and_visualization/output/skeleton_sample.png`

## 10. 问题分析

MediaPipe 检测失败会让某些帧变成全 0 特征，降低骨架序列质量。如果整段视频检测率很低，分类模型很难学习有效动作模式。

数据量较小时，Transformer 仍可能出现过拟合，表现为训练准确率升高但测试准确率不稳定。本次 `demo_fast` 只有 30 个样本，测试集仅 6 个样本，因此 0.3333 的准确率只能说明流程已跑通，不能代表最终模型性能。可以通过处理完整数据、增加数据增强、减少模型规模或增加正则化缓解。

相比直接处理原始视频像素，骨架方法计算量更低，输入维度更小，也更容易在 CPU 上跑通课堂实验。

但只使用人体骨架会忽略球拍、羽毛球、场地位置等信息，这些信息对区分相近击球动作可能有帮助。

## 11. 总结

本实验完成了从原始羽毛球视频到骨架序列、再到 Skeleton Transformer 分类的完整流程。核心数据格式为 `[30,132]`，模型通过 Transformer Encoder 学习时间帧之间的动作关系，最终输出 6 类击球动作预测。
