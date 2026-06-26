# Task133 Skeleton Transformer 训练说明

## 模型结构
输入 X 为 [B,T,132]，先经过 Linear Embedding 将 132 维骨架特征映射到 d_model=128，再加入可学习位置编码，经过 2 层 Transformer Encoder，最后 Mean Pooling 并输入 MLP 分类器输出 6 类 logits。

## Transformer Encoder 作用
Transformer Encoder 通过自注意力建模不同时间帧之间的关系，适合从击球动作的骨架序列中学习时序模式。

## Mean Pooling 的作用
Mean Pooling 将 T 个时间步的特征平均为一个视频级表示，用于最终分类。

## 损失函数和优化器
损失函数为 CrossEntropyLoss，优化器为 Adam，学习率为 0.001。

## 最佳测试准确率
best_epoch=1, best_test_acc=0.3333
