# Lab13 实验总结

本实验目录 `lab13` 已按 task131 到 task135 划分：

- task131：扫描原始视频数据并统计六类动作数量。
- task132：使用 MediaPipe Pose 提取 `[30,132]` 骨架序列并划分训练集/测试集。
- task133：训练 Skeleton Transformer 分类模型。
- task134：评估测试集并支持单视频推理。
- task135：保存骨架关键点可视化图片。

数据标签顺序固定为：

```text
0 forehand drive
1 forehand lift
2 forehand net shot
3 forehand clear
4 backhand drive
5 backhand net shot
```

当前 `lab13/data/raw` 指向已找到的 Windows 桌面 archive 数据目录，避免重复复制原始视频。

## 本次运行结果

- 数据集检查：836 个视频全部映射到六类动作。
- 快速预处理：`--demo_fast` 每类 5 个视频，共 30 个样本。
- `X_train.npy`: `[24,30,132]`
- `X_test.npy`: `[6,30,132]`
- 训练：CPU，5 epochs，batch size 8。
- best_test_acc: 0.3333。
- 评估：生成 confusion matrix、classification report 和 test predictions。
- 单视频推理：`forehand_drive/001.mp4`，预测 `backhand drive`，置信度 0.201772。
- 骨架可视化：已生成 `skeleton_sample.png` 和 `skeleton_sequence_overview.png`。
