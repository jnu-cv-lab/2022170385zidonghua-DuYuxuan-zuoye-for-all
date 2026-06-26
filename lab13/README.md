# Lab13 Skeleton Transformer for Badminton Action Recognition

本实验使用 Kaggle `badminton_storke_video` 视频数据集完成羽毛球击球动作识别。流程不是直接输入原始视频像素，而是先用 OpenCV 读取视频，再用 MediaPipe Pose 提取每帧 33 个关键点，每个关键点包含 `x, y, z, visibility`，形成 `33×4=132` 维骨架特征。每个视频统一重采样为 `[30,132]`，最后使用轻量 Transformer Encoder 分类 6 类击球动作。

## 目录结构

- `data/raw`: 原始 Kaggle 视频数据。本机当前为指向 Windows 桌面 archive 的软链接。
- `data/processed`: 预处理后的 `X_train.npy`、`y_train.npy`、`X_test.npy`、`y_test.npy`、`label_map.json` 等。
- `task131_dataset_check`: 数据集扫描和类别统计。
- `task132_preprocess_skeleton`: MediaPipe Pose 骨架提取和训练/测试集划分。
- `task133_train_transformer`: Skeleton Transformer 模型与训练脚本。
- `task134_test_and_infer`: 测试集评估与单视频推理。
- `task135_report_and_visualization`: 骨架可视化。
- `docs`: 总实验报告和总结。

## 安装依赖

```bash
pip install -r lab13/requirements_lab13.txt
```

如果当前 Python 环境缺少 `mediapipe`，预处理、推理和可视化会停止并提示安装命令。本实验使用 `mediapipe==0.10.21`，因为当前 Python 3.12 环境中较新的 `mediapipe 0.10.35` 不再提供本实验需要的 `mediapipe.solutions.pose` 接口。

## 标签顺序

```text
0: forehand drive
1: forehand lift
2: forehand net shot
3: forehand clear
4: backhand drive
5: backhand net shot
```

脚本兼容类别文件夹中的空格和下划线，例如 `forehand drive` 与 `forehand_drive`。

## 运行顺序

```bash
python lab13/task131_dataset_check/check_dataset.py
python lab13/task132_preprocess_skeleton/preprocess_skeleton.py --demo_fast
python lab13/task132_preprocess_skeleton/preprocess_skeleton.py
python lab13/task133_train_transformer/train_transformer.py --epochs 20 --batch_size 16
python lab13/task134_test_and_infer/evaluate_model.py
python lab13/task134_test_and_infer/infer_single_video.py --video_path <某个真实视频路径>
python lab13/task135_report_and_visualization/visualize_skeleton_sequence.py --video_path <某个真实视频路径>
```

如果 CPU 训练较慢，可以先运行：

```bash
python lab13/task133_train_transformer/train_transformer.py --epochs 5 --batch_size 8
```

## 主要输出

- `task131_dataset_check/output/dataset_overview.txt`
- `task131_dataset_check/output/dataset_overview.csv`
- `data/processed/X_train.npy`
- `data/processed/y_train.npy`
- `data/processed/X_test.npy`
- `data/processed/y_test.npy`
- `data/processed/label_map.json`
- `task133_train_transformer/output/best_model.pth`
- `task133_train_transformer/output/training_curve.png`
- `task134_test_and_infer/output/confusion_matrix.png`
- `task134_test_and_infer/output/classification_report.txt`
- `task134_test_and_infer/output/inference_result.txt`
- `task135_report_and_visualization/output/skeleton_sample.png`
- `docs/lab13_experiment_report.md`

## 本次 demo_fast 运行结果

- 数据集检查：836 个视频全部映射到六类动作。
- 快速预处理：每类 5 个视频，共 30 个样本。
- `X_train.npy` 形状：`[24, 30, 132]`
- `X_test.npy` 形状：`[6, 30, 132]`
- 训练设置：CPU，5 epochs，batch size 8，Adam，学习率 0.001。
- 最佳测试准确率：0.3333。
- 单视频推理样例：`forehand_drive/001.mp4`，预测为 `backhand drive`，置信度 0.201772。
