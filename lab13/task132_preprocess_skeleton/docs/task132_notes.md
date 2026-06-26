# Task132 MediaPipe 骨架预处理说明

## MediaPipe Pose 提取方法
程序使用 OpenCV 逐帧读取视频，将 BGR 图像转为 RGB 后输入 MediaPipe Pose，提取每帧人体 33 个关键点。

## 33×4=132 维含义
每个关键点包含 x、y、z、visibility 四个数值，因此单帧骨架特征为 33×4=132 维。

## [30,132] 的含义
每个视频被重采样为 30 帧，每帧 132 维，最终单个视频样本形状为 [30,132]。

## 归一化方法
程序以左右髋部中心作为坐标原点，以左右肩部距离作为尺度，对 x、y、z 坐标做平移和缩放；visibility 保持原值。肩宽接近 0 时使用 1.0 避免除零。

## 训练集/测试集数量
- 训练集: 24
- 测试集: 6

## 输出文件清单
- `/home/david/cv-course/lab13/data/processed/X_train.npy`
- `/home/david/cv-course/lab13/data/processed/y_train.npy`
- `/home/david/cv-course/lab13/data/processed/X_test.npy`
- `/home/david/cv-course/lab13/data/processed/y_test.npy`
- `/home/david/cv-course/lab13/data/processed/label_map.json`
- `/home/david/cv-course/lab13/data/processed/preprocess_log.csv`
- `/home/david/cv-course/lab13/data/processed/preprocess_summary.txt`
