# Lab14 报告素材整理

## 1. 建议插入的图片

- 4 张不同姿态合成棋盘格原图拼图：`output/samples/sample_4_raw_images.png`
- 2 张角点检测结果拼图：`output/samples/sample_2_corner_images.png`
- 原图与去畸变图对比：`output/samples/sample_undistort_compare.png`
- 如需查看全部原图：`data/synthetic_raw/calibration_01.png` 到 `data/synthetic_raw/calibration_24.png`
- 如需查看全部角点检测图：`output/corners/`
- 去畸变结果图：`output/undistorted/`

## 2. 标定结果文件位置

- 真实合成相机参数：`output/results/true_camera_parameters.json`
- 完整标定结果文本：`output/results/calibration_results.txt`
- 完整标定结果 JSON：`output/results/calibration_results.json`
- 相机内参矩阵 K：`output/results/camera_matrix_K.csv`
- 畸变参数 D：`output/results/distortion_coefficients_D.csv`
- 每张图片重投影误差：`output/results/per_image_reprojection_errors.csv`

## 3. 简要分析模板

### fx 和 fy 是否接近

可以比较标定得到的 K 矩阵中 fx 与 fy。由于本实验设置的真实值为 fx = 900、fy = 920，两者应当比较接近，但不完全相同。若估计结果接近该设置，说明标定结果合理。

### cx 和 cy 是否接近图像中心

图像分辨率为 1280×960，因此图像中心约为 (640, 480)。可以检查 K 矩阵中的 cx 和 cy 是否接近 (640, 480)。本实验真实主点也设置在图像中心，因此估计值应接近该位置。

### 重投影误差是否合理

重投影误差越小，表示标定模型对检测角点的解释越好。由于本实验使用数学合成棋盘格并且图像清晰，重投影误差应保持在较低水平。报告中可以引用 `calibration_results.txt` 中的 RMS 重投影误差和手动计算平均重投影误差。

### 是否有角点检测失败

查看 `calibration_results.txt` 或 `calibration_results.json` 中的失败图片列表。如果失败数量为 0，说明 24 张合成图片均成功检测到棋盘格角点；如果存在失败图片，需要说明失败原因可能是姿态过大、棋盘格太靠近边缘或图像质量不足。

### 如果改进实验，可以如何改进

可以改进的方向包括：使用真实相机拍摄棋盘格图片进行真实标定；增加更多姿态和距离变化；加入更接近真实拍摄过程的光照变化、轻微模糊或传感器噪声；比较不同角点检测算法和不同畸变模型对结果的影响。
