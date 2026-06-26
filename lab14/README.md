# Lab14：棋盘格相机标定

本文件夹是计算机视觉实验 lab14，主题为使用棋盘格进行相机标定 Camera Calibration Using a Checkerboard Pattern。

## 项目结构

```text
lab14/
├── data/
│   └── synthetic_raw/              # 24 张合成棋盘格原图
├── src/
│   ├── generate_synthetic_checkerboards.py
│   └── camera_calibration.py
├── output/
│   ├── corners/                    # 绘制角点检测结果图
│   ├── undistorted/                # 去畸变图
│   ├── compare/                    # 原图与去畸变对比图
│   ├── samples/                    # 报告可直接使用的样例图
│   └── results/                    # K、D、误差、CSV、JSON、TXT 结果
├── docs/
│   ├── experiment_notes.md
│   └── report_materials.md
└── README.md
```

## 运行方式

在 `lab14` 目录下运行：

```bash
python src/generate_synthetic_checkerboards.py
python src/camera_calibration.py
```

第一条命令生成 24 张严格数学透视投影的合成棋盘格图片，并保存真实相机参数。第二条命令读取图片，检测 9×6 内角点，完成亚像素优化、相机标定、重投影误差计算和去畸变处理。

## 输出结果说明

- 合成原图：`data/synthetic_raw/`
- 角点检测结果图：`output/corners/`
- 去畸变图：`output/undistorted/`
- 原图与去畸变对比图：`output/compare/undistort_compare.png`
- 报告样例图：`output/samples/`
- 标定结果文件：`output/results/`

主要结果文件包括：

- `output/results/true_camera_parameters.json`
- `output/results/calibration_results.txt`
- `output/results/calibration_results.json`
- `output/results/camera_matrix_K.csv`
- `output/results/distortion_coefficients_D.csv`
- `output/results/per_image_reprojection_errors.csv`

## 注意

本实验使用数学合成图片进行流程验证。正式真实相机标定时，应使用同一真实相机拍摄的棋盘格图片，且图片应覆盖不同位置、距离、倾斜角和旋转姿态。
