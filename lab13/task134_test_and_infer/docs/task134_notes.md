# Task134 测试评估与单视频推理说明

## 当前状态

由于尚未完成 task132 预处理和 task133 训练，当前还没有可加载的 `best_model.pth`，因此评估与推理结果文件尚未生成。

## 评估输出

`evaluate_model.py` 会加载测试集和 `best_model.pth`，输出 accuracy、confusion matrix、classification report，并保存：

- `task134_test_and_infer/output/confusion_matrix.png`
- `task134_test_and_infer/output/classification_report.txt`
- `task134_test_and_infer/output/evaluation_summary.txt`
- `task134_test_and_infer/output/test_predictions.csv`

## 单视频推理输出

`infer_single_video.py` 会使用与预处理一致的 MediaPipe Pose 提取、重采样和归一化逻辑，输出预测类别、置信度和六类 softmax 概率，并保存：

- `task134_test_and_infer/output/inference_result.txt`
- `task134_test_and_infer/output/inference_probabilities.csv`
