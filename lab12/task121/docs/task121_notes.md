# 任务一：实现 Sinusoidal Position Encoding

## 使用公式

对于位置 `pos` 和模型维度 `d_model`，Sinusoidal Position Encoding 的计算公式为：

```text
PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

## 参数设置

- `seq_len = 32`
- `d_model = 16`
- 生成的位置编码矩阵形状为 `(32, 16)`

## 输出文件说明

- `output/position_encoding_matrix.csv`：完整的位置编码矩阵。
- `output/position_encoding_heatmap.png`：位置编码矩阵热力图。
- `output/position_encoding_curves.png`：前几个维度随位置变化的曲线图。

## 简单说明

偶数维使用 `sin` 函数进行编码，奇数维使用 `cos` 函数进行编码。
