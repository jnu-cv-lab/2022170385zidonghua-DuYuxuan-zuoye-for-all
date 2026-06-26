# 任务三：实现高维 RoPE

## 任务名称

实现高维 RoPE（Rotary Position Embedding）。

## RoPE 的基本思想

RoPE 不是把位置编码直接加到输入向量上，而是根据位置对输入向量进行旋转。不同位置对应不同的旋转角度，因此向量在经过 RoPE 后会携带位置信息。

## 高维 RoPE 的实现方法

高维 RoPE 将高维向量拆成多个二维子空间，每两个维度一组。例如第 0、1 维为一组，第 2、3 维为一组，第 4、5 维为一组。每一组二维向量根据当前位置 `pos` 和对应频率 `theta_i` 进行二维旋转。

频率设置为：

```text
theta_i = 1 / 10000^(2i / d_model)
```

## 二维分组旋转公式

对于第 `i` 个二维分组 `[x_2i, x_2i+1]`，旋转公式为：

```text
x'_2i   = x_2i * cos(pos * theta_i) - x_2i+1 * sin(pos * theta_i)
x'_2i+1 = x_2i * sin(pos * theta_i) + x_2i+1 * cos(pos * theta_i)
```

## 实验参数

- `seq_len = 8`
- `d_model = 8`
- `random seed = 42`

## 输出文件说明

- `output/input_matrix.csv`：保存原始输入矩阵。
- `output/rope_output_matrix.csv`：保存经过 RoPE 旋转后的矩阵。
- `output/rope_norm_check.csv`：保存每个位置旋转前后的向量范数对比。
- `output/rope_input_heatmap.png`：展示 RoPE 前输入矩阵的热力图。
- `output/rope_output_heatmap.png`：展示 RoPE 后输出矩阵的热力图。
- `output/rope_norm_comparison.png`：展示 RoPE 前后向量范数的对比曲线。

## 简单说明

RoPE 旋转会改变向量方向，但基本保持向量长度不变。
