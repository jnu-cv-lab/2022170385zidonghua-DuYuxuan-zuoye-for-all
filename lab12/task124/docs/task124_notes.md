# 任务四：对比 E+pos 和 RoPE 的输入方式

## 任务名称

对比 E+pos 和 RoPE 的输入方式。

## E+pos 简要说明

E+pos 是将 token embedding 与位置编码直接相加。这种方式会把内容信息和位置信息混合在同一个向量中。

## RoPE 简要说明

RoPE 不是加法，而是根据位置对向量进行旋转。RoPE 通过改变向量方向注入位置信息。

## 实验参数

- `seq_len = 8`
- `d_model = 8`
- `random seed = 42`

## 输出文件说明

- `output/embedding_matrix.csv`：原始 embedding 矩阵。
- `output/position_encoding_matrix.csv`：sinusoidal position encoding 矩阵。
- `output/e_plus_pos_matrix.csv`：E+pos 方式得到的矩阵。
- `output/rope_matrix.csv`：RoPE 方式得到的矩阵。
- `output/norm_comparison.csv`：三种方式的向量范数对比结果。
- `output/embedding_heatmap.png`：原始 embedding 热力图。
- `output/e_plus_pos_heatmap.png`：E+pos 后矩阵热力图。
- `output/rope_heatmap.png`：RoPE 后矩阵热力图。
- `output/norm_comparison.png`：E、E+pos、RoPE 三种方式的范数对比图。
- `output/position3_component_comparison.png`：第 3 个位置的向量分量对比柱状图。

## 简单结论

E+pos 通常会改变向量长度。RoPE 基本保持向量长度不变。两者都能注入位置信息，但 RoPE 的方式更接近一种结构化的旋转变换。
