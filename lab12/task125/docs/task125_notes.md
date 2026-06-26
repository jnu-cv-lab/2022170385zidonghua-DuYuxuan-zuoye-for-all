# 任务五：用数值实验验证 RoPE 的相对位置性质

## 任务名称

用数值实验验证 RoPE 的相对位置性质。

## RoPE 相对位置性质

RoPE 作用在 Q 和 K 上。Q 和 K 经过不同位置的旋转后，它们的点积可以转化为与相对位置 `n-m` 有关的形式。因此，RoPE 可以让 attention score 自然包含相对位置信息。

验证公式为：

```text
dot(RoPE(q, m), RoPE(k, n)) = dot(q, RoPE(k, n - m))
```

## 实验参数

- `seq_len = 8`
- `d_model = 8`
- `random seed = 42`

## 使用固定 q 和固定 k 的原因

为了排除内容向量变化的影响，本实验中所有位置使用同一个 `q` 和同一个 `k`。这样可以更清楚地观察相对位置 `n-m` 对点积的影响。

## 输出文件说明

- `output/rope_relative_position_results.csv`：保存所有位置对 `(m, n)` 的 direct score、relative score 和误差。
- `output/rope_attention_score_matrix.csv`：保存 attention score 矩阵。
- `output/rope_relative_error_matrix.csv`：保存 direct score 与 relative score 的误差矩阵。
- `output/score_by_relative_position.csv`：保存按相对位置分组后的平均 score 和样本数量。
- `output/rope_attention_score_heatmap.png`：attention score 热力图。
- `output/rope_relative_error_heatmap.png`：误差热力图。
- `output/score_by_relative_position.png`：相对位置与平均 score 的关系图。
- `output/direct_vs_relative_score.png`：direct score 与 relative score 的散点对比图。

## 简单结论

如果 direct score 和 relative score 的误差接近 0，则说明 RoPE 的点积天然包含相对位置信息。
