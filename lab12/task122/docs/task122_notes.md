# 任务二：实现二维向量旋转

## 任务名称

实现二维向量旋转。

## 二维旋转矩阵公式

输入弧度制角度 `theta`，二维旋转矩阵为：

```text
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

旋转后的向量通过矩阵乘法得到：

```text
v_rotated = R(theta) @ v
```

## 测试设置

- 测试向量：`[1, 0]`
- 测试角度：`0°`、`45°`、`90°`、`180°`、`-90°`

## 输出文件说明

- `output/vector_rotation_results.csv`：保存不同角度下的旋转结果。
- `output/vector_rotation_multiple_angles.png`：展示原始向量和多个角度旋转后的向量。
- `output/vector_rotation_90deg.png`：单独展示 `[1, 0]` 旋转 90° 后的效果。

## 简单说明

二维旋转不会改变向量长度，只会改变向量方向。
