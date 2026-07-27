# Day 17 算法推演：手写一元线性梯度下降

## 1. 数据与目标关系

```python
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 5.0, 7.0])
```

理想关系是：

```text
y = 2x + 1
```

从 `w=0, b=0` 开始。

## 2. 第 0 步前向

预测：

```text
[0, 0, 0, 0]
```

误差：

```text
[-1, -3, -5, -7]
```

MSE：

```text
(1 + 9 + 25 + 49) / 4 = 21
```

## 3. 第 0 步梯度

```text
grad_w = 2 mean(error × x)
       = 2 × mean([0, -3, -10, -21])
       = -17

grad_b = 2 mean(error)
       = 2 × (-4)
       = -8
```

若学习率 `0.05`：

```text
w_new = 0 - 0.05 × (-17) = 0.85
b_new = 0 - 0.05 × (-8)  = 0.40
```

负梯度加上更新式中的减号，使参数向正方向移动。

## 4. 第 1 步

新预测：

```text
0.85x + 0.40 = [0.40, 1.25, 2.10, 2.95]
```

重新计算误差、损失与梯度。不要沿用第 0 步误差。

## 5. 封装运行函数

```python
def run_gradient_descent(learning_rate, n_steps=80):
    w = 0.0
    b = 0.0
    records = []

    for step in range(n_steps + 1):
        prediction = x * w + b
        error = prediction - y
        loss = np.mean(error ** 2)
        records.append({"step": step, "loss": loss, "w": w, "b": b})

        if step == n_steps:
            break

        grad_w = 2.0 * np.mean(error * x)
        grad_b = 2.0 * np.mean(error)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return pd.DataFrame(records)
```

先记录再更新，因此每行参数与该行 loss 一一对应。

## 6. 比较学习率

用相同初始化与步数：

```python
small = run_gradient_descent(0.005)
medium = run_gradient_descent(0.05)
large = run_gradient_descent(0.5)
```

比较时不能只看最后一行；应查看完整曲线、有限值以及是否震荡。大步长可能发散，具体阈值由当前数据尺度决定。

## 7. 合理性检查

```python
assert history.iloc[0]["loss"] == 21.0
assert np.isfinite(history[["loss", "w", "b"]]).all().all()
assert history.iloc[-1]["loss"] < history.iloc[0]["loss"]
```

最后一条只适用于预先选作“稳定示例”的学习率，不应强加给故意发散的对照。

## 8. 有限差分检查梯度（扩展）

不用微积分也可以近似检查：

```text
dL/dw ≈ [L(w+ε,b) - L(w-ε,b)] / (2ε)
```

若解析梯度与有限差分相差很大，优先检查公式与数组对齐。有限差分只是数值检查，不替代理解。

## 9. 结论边界

这段代码能说明梯度下降机制和学习率影响；它不能说明 MLP 已训练、不能说明验证泛化，也没有任何 ESOL 或粘合剂性能含义。
