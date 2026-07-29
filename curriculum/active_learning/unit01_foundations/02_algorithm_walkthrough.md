# Unit 01 算法走读：一次无泄漏 query

## 手算

假设三个候选的预测如下，任务是最大化：

| 候选 | 预测均值 \(\mu\) | 标准差 \(\sigma\) |
|---|---:|---:|
| A | 7.0 | 0.2 |
| B | 6.5 | 1.0 |
| C | 6.8 | 0.4 |

当 \(\beta=1\)：

\[
\mathrm{UCB}=\mu+\beta\sigma
\]

- A：7.2
- B：7.5
- C：7.2

因此选择 B。B 的预测均值不是最高，但探索价值更大。

## 中文伪代码

```text
读取已标注数据
训练代理模型
只读取候选特征进行预测
计算每个候选的采集分数
固定 query_id
保存选择理由
越过标签揭示线
Oracle 返回 query_id 的真实标签
加入已标注集
```

## 变量形状

| 变量 | 典型形状 | 含义 |
|---|---|---|
| `X_labeled` | `(n_labeled, n_features)` | 已测样本特征 |
| `y_labeled` | `(n_labeled,)` | 已测目标 |
| `X_pool` | `(n_pool, n_features)` | 未测候选特征 |
| `prediction_mean` | `(n_pool,)` | 每个候选预测 |
| `prediction_std` | `(n_pool,)` | 每个候选不确定性 |
| `query_position` | 标量整数 | 候选数组中的位置 |

`(n_labeled,)` 右侧为空，是因为 `y` 是一维数组，不是缺少一个维度。

## Python 语法

```python
query_position = int(np.argmax(ucb_score))
query_id = pool_ids[query_position]
```

- `np.argmax(...)` 返回最大值所在位置；
- `int(...)` 转成普通 Python 整数；
- `pool_ids[query_position]` 用位置取得永久候选编号；
- 位置会随数组过滤改变，永久编号不能用位置代替。

## 标签揭示线

错误：

```python
query_position = np.argmax(y_pool)
```

正确：

```python
query_position = np.argmax(ucb_score)
query_id = pool_ids[query_position]

# 下面才允许按 query_position 取得 y_pool。
selected_y = y_pool[query_position]
```
