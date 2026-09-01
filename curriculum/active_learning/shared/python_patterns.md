# 主动学习代码里的 Python 语法速查

本页只解释 Unit 4–9 反复出现的代码结构。先知道“输入、动作、输出”，
再回到 Notebook 看算法；不要求一次背完。

## 1. 函数参数为什么比全局变量安全

```python
def choose_candidate(X_labeled, y_labeled, X_pool):
    ...
```

- `def` 定义函数；
- 括号内是调用者必须交给函数的输入；
- 函数没有 `y_pool` 参数，就不应该读取候选答案；
- 这比把所有数组放在全局作用域、只靠注释说“不要偷看”更容易审查。

## 2. 回调函数

```python
def run_strategy(..., oracle, evaluator):
    observed_y = oracle(query_id)
    test_rmse = evaluator(model)
```

这里的 `oracle` 和 `evaluator` 也是函数：

- policy 先固定 `query_id`；
- `oracle(query_id)` 只返回该候选标签；
- `evaluator(model)` 只计算事后指标；
- policy 不需要知道 Oracle 内部是查表、DFT 还是真实实验。

## 3. `.copy()` 为什么重要

```python
run_strategy(initial_indices.copy())
```

列表和 NumPy 数组传入函数后可能被原地修改。`.copy()` 建立独立副本，
防止先运行的策略改变后运行策略的初始集。

## 4. 列表推导式

```python
pool = [
    index
    for index in range(len(X_all))
    if index not in labeled
]
```

等价于：

```python
pool = []
for index in range(len(X_all)):
    if index not in labeled:
        pool.append(index)
```

它建立“所有尚未标注的位置”列表。

## 5. 局部位置和全局位置

```python
query_local = 2
query_global = pool[query_local]
```

`query_local` 是剩余池内的位置；`query_global` 才是永久数据行位置。
更新 `labeled` 和日志时必须保存全局位置或永久候选 ID。

## 6. `groupby(...).agg(...)`

```python
summary = runs.groupby(
    ["strategy", "round"],
    as_index=False,
).agg(
    regret_mean=("simple_regret", "mean"),
    regret_std=("simple_regret", "std"),
)
```

逐步理解：

1. `groupby` 按策略和轮次分组；
2. `regret_mean=(源列, 统计方法)` 创建新列；
3. 每个新列只汇总同组的多次种子结果；
4. `as_index=False` 让分组键继续是普通列，便于画图和保存 CSV。

## 7. `.loc` 按行列修改

```python
table.loc[
    table["feasible"],
    "review_status",
] = "approved"
```

逗号左边选择行，右边选择列。这里把所有 `feasible=True` 行的状态改为
`approved`。它只是教学模拟；真实审批必须由授权人员完成。

## 8. `zip` 同步遍历

```python
for candidate_id, status in zip(
    table["candidate_id"],
    table["review_status"],
):
    ...
```

`zip` 每次从两列各取一个值。两列长度不一致时会在较短列结束，
因此使用前要确认它们来自同一个 DataFrame。

## 9. 固定并列规则

```python
order = np.lexsort((candidate_ids, -score))
query_local = int(order[0])
```

- `lexsort` 的最后一个键是主排序键；
- 先按 `-score`，即分数从高到低；
- 分数相同时按候选 ID；
- 不再让数组当前顺序悄悄决定结果。

## 10. 常见 shape

| 变量 | 常见 shape | 含义 |
|---|---|---|
| `X_labeled` | `(n_labeled, n_features)` | 已获得标签的输入 |
| `y_labeled` | `(n_labeled,)` | 一维目标 |
| `X_pool` | `(n_pool, n_features)` | 可选择候选 |
| `member_predictions` | `(n_members, n_pool)` | 每个模型对每个候选的预测 |
| `prediction_mean` | `(n_pool,)` | 候选预测均值 |
| `prediction_std` | `(n_pool,)` | 候选不确定性或分歧 |

看到代码不懂时，先在个人 Notebook 加：

```python
print(type(variable))
print(np.asarray(variable).shape)
```

若希望保留课程源 Notebook 的预存输出，先在本机复制一份；个人笔记与结果放在 `learning_outputs/`。
