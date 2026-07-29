# Unit 04 参考答案

## 练习 1

`query_global = pool_indices[2] = 7`。

## 练习 2

```python
assert set(labeled_indices).isdisjoint(pool_indices)
assert len(labeled_indices) == len(set(labeled_indices))
assert len(labeled_indices) + len(pool_indices) == n_candidates
```

## 练习 3

query 前：`run_seed`、`strategy`、`round`、`candidate_id`、`prediction_mean`、`prediction_std`、`acquisition_score`、`model_version`。query 后：`observed_y`、`best_so_far`、`simple_regret`。

## 练习 4

差异可能来自初始集而不是策略。应共享每个种子的初始集并进行多次重复。

## 练习 5

合理：预算耗尽、池为空、达到预先定义目标。不合理：看到 UCB 领先后立即停止，但 Random 落后时继续运行。
