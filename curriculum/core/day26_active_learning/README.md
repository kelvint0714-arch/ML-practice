# Day 26：主动学习候选池模拟

> 状态：待学习、待模拟。今天的“揭示标签”只模拟实验返回，不代表真实实验已经完成。

## 今天为什么学

粘合剂实验成本可能较高，主动学习希望推荐下一批候选。

核心规则是模型选择候选时不能偷看候选真实性能标签。

今天在已有练习数据上模拟一轮“选择—揭示—更新”，
并把随机选择作为对照。

## 前置条件

- 完成 [Day 25](../day25_uncertainty/README.md)；
- 能计算集成平均和启发式分歧；
- 已区分已标注集、候选池和独立测试集；
- 知道主动学习不能替代化学可行性审核；
- 采集规则在运行前已经固定。

## 今日产出

1. 一个初始已标注集和一个隐藏标签候选池；
2. 一轮基于模型分歧的 query；
3. 一轮同预算随机 query；
4. query 前后模型指标对照；
5. 一份发给化学组的候选交接字段草稿。

## 核心概念

### 1. 三个数据区域

- 已标注集：特征和标签都可用于训练；
- 候选池：选择时只能读取特征；
- 独立评估集：只用于比较策略，不加入候选池。

### 2. Query

Query 是本轮准备请求实验的候选编号。
它可以基于预测、模型分歧、代表性和可行性约束，
但不能基于候选真实标签。

### 4. 主动策略必须有对照

同样的初始数据和新增预算下，
与随机选择比较，
才能判断主动策略是否提供额外价值。

## 分步骤任务

1. 从练习训练数据建立小型已标注集。
2. 将剩余训练样本作为候选池。
3. 把候选标签保存在选择代码不会访问的位置。
4. 在已标注集上训练多个集成成员。
5. 只用 `X_pool` 生成候选预测矩阵。
6. 计算每个候选的分歧分数。
7. 根据分歧和固定预算确定 `query_ids`。
8. 保存被选编号和选择理由。
9. 此时才模拟读取 `y_pool[query_ids]`。
10. 把新样本加入已标注集并重新训练。
11. 用独立评估集检查更新前后指标。
12. 用同预算运行随机 query。
13. 重复多个初始种子，避免单次偶然。
14. 写明模拟结果不能替代真实实验验证。

## 核心代码骨架

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

seeds = [11, 22, 33, 44, 55]
pool_predictions = []

for seed in seeds:
    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        random_state=seed,
    )
    model.fit(X_labeled, y_labeled)
    pool_predictions.append(model.predict(X_pool))

prediction_matrix = np.vstack(pool_predictions)
acquisition_score = prediction_matrix.std(axis=0)
batch_size = 5
query_ids = np.argsort(acquisition_score)[-batch_size:][::-1]

# 到这一行，query 已固定；此前不得读取 y_pool。
selected_X = X_pool[query_ids]
selected_ids = pool_ids[query_ids]

# 下面一行只模拟“实验完成后返回真实性能”。
selected_y = y_pool[query_ids]
X_labeled_next = np.concatenate([X_labeled, selected_X], axis=0)
y_labeled_next = np.concatenate([y_labeled, selected_y], axis=0)

print(selected_ids)
print(acquisition_score[query_ids])
```

## 只解释今天新增的语法

- `np.argsort(score)` 返回按分数从小到大排列的索引。
- `[-batch_size:]` 取最高分的最后若干个索引。
- `[::-1]` 把顺序反转为从高到低。
- `X_pool[query_ids]` 按索引取出被选候选。
- `np.concatenate(..., axis=0)` 按样本行拼接新旧数据。
- 注释位置划出了“选择前”和“标签返回后”的权限边界。

## 最重要的泄漏审计

在 `query_ids` 固定之前，
搜索代码并确认没有出现：

- `y_pool`；
- 候选绝对误差；
- 候选真实排名；
- 用候选标签调采集权重；
- 根据模拟结果反复改同一轮 query。

可以用标签评价完整策略，
但不能让同一轮策略先看答案再选题。

## 真实粘合剂接口从今天开始

算法组发给化学组的 query 至少包含：

| 字段 | 用途 |
|---|---|
| 候选编号 | 保持往返对应 |
| 配方与工艺字段 | 让实验人员知道做什么 |
| 模型预测 | 仅供参考，不是真实性能 |
| 采集分数 | 说明选择理由 |
| 可行性状态 | 由化学组审核 |
| 禁止原因 | 记录安全或工艺约束 |

化学组返回：

- 实际测试值、单位、标准、重复次数和误差；
- 实际配方、偏差、实验日期、批次及失败原因。

未经化学组批准，
算法不能直接把高分歧候选当成实验任务。

随机对照必须使用相同初始已标注集、每轮预算、总轮数和评估集，
并覆盖多个预先声明的种子。
不能只展示主动学习最好的一次和随机选择最差的一次。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| query 前读取 `y_pool` | 直接标签泄漏 | 固定后才揭示 |
| 只和旧模型比 | 无法评价选择策略 | 加同预算随机对照 |
| 候选含不可行配方 | 实验无法执行 | 先做化学约束过滤 |
| 一轮成功就下结论 | 偶然性很大 | 多初始种子重复 |
| 把模拟称为实验 | 证据等级错误 | 明确“池模拟” |

## 完成标准

- [ ] 已标注集、候选池和评估集互相区分；
- [ ] query 前代码没有读取候选标签；
- [ ] 候选只根据特征和冻结采集函数排序；
- [ ] 标签仅在 query 固定后模拟揭示；
- [ ] 已建立同预算随机对照；
- [ ] 候选交接表包含化学可行性审核；
- [ ] 没有声称真实实验已经完成。

## 自测问题

1. 候选池与独立评估集有什么区别？
2. 为什么 `selected_y` 必须出现在 query 固定之后？
3. 主动学习为什么仍需要随机对照？
4. 高分歧候选为什么不能自动送实验？
5. 真实实验返回时至少要记录哪些信息？

## 上一天 / 下一天

- 上一天：[Day 25：启发式不确定性](../day25_uncertainty/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 27：从论文映射到数据字段](../day27_paper_to_schema/README.md)
