# Day 26：主动学习候选池模拟

> 状态：待学习、待模拟。今天的“揭示标签”只模拟实验返回，不代表真实实验已经完成。

## 本日完整学习包

1. [概念精讲](01_concepts.md)
2. [算法走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 是已有 ESOL 标签上的池模拟课程材料；它没有合成、测试或推荐真实下游任务。

## 今天为什么学

下游任务实验成本可能较高，主动学习希望推荐下一批候选。

核心规则是模型选择候选时不能偷看候选真实性能标签。

今天在已有练习数据上模拟一轮“选择—揭示—更新”，
并把随机选择作为对照。

## 前置条件

- 完成 [Day 25](../day25_uncertainty/README.md)；
- 能计算集成平均和启发式分歧；
- 已区分已标注集、候选池和固定外部对照集；
- 知道主动学习不能替代化学可行性审核；
- 采集规则在运行前已经固定。

## 今日产出

1. 一个初始已标注集和一个隐藏标签候选池；
2. 一轮基于模型分歧的 query；
3. 一轮同预算随机 query；
4. 在同一固定外部 valid 上的单轮指标对照；
5. 一份发给领域团队的候选交接字段草稿。

## 核心概念

### 1. 三个数据区域

- 已标注集：特征和标签都可用于训练；
- 候选池：选择时只能读取特征；
- 固定外部对照集：不加入候选池，只在 query 与更新完成后按同一口径比较。当前 ESOL valid 在课程前面已经被开发性查看过，所以它不是严格未见的最终 test。

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
11. 用固定外部 valid 检查更新前后指标；不把它称为严格独立 test。
12. 用同预算运行随机 query。
13. 写明本教程只演示一轮，单轮胜负不能证明策略优越。
14. 把多初始化种子、多轮学习曲线列为完成本教程后的正式实验扩展。

## 核心代码骨架

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

seeds = [11, 22, 33, 44, 55]
pool_predictions = []

for seed in seeds:
    model = RandomForestRegressor(
        n_estimators=120,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_labeled, y_labeled)
    pool_predictions.append(model.predict(X_pool))

prediction_matrix = np.vstack(pool_predictions)
acquisition_score = prediction_matrix.std(axis=0)
batch_size = 10
query_pos = np.lexsort((
    pool_ids.astype(str),
    -acquisition_score,
))[:batch_size]

# 同预算随机对照也在标签揭示前固定。
random_rng = np.random.default_rng(2026)
random_pos = random_rng.choice(
    len(X_pool), size=batch_size, replace=False
)

# 到这一行，主动 query 已固定；随机对照也必须在标签揭示前固定。
selected_X = X_pool[query_pos]
selected_ids = pool_ids[query_pos]
random_X = X_pool[random_pos]
random_ids = pool_ids[random_pos]

# 下面一行只模拟“实验完成后返回真实性能”。
# 模拟 oracle 也到标签揭示线之后才接收隐藏标签。
selected_y = y_pool[query_pos]
random_y = y_pool[random_pos]
X_labeled_next = np.concatenate([X_labeled, selected_X], axis=0)
y_labeled_next = np.concatenate([y_labeled, selected_y], axis=0)
X_random_next = np.concatenate([X_labeled, random_X], axis=0)
y_random_next = np.concatenate([y_labeled, random_y], axis=0)

print(selected_ids)
print(acquisition_score[query_pos])
print(random_ids)
```

## 只解释今天新增的语法

- `np.lexsort((candidate_id, -score))` 先按分歧从高到低排，并在分数相同时按候选 ID 排，保证并列规则固定。
- `[:batch_size]` 取排序后的前若干个位置。
- `X_pool[query_pos]` 按位置取出被选候选。
- `default_rng(2026).choice(..., replace=False)` 用固定种子抽取不重复的同预算随机对照。
- `np.concatenate(..., axis=0)` 按样本行拼接新旧数据。
- 注释位置划出了“选择前”和“标签返回后”的权限边界。

## 最重要的泄漏审计

在 `query_pos` 和 `selected_ids` 固定之前，
搜索代码并确认没有出现：

- `y_pool`；
- 候选绝对误差；
- 候选真实排名；
- 用候选标签调采集权重；
- 根据模拟结果反复改同一轮 query。

可以在 query 固定后用标签评价完整策略，
但不能让同一轮策略先看答案再选题。

## 真实下游任务接口从今天开始

算法组发给领域团队的 query 至少包含：

| 字段 | 用途 |
|---|---|
| 候选编号 | 保持往返对应 |
| 配方与工艺字段 | 让实验人员知道做什么 |
| 模型预测 | 仅供参考，不是真实性能 |
| 采集分数 | 说明选择理由 |
| 可行性状态 | 由领域团队审核 |
| 禁止原因 | 记录安全或工艺约束 |

领域团队返回：

- 实际测试值、单位、标准、重复次数和误差；
- 实际配方、偏差、实验日期、批次及失败原因。

未经领域团队批准，
算法不能直接把高分歧候选当成实验任务。

随机对照必须使用相同初始已标注集、预算、模型和外部 valid。
本日只完成单轮机制演示，因此结果表只说明代码流程；
若要讨论策略有效性，必须另做多轮、多预先声明种子的学习曲线，
不能只展示主动学习最好的一次和随机选择最差的一次。

## 专题深化入口

Day 26 是主动学习桥接课，不负责完整论文级比较。完成核心路线并满足先修条件后，进入 [主动学习 Unit 1–9 专题](../../active_learning/README.md)，继续学习：

- GP/RF/神经网络代理模型与不确定性；
- UCB、PI、EI、TS 等采集函数；
- 多轮、多种子和 regret 学习曲线；
- 批量、多样性、成本和化学约束；
- GNN/PBNN/DKL 与物理先验在闭环中的位置；
- 论文复现与真实实验 Oracle 接口。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| query 前读取 `y_pool` | 直接标签泄漏 | 固定后才揭示 |
| 只和旧模型比 | 无法评价选择策略 | 加同预算随机对照 |
| 候选含不可行配方 | 实验无法执行 | 先做化学约束过滤 |
| 一轮成功就下结论 | 偶然性很大 | 多初始种子重复 |
| 把模拟称为实验 | 证据等级错误 | 明确“池模拟” |

## 完成标准

- [ ] 已标注集、候选池和固定外部 valid 互相区分；
- [ ] query 前代码没有读取候选标签；
- [ ] 候选只根据特征和冻结采集函数排序；
- [ ] 标签仅在 query 固定后模拟揭示；
- [ ] 已建立同预算随机对照；
- [ ] 候选交接表包含化学可行性审核；
- [ ] 没有声称真实实验已经完成。
- [ ] 没有把单轮胜负或课程 valid 包装成策略优越性/最终性能；

## 自测问题

1. 候选池与固定外部对照集有什么区别？为什么课程 valid 仍不是严格未见 test？
2. 为什么 `selected_y` 必须出现在 query 固定之后？
3. 主动学习为什么仍需要随机对照？
4. 高分歧候选为什么不能自动送实验？
5. 真实实验返回时至少要记录哪些信息？

## 上一天 / 下一天

- 上一天：[Day 25：启发式不确定性](../day25_uncertainty/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 27：从论文映射到数据字段](../day27_paper_to_schema/README.md)
- 后续专题：[主动学习 Unit 路线](../../active_learning/README.md)
