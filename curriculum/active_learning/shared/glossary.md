# 主动学习术语表

| 术语 | 中文解释 | 本课程变量 |
|---|---|---|
| labeled set | 已经有真实性能标签、允许训练的数据 | `X_labeled`, `y_labeled` |
| pool | 尚未获取标签的候选集合 | `X_pool`, `pool_ids` |
| surrogate model | 用较便宜预测近似昂贵实验的模型 | GP、RF、MLP、GNN 等 |
| uncertainty | 模型对预测缺乏把握的程度 | `prediction_std` |
| acquisition function | 把预测与不确定性变成选样分数的规则 | UCB、EI、PI 等 |
| query | 本轮决定请求标签的候选 | `query_ids` |
| Oracle | 返回真实性能标签的来源 | 公开数据查表、DFT、真实实验 |
| budget | 允许新增多少标签/实验 | `total_budget` |
| batch size | 每轮同时选择多少个候选 | `batch_size` |
| exploration | 优先了解未知区域 | 高不确定性 |
| exploitation | 优先测试预测性能好的候选 | 高预测均值 |
| best-so-far | 截至当前已测样本中的最好性能 | `best_observed` |
| simple regret | 全局最优与当前已发现最优的差 | `global_best - best_observed` |
| calibration | 预测不确定性是否具有可靠统计含义 | 覆盖率、NLPD 等 |
| diversity | 避免同一批候选过于相似 | 距离、聚类、去重 |
| feasibility | 候选是否安全、可制备、满足比例和工艺约束 | `feasible_mask` |
| aleatoric uncertainty | 测量或过程本身的随机噪声 | 重复实验估计 |
| epistemic uncertainty | 因有限数据造成的模型认知不足 | GP 后验、集成分歧 |
| offline replay | 在已有完整数据上隐藏标签，模拟逐轮实验 | 本课程主要模式 |

术语出现时要回到具体变量：它读取什么、什么时候可见、由谁产生。只会背英文不算掌握。
