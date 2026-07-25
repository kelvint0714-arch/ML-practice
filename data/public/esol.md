# 数据说明

## ESOL / Delaney

### 任务

- 类型：单任务分子性质回归；
- 样本数：1128；
- 标签：`measured log solubility in mols per litre`；
- 目标空间：原始 logS，Day 1 不进行标签标准化。

### 来源

- 加载器：DeepChem `dc.molnet.load_delaney`；
- 原始工作：John S. Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*, 2004；
- 论文：<https://pubs.acs.org/doi/10.1021/ci034243x>；
- DeepChem 入口：<https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#delaney-datasets>。

原始数据不提交到本仓库，由 DeepChem 下载并缓存到 `.cache/deepchem/`。本仓库尚未独立核验原始数据的重新分发条款，因此在对外重新分发数据前需要再次确认许可。

### Day 1 配置

| 项目 | 设置 |
|---|---|
| DeepChem | 2.8.0 |
| Featurizer | ECFP，radius=2，1024 维 |
| Splitter | scaffold |
| Transformers | `[]` |
| Train / Validation / Test | 902 / 113 / 113 |
| 模型随机种子 | 42 |
| 评价指标 | MAE、RMSE、R² |
| 模型选择主指标 | validation RMSE |

### 复现边界

- 首次运行可能需要联网下载数据；
- 特征可能受 DeepChem 和 RDKit 版本影响，因此需要保留精确环境版本；
- 902 / 113 / 113 和 1024 维特征是当前加载验收条件；
- train/validation 用于 Day 1 固定候选模型比较；
- 旧 Notebook 已暴露当前 test split 的随机森林分数，本轮不生成新的 test 预测；
- 本阶段只验证 ECFP 传统机器学习基线，不代表最终分子图模型结果。
