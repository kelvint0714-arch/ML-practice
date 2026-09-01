# ESOL 参考基线

这是 Day 7 配套的完整、可运行参考实现。它把前六天分别学习的模型、指标和数据划分串成一条端到端流程，供学习者阅读、复现和审计；它不是个人实验工作区。

## 学习目标

- 在固定 scaffold 划分和 1024 维 ECFP 特征上训练传统回归基线；
- 比较 Dummy、Ridge、决策树和随机森林；
- 观察不限深决策树的过拟合迹象；
- 理解配置、指标和环境版本为何需要一起保存。

## 数据与协议

- 数据：公开 ESOL / Delaney，共 1128 个分子；
- 划分：scaffold train/validation/test = 902/113/113；
- 特征：ECFP，radius=2，1024 维；
- 标签：原始 logS；
- 模型选择指标：validation RMSE。

数据来源和适用边界见 [ESOL 数据说明](../../../../data/public/esol.md)。

## 运行方式

先完成 [Day 7 教学单元](../README.md)，再在仓库根目录执行：

```bash
conda activate esol
python -m nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  curriculum/core/day07_integrated_baseline/reference_baseline/esol_baseline.ipynb
```

请从空内核按顺序运行，不要通过反复查看测试集来选择模型。

## 保存内容

- `results/baseline_metrics.csv`：各模型的训练集和验证集指标；
- `results/run_config.json`：数据、特征、模型、版本和测试集策略；
- Notebook 保存输出：帮助学习者核对自己的运行过程。

## 证据边界

旧版实现曾查看当前 test split，因此这里保留其历史说明，但不把该 split 宣称为严格未见的最终测试集。保存的分数只属于公开 ESOL 教学协议，不能自动外推到其他数据集或下游任务。

返回：[Day 7](../README.md) · [核心课程目录](../../README.md) · [总进度](../../../PROGRESS.md)
