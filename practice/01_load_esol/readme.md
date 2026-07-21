# ESOL Day 1 实验

## 目标

在固定 scaffold 划分和 1024 维 ECFP 特征上，建立一条可复现的传统机器学习基线，并判断单棵决策树是否明显过拟合。

## 实验输入

- 数据：ESOL / Delaney，1128 个分子；
- 划分：scaffold train/validation/test = 902/113/113；
- 特征：ECFP，radius=2，1024 维；
- 标签：原始 logS，`transformers=[]`；
- 候选模型：Dummy、Ridge、不限深树、限制树、随机森林；
- 选择主指标：validation RMSE。

更完整的数据来源与证据边界见根目录 [`DATASETS.md`](../../DATASETS.md)。

## 运行

在仓库根目录执行：

```bash
conda activate esol-repro
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  practice/01_load_esol/01_load_esol.ipynb
```

## 产物

- `results/baseline_metrics.csv`：每个模型的 train/validation 长表指标；
- `results/run_config.json`：数据参数、模型参数、环境版本和 test 使用策略；
- `notes.md`：面向人的结果解释与下一步。

## 测试集边界

旧 Notebook 已经查看过当前 test split 上的随机森林分数。因此 Day 1 不重复生成 test 预测，也不将该 split 包装成严格未见的最终证据。
