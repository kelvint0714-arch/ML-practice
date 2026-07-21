# ML-practice

一个面向化学与材料机器学习的可复现练习仓库。当前唯一执行主线是：先把 ESOL 分子溶解度传统机器学习基线做可信，再进入 MLP、GNN 和融合实验。

## 当前里程碑：ESOL Day 1

已纳入的基线：

- Dummy mean；
- Ridge；
- 不限深决策树（过拟合诊断）；
- 限制后决策树；
- 随机森林。

所有模型使用同一 ESOL scaffold 划分和 1024 维 ECFP，在原始 logS 标签空间统一比较 MAE、RMSE 和 R²。

当前一次固定验证结果中，随机森林的 validation RMSE 为 1.7031 logS、R² 为 0.2578，在五个事先固定的候选模型中最好。这只是单次划分证据，不是最终测试结论。

## 仓库入口

- [数据和划分协议](DATASETS.md)
- [ESOL 实验说明](practice/01_load_esol/readme.md)
- [ESOL Day 1 Notebook](practice/01_load_esol/01_load_esol.ipynb)
- [Day 1 实验结论](practice/01_load_esol/notes.md)
- [长期学习方案](Learning-plan.md)
- [长期资源清单](Processing-plan_resources.md)

`Learning-plan.md` 和 `Processing-plan_resources.md` 保留为长期参考；当前可执行任务以本 README 为准。

## 环境

已验证环境：Python 3.10.20、DeepChem 2.8.0、RDKit 2026.3.3 和 scikit-learn 1.7.2。完整版本见 `requirements.txt`。

```bash
conda create -n esol-repro python=3.10.20 -y
conda activate esol-repro
python -m pip install -r requirements.txt
```

DeepChem 导入时可能提示缺少 PyTorch、TensorFlow、JAX 或 PyG。它们是 Day 1 不需要的可选深度学习依赖，不影响 ECFP + scikit-learn 基线。

## 从头复现

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

首次运行可能需要联网下载 ESOL。数据和特征缓存会写入 `.cache/deepchem/`，该目录不进入 Git。

## 产物

执行 Notebook 后生成：

```text
practice/01_load_esol/results/
├── baseline_metrics.csv
└── run_config.json
```

Day 1 只用 train/validation 比较固定候选模型。旧 Notebook 已经查看过当前 test split，因此不将它包装成严格未见的最终证据。
