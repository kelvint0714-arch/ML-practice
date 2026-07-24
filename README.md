# ML-practice

一个面向化学与材料机器学习的可复现练习仓库。当前采用“两线并行”：化学组继续确认粘合剂数据，算法侧同时完成可迁移的基线、神经网络与融合方法预研。

## 当前工作方式：两线并行

| 工作线 | 当前状态 | 当前入口 | 本阶段交付 |
|---|---|---|---|
| A. 粘合剂数据线 | 等待导师与化学组确认字段并提供样例 | [数据线说明](tracks/01_data/README.md) | 经确认的数据字典、3–5 行格式样例、首批真实数据 |
| B. 算法预研线 | ESOL Day 1 传统基线已完成 | [算法线说明](tracks/02_algorithm/README.md) | 传统模型、MLP、混合模型的统一对照流程 |

两条线的边界、衔接条件和本周任务见 [双线工作总览](tracks/README.md)。公开数据只用于方法开发和流程验收；在真实粘合剂数据到达前，不将公开数据结果表述为本项目的粘合剂实验结论。没有 SMILES、分子图或可靠结构文件前，不启动 GNN 实验。

## 算法线当前里程碑：ESOL Day 1

已纳入的基线：

- Dummy mean；
- Ridge；
- 不限深决策树（过拟合诊断）；
- 限制后决策树；
- 随机森林。

所有模型使用同一 ESOL scaffold 划分和 1024 维 ECFP，在原始 logS 标签空间统一比较 MAE、RMSE 和 R²。

当前一次固定验证结果中，随机森林的 validation RMSE 为 1.7031 logS、R² 为 0.2578，在五个事先固定的候选模型中最好。这只是单次划分证据，不是最终测试结论。

## 仓库入口

- [双线工作总览](tracks/README.md)
- [A. 粘合剂数据线](tracks/01_data/README.md)
- [B. 算法预研线](tracks/02_algorithm/README.md)
- [数据和划分协议](DATASETS.md)
- [ESOL 实验说明](practice/01_load_esol/readme.md)
- [ESOL Day 1 Notebook](practice/01_load_esol/01_load_esol.ipynb)
- [ESOL Day 1 零基础逐行学习入口](tracks/02_algorithm/day01_beginner/README.md)
- [Day 1 实验结论](practice/01_load_esol/notes.md)
- [粘合剂重要化学性质数据格式 v0.3](data_templates/adhesive/README.md)
- [长期学习方案](Learning-plan.md)
- [长期资源清单](Processing-plan_resources.md)

`Learning-plan.md` 和 `Processing-plan_resources.md` 保留为长期参考；当前可执行任务以本 README 为准。

## 粘合剂数据准备

`data_templates/adhesive/` 保存了一份不预设环氧体系的粘合剂数据格式讨论稿。它用于让导师和化学组确认字段，尚不是真实数据集，也不代表已经选定了最终模型。

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
