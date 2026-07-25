# ML-practice

面向化学与材料机器学习的学习与实验仓库。仓库把“课程任务”“真实实验”“数据接口”和“参考资料”分开，避免把已写好的任务卡误认为已经完成的实验。

## 现在从这里开始

1. 打开唯一导航：[一步一步学习目录](curriculum/PROGRESS.md)
2. 永远从目录里的第一个 `[ ]` 开始，完成并通过自测后再勾选
3. 目录会从 Day 1 的第一篇讲解，一直带你走到 Day 28；GNN 是通过门槛后才开始的可选内容

`Day` 是学习单元，不是必须一天完成的期限。Day 1 没有完成自测前，不要急着跳到后面的模型。

## 两条研究工作线

| 工作线 | 当前状态 | 入口 | 现阶段交付 |
|---|---|---|---|
| 粘合剂数据线 | 等待导师与化学组确认体系、字段和真实样例 | [数据线说明](data/adhesive/README.md) | 数据字典、3–5 行格式样例、首批获授权真实数据 |
| 算法预研线 | ESOL Day 1 已可复现；学习从 Day 1 开始 | [算法线说明](curriculum/README.md) | 传统模型、MLP、混合模型与主动学习的统一对照流程 |

两条线的边界和衔接条件见 [项目路线图](docs/project_roadmap.md)。公开数据只用于方法开发；真实粘合剂数据到达前，不把 ESOL 结果表述为粘合剂实验结论。没有可靠结构表示前，不启动粘合剂 GNN。

## 仓库结构

```text
ML-practice/
├── .github/workflows/       # GitHub 自动仓库检查
├── curriculum/             # 学习任务：核心 28 天、可选 GNN、共享词典
├── data/                   # 公开数据说明与粘合剂数据接口
├── experiments/            # 真正运行过或正在执行的实验
├── docs/                   # 项目路线、参考资料和旧方案归档
├── scripts/                # 仓库检查工具
├── requirements.txt        # 已复现实验的完整环境
└── requirements-learning.txt
```

详细入口：

- [一步一步学习目录](curriculum/PROGRESS.md)
- [课程总览](curriculum/README.md)
- [Day 1–28 核心课程](curriculum/core/README.md)
- [Day 29–35 可选 GNN](curriculum/optional_gnn/README.md)
- [数据目录](data/README.md)
- [实验目录](experiments/README.md)
- [项目文档](docs/README.md)

## 当前可复现实验

ESOL Day 1 使用固定 scaffold 划分、1024 维 ECFP，以及 Dummy、Ridge、决策树和随机森林基线。实验说明、结果和证据边界都保存在 [实验目录](experiments/esol/day01_baseline/README.md)；公开数据来源见 [ESOL 数据说明](data/public/esol.md)。

任务卡已经写好不等于本人已经完成。`curriculum/` 描述“应该做什么”，`experiments/` 只保存“实际做过什么”。

## 环境与复现

已验证环境为 Python 3.10.20、DeepChem 2.8.0、RDKit 2026.3.3 和 scikit-learn 1.7.2。

```bash
conda create -n esol-repro python=3.10.20 -y
conda activate esol-repro
python -m pip install -r requirements-learning.txt

python -m nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  experiments/esol/day01_baseline/esol_baseline.ipynb
```

首次运行可能需要联网下载 ESOL；缓存写入 `.cache/deepchem/`，不进入 Git。

## 仓库自检

```bash
python scripts/check_repository.py
```

检查器验证目录结构、35 个任务卡、Markdown 链接、Python 代码块、Notebook、ESOL 结果文件和 Excel 包结构。GitHub Actions 会在 `main` 的推送和 Pull Request 上自动运行同一检查，但不会把任何学习日自动标为完成。

## 分支规则

`main` 是唯一长期分支。课程、数据、实验和文档修改使用短期分支，检查通过并合并后删除。完整规则见 [分支与仓库维护规则](docs/development_workflow.md)。

## 数据安全边界

- 仓库中的粘合剂 Excel 是空白字段讨论稿，不是真实数据集；
- 真实配方或实验记录上传前必须确认保密、署名和共享权限；
- 未授权数据放在受控位置，不提交到 GitHub；
- 3–5 行样例只用于格式验收，不能作为可靠训练数据。
