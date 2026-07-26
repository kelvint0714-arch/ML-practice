# ESOL 参考基线 E01（课程 Day 7 综合材料）

> 所属工作线：[B. 算法预研线](../../../curriculum/README.md)

> 算法入门学习：[Day 1 先理解算法，再接触代码](../../../curriculum/core/day01_beginner/README.md)

目录名 `day01_baseline` 和 Notebook 内部的 “Day 1” 是早期实验编号，为了保留历史路径和已记录的复现配置而不改名；它们不是当前课程 Day 1。当前课程在 Day 7 才运行这份参考基线。

## 目标

在固定 scaffold 划分和 1024 维 ECFP 特征上，建立一条可复现的传统机器学习基线，并判断单棵决策树是否明显过拟合。

## 实验输入

- 数据：ESOL / Delaney，1128 个分子；
- 划分：scaffold train/validation/test = 902/113/113；
- 特征：ECFP，radius=2，1024 维；
- 标签：原始 logS，`transformers=[]`；
- 候选模型：Dummy、Ridge、不限深树、限制树、随机森林；
- 选择主指标：validation RMSE。

更完整的数据来源与证据边界见 [ESOL 数据说明](../../../data/public/esol.md)。

## 运行

在仓库根目录执行：

```bash
conda activate esol-repro
python -m nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  experiments/esol/day01_baseline/esol_baseline.ipynb
```

第一次学习时不要直接从本页开始，也不要先修改模型参数。先按 [一步一步学习目录](../../../curriculum/PROGRESS.md) 学习算法；到 Day 7 再从空内核按顺序运行本 Notebook。Notebook 中保存的旧输出是历史快照，切换 Git 分支不会自动刷新输出文字。

## 产物

- `results/baseline_metrics.csv`：每个模型的 train/validation 长表指标；
- `results/run_config.json`：数据参数、模型参数、环境版本和 test 使用策略；
- `notes.md`：面向人的结果解释与下一步。

## 测试集边界

旧 Notebook 已经查看过当前 test split 上的随机森林分数。因此本参考基线不重复生成 test 预测，也不将该 split 包装成严格未见的最终证据。
