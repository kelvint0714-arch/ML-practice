# ESOL Day 1 Notebook 逐行讲解

原 Notebook 有 10 个代码单元、212 行非空物理代码。为了让手机和 GitHub 页面更容易打开，逐行解释分为上下两部分；两部分合起来覆盖全部代码。

- [上半部分：环境、加载数据、数据检查、准备 X/y](notebook_line_by_line_part1.md)
- [下半部分：指标、模型、训练、排名、保存与验收](notebook_line_by_line_part2.md)

阅读时请同时打开原始 [`01_load_esol.ipynb`](../../../practice/01_load_esol/01_load_esol.ipynb)。教程中的 `C9:L06` 表示“原 Notebook 的代码 Cell 9、本单元第 6 个物理代码行”，不是整个文件的第 906 行。

## 三种学习标签

| 标签 | 要求 |
|---|---|
| `[核心]` | Day 1 必须能用自己的话解释 |
| `[工程]` | 知道它为复现、检查或保存服务即可 |
| `[高级]` | 逐行说明仍然提供，但今天不要求独立写出 |

不要因为第一格出现 Git、路径和异常处理就认为自己不适合机器学习。第一格大部分是工程复现代码。第一次阅读可以先看上半部分中的“加载数据”和“准备 X/y”，再回头看环境记录。

## 变量与类型/形状总地图

| 变量 | 运行时类型 | 形状或长度 | 本实验含义 |
|---|---|---:|---|
| `SEED` | `int` | — | 随机种子 42 |
| `RUN_ID` | `str` | — | 本次实验名称 |
| `REPO_ROOT` | `pathlib.Path` | — | 仓库根目录 |
| `DATA_DIR` | `pathlib.Path` | — | DeepChem 缓存目录 |
| `RESULTS_DIR` | `pathlib.Path` | — | 输出目录 |
| `VERSIONS` | `dict[str, str]` | 7 项 | Python 与依赖版本 |
| `featurizer` | DeepChem `CircularFingerprint` | 1024 位、radius 2 | 把分子转换为 ECFP 数值表示 |
| `tasks` | `list[str]` | 1 项 | 预测目标名称 |
| `datasets` | 3 个 DeepChem Dataset 的序列 | 3 项 | train/valid/test |
| `transformers` | `list` | 0 项 | 没有标签变换 |
| `train_dataset.X` | NumPy 类二维数组 | `(902, 1024)` | 训练输入 |
| `train_dataset.y` | NumPy 类二维数组 | `(902, 1)` | 训练标签原形状 |
| `valid_dataset.X` | NumPy 类二维数组 | `(113, 1024)` | 验证输入 |
| `valid_dataset.y` | NumPy 类二维数组 | `(113, 1)` | 验证标签原形状 |
| `X_train` | `numpy.ndarray` | `(902, 1024)` | sklearn 使用的训练输入 |
| `y_train` | `numpy.ndarray` | `(902,)` | sklearn 使用的一维训练标签 |
| `X_valid` | `numpy.ndarray` | `(113, 1024)` | sklearn 使用的验证输入 |
| `y_valid` | `numpy.ndarray` | `(113,)` | sklearn 使用的一维验证标签 |
| `models` | `dict[str, estimator]` | 5 项 | 五个尚未训练的候选模型 |
| `fitted_models` | `dict[str, estimator]` | 5 项 | `.fit()` 后的模型 |
| `records` | `list[dict]` | 最后 10 项 | 5 个模型 × 2 个数据集的记录 |
| `results` | `pandas.DataFrame` | `(10, 14)` | 指标长表 |
| `valid_ranked` | `pandas.DataFrame` | `(5, 14)` | 仅验证行并按误差排序 |
| `summary` | `pandas.DataFrame` | `(5, 6)` | 模型 × train/valid 指标宽表 |

## 先弄清两个容易混淆的事实

### 1. Notebook 保存的输出是历史快照

当前 Git 分支是 `agent/data-algorithm-tracks`，但 Notebook 页面中保存的输出仍显示：

```text
Branch: baseline/esol-repro
Source commit: 5bd7b5c...
```

这不是 Python 自相矛盾。`.ipynb` 文件会把上次运行的输出文字一起保存；切换 Git 分支只更换文件状态，不会自动重新执行代码。重新从空内核运行后，分支、提交和耗时可能变化。

### 2. logS 不是“百分比准确率”

本任务预测的是溶解度的对数空间，通常理解为 `log10(溶解度 / mol·L⁻¹)`。MAE 和 RMSE 因此处在 logS 数值尺度上，不是百分比；R² 也不是准确率。

## 阅读完成标准

阅读两部分后，你应能沿着下面的数据流指给别人看：

```text
分子 → ECFP → X_train/y_train → model.fit
                                  ↓
X_valid → model.predict → 指标 → records → results → 排名与保存
```

如果暂时说不出 Git、JSON 或 pandas `pivot` 的完整语法没有关系；如果分不清 `X`、`y`、`.fit()` 和 `.predict()`，则还不能进入 Day 2。
