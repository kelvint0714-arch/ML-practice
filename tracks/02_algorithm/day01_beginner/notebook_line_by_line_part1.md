# ESOL Day 1 Notebook 逐行讲解（第一部分）

对应 Notebook：

`practice/01_load_esol/01_load_esol.ipynb`

本文假设读者刚开始学习 Python。它只讲四个代码单元：

- 代码单元 1：导入、随机种子、路径、Git 和版本信息；
- 代码单元 3：建立 ECFP 特征器并加载 ESOL；
- 代码单元 4：检查数据数量、形状、数值和精确 SMILES 重叠；
- 代码单元 6：建立 sklearn 使用的 `X_train`、`y_train`、`X_valid`、`y_valid`。

模型定义和训练不在本文件中。

---

## 1. 怎样阅读行号

`C3:L05` 表示：

- `C3`：Notebook JSON 中的代码单元 3；
- `L05`：该代码单元内部第 5 个物理行；
- 空白行没有执行作用，因此不编号解释；
- 单独的 `)`、`}` 等行也会计入，并说明它关闭了哪个结构。

代码单元序号会从 1 跳到 3，是因为中间还有 Markdown 单元，不是代码缺失。

---

## 2. 这四个单元在做什么

```text
导入 Python 工具
        ↓
找到 ML-practice 仓库和缓存目录
        ↓
加载 ESOL 分子数据
        ↓
SMILES → 1024 维 ECFP
        ↓
scaffold 划分 train / valid / test
        ↓
检查形状、有限值和精确 ID 重叠
        ↓
得到可供 sklearn 使用的 X 和 y
```

最重要的四个数组是：

| 对象 | 类型 | 形状 | 意义 |
|---|---|---:|---|
| `X_train` | `numpy.ndarray` | `(902, 1024)` | 902 个训练分子，每个分子 1024 个输入特征 |
| `y_train` | `numpy.ndarray` | `(902,)` | 902 个训练分子的真实 logS |
| `X_valid` | `numpy.ndarray` | `(113, 1024)` | 113 个验证分子的输入特征 |
| `y_valid` | `numpy.ndarray` | `(113,)` | 113 个验证分子的真实 logS |

`X` 是模型看到的输入，`y` 是希望模型预测的答案。

---

## 3. 先理解 logS

ESOL 的目标名称是：

```text
measured log solubility in mols per litre
```

可理解为“以摩尔每升表示的溶解度的对数”。该数据集通常使用：

```text
logS = log10(S)
```

| 原始浓度 `S` | `logS` |
|---:|---:|
| `1 mol/L` | `0` |
| `0.1 mol/L` | `-1` |
| `0.01 mol/L` | `-2` |
| `0.001 mol/L` | `-3` |

所以必须记住：

- logS 为负不代表原始浓度为负；
- logS 相差 1，原始浓度大约相差 10 倍；
- 后续的 MAE、RMSE 使用 logS 单位，不是百分比。

---

# 代码单元 1：导入和环境记录

## 4. 单元目的

本单元不读取 ESOL，也不训练模型。它负责：

1. 导入库；
2. 固定随机种子；
3. 寻找 Git 仓库；
4. 创建缓存和结果目录；
5. 记录分支、commit、工作区状态和依赖版本。

## 5. 输入、输出、类型

| 名称 | 类型 | 值或作用 |
|---|---|---|
| 当前目录 | `Path` | Jupyter 启动位置 |
| `SEED` | `int` | `42` |
| `RUN_ID` | `str` | `esol_ecfp_scaffold_seed42` |
| `REPO_ROOT` | `Path` | Git 仓库根目录 |
| `DATA_DIR` | `Path` | `.cache/deepchem` |
| `RESULTS_DIR` | `Path` | `practice/01_load_esol/results` |
| `SOURCE_COMMIT` | `str` | 运行时 commit 哈希 |
| `BRANCH` | `str` | 运行时分支 |
| `GIT_DIRTY` | `bool` | 是否检测到未提交状态，另有 caveat |
| `VERSIONS` | `dict[str, str]` | Python 和依赖版本 |

## 6. 每个非空物理行

| 行号 | 原代码 | 结构、类型和作用 |
|---|---|---|
| `C1:L01` | `from pathlib import Path` | 用 `from 模块 import 名称` 导入 `Path` 类。它用于安全表示文件路径。 |
| `C1:L02` | `import json` | 导入标准库 `json` 模块，后面把字典序列化为 JSON 文本。 |
| `C1:L03` | `import platform` | 导入 `platform`，用于读取 Python 版本。 |
| `C1:L04` | `import subprocess` | 导入外部进程工具，后面用它运行只读 Git 命令。 |
| `C1:L05` | `import time` | 导入计时模块；这里只导入，尚未开始计时。 |
| `C1:L07` | `from rdkit import RDLogger` | 从 RDKit 导入日志控制器。RDKit 是化学信息学工具。 |
| `C1:L08` | `RDLogger.DisableLog("rdApp.warning")` | 调用方法隐藏 RDKit warning。它不修复有问题的分子，只是不显示这类日志。 |
| `C1:L10` | `import deepchem as dc` | 导入 DeepChem，并绑定常用别名 `dc`。 |
| `C1:L11` | `import numpy as np` | 导入 NumPy 为 `np`，用于数值数组。 |
| `C1:L12` | `import pandas as pd` | 导入 pandas 为 `pd`，用于结果表格。 |
| `C1:L13` | `import rdkit` | 导入 RDKit 顶层包，主要为了读取 `rdkit.__version__`。 |
| `C1:L14` | `import scipy` | 导入 SciPy；本单元主要记录其版本。 |
| `C1:L15` | `import sklearn` | 导入 scikit-learn 顶层包，主要记录版本。 |
| `C1:L16` | `from IPython.display import Markdown, display` | 导入 Jupyter 富文本显示工具，不是机器学习模型。 |
| `C1:L17` | `from sklearn.dummy import DummyRegressor` | 导入预测均值等简单规则的回归基线。 |
| `C1:L18` | `from sklearn.ensemble import RandomForestRegressor` | 导入随机森林回归类。 |
| `C1:L19` | `from sklearn.linear_model import Ridge` | 导入带 L2 正则化的线性回归类。 |
| `C1:L20` | `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score` | 一次导入 MAE、MSE 和 R² 三个评价函数。 |
| `C1:L21` | `from sklearn.tree import DecisionTreeRegressor` | 导入决策树回归类。 |
| `C1:L23` | `SEED = 42` | 用 `=` 把整数 42 赋给变量。全大写是“常量”命名习惯，但 Python 不强制。 |
| `C1:L24` | `RUN_ID = f"esol_ecfp_scaffold_seed{SEED}"` | f-string 把 `{SEED}` 替换为 42，得到实验标识字符串。 |
| `C1:L25` | `np.random.seed(SEED)` | 设置 NumPy 全局随机种子；它不会自动控制所有库。 |
| `C1:L27` | `def find_repo_root(start=Path.cwd().resolve()):` | 定义函数。默认参数是定义函数时求得的当前绝对路径。冒号后进入缩进函数体。 |
| `C1:L28` | `for candidate in (start, *start.parents):` | 遍历当前路径和所有父路径。`*` 把父路径序列展开进元组。 |
| `C1:L29` | `if (candidate / ".git").exists():` | `/` 对 `Path` 表示连接路径；`.exists()` 返回布尔值。 |
| `C1:L30` | `return candidate` | 找到 `.git` 后立即返回当前候选目录并结束函数。 |
| `C1:L31` | `raise RuntimeError(...)` | 全部路径都找不到 `.git` 时主动抛错并停止。 |
| `C1:L33` | `def git_output(*args):` | 定义可接收任意个位置参数的函数；`args` 在函数内是元组。 |
| `C1:L34` | `try:` | 开始异常处理：先尝试执行缩进代码。 |
| `C1:L35` | `return subprocess.check_output(` | 开始一个跨三行的外部命令调用，并准备返回结果。 |
| `C1:L36` | `["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL` | 构造 Git 命令列表；在仓库运行；输出转成字符串；隐藏 stderr。 |
| `C1:L37` | `).strip()` | 关闭 `check_output` 调用，并删除返回字符串首尾空白和换行。与 L35–L36 是同一表达式。 |
| `C1:L38` | `except (OSError, subprocess.CalledProcessError):` | 捕获操作系统错误或 Git 非零退出错误。 |
| `C1:L39` | `return "unavailable"` | Git 读取失败时返回非空字符串，而不是继续抛错。 |
| `C1:L41` | `REPO_ROOT = find_repo_root()` | 调用函数，得到仓库根目录 `Path`。 |
| `C1:L42` | `DATA_DIR = REPO_ROOT / ".cache" / "deepchem"` | 建立 DeepChem 缓存路径对象；此时尚未创建目录。 |
| `C1:L43` | `RESULTS_DIR = REPO_ROOT / "practice" / "01_load_esol" / "results"` | 建立结果目录路径对象。 |
| `C1:L44` | `DATA_DIR.mkdir(parents=True, exist_ok=True)` | 创建缓存目录；缺失父目录也创建；已存在不报错。 |
| `C1:L45` | `RESULTS_DIR.mkdir(parents=True, exist_ok=True)` | 用同样规则创建结果目录。 |
| `C1:L47` | `SOURCE_COMMIT = git_output("rev-parse", "HEAD")` | 运行 `git rev-parse HEAD`，得到 commit 哈希字符串。 |
| `C1:L48` | `BRANCH = git_output("branch", "--show-current")` | 运行 Git 命令，得到运行时分支字符串。 |
| `C1:L49` | `GIT_DIRTY = bool(git_output("status", "--porcelain"))` | 空 Git 输出转为 `False`，非空输出转为 `True`；失败回退也会误变成 `True`。 |
| `C1:L50` | `VERSIONS = {` | 开始版本字典；L50–L58 是一个整体。 |
| `C1:L51` | `"python": platform.python_version(),` | 键 `"python"` 对应 Python 版本字符串。 |
| `C1:L52` | `"deepchem": dc.__version__,` | 记录 DeepChem 版本。 |
| `C1:L53` | `"rdkit": rdkit.__version__,` | 记录 RDKit 版本。 |
| `C1:L54` | `"numpy": np.__version__,` | 记录 NumPy 版本。 |
| `C1:L55` | `"pandas": pd.__version__,` | 记录 pandas 版本。 |
| `C1:L56` | `"scipy": scipy.__version__,` | 记录 SciPy 版本。 |
| `C1:L57` | `"scikit-learn": sklearn.__version__,` | 记录 scikit-learn 版本。 |
| `C1:L58` | `}` | 关闭从 L50 开始的字典；没有独立计算含义。 |
| `C1:L60` | `print("Repository:", REPO_ROOT.name)` | 打印仓库路径最后一部分，例如 `ML-practice`。 |
| `C1:L61` | `print("Branch:", BRANCH)` | 打印最后执行本单元时记录的分支。 |
| `C1:L62` | `print("Source commit:", SOURCE_COMMIT)` | 打印最后执行本单元时记录的 commit。 |
| `C1:L63` | `print(json.dumps(VERSIONS, indent=2, ensure_ascii=False))` | 把版本字典格式化成两空格缩进的 JSON 后打印。 |

## 7. 需要成块理解的语法

### 7.1 `find_repo_root`

```python
def find_repo_root(start=Path.cwd().resolve()):
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(...)
```

缩进表示包含关系：

- `for` 属于函数；
- `if` 属于 `for`；
- `return` 属于 `if`；
- `raise` 属于函数，但不属于 `for` 的单次循环体结果。

函数会从当前目录向上查找，找到第一个带 `.git` 的目录。

### 7.2 `git_output`

```python
def git_output(*args):
    try:
        return subprocess.check_output(...).strip()
    except (...):
        return "unavailable"
```

例如：

```python
git_output("rev-parse", "HEAD")
```

`args` 是：

```python
("rev-parse", "HEAD")
```

`["git", *args]` 展开后是：

```python
["git", "rev-parse", "HEAD"]
```

函数定义时虽然还没有给 `REPO_ROOT` 赋值，但函数体在稍后调用时才查找全局变量，所以这里可以运行。

### 7.3 `GIT_DIRTY` 的回退缺陷

正常情况：

| `git status --porcelain` 的字符串 | `bool(...)` | 含义 |
|---|---:|---|
| `""` | `False` | 没检测到未提交改动 |
| 非空状态文本 | `True` | 检测到未提交改动 |

但 Git 失败时 `git_output` 返回：

```python
"unavailable"
```

而：

```python
bool("unavailable") is True
```

因此 `GIT_DIRTY=True` 可能有两种原因：

1. 工作区确实有未提交修改；
2. Git 命令读取失败。

它不影响模型数字，但影响元数据解释。

## 8. 已保存输出为什么不是当前分支

Notebook 保存的输出显示：

```text
Branch: baseline/esol-repro
Source commit: 5bd7b5c64b07be2091c827251aeb21e23c638863
```

审计时仓库实际位于：

```text
Branch: agent/data-algorithm-tracks
Commit: 3477d58...
```

原因是 `.ipynb` 不只保存源代码，还保存：

- 上次执行编号；
- 标准输出；
- warning；
- 表格和 Markdown 显示结果。

切换 Git 分支不会自动重新执行 Notebook。

所以旧输出表示：

> 上次运行并保存代码单元时的环境。

它不一定表示：

> 此刻打开 Notebook 时的环境。

重新运行后输出才会更新，但那也会形成一次新实验运行。

## 9. DeepChem 的可选依赖 warning

已保存输出包含：

```text
No module named 'torch'
No module named 'tensorflow'
No module named 'jax'
```

它们表示部分 DeepChem 神经网络功能不可用。

当前 Day 1 使用 scikit-learn 的传统模型，所以：

- 这些 warning 不会阻止当前数据加载和传统模型实验；
- warning 不等于当前单元已经失败；
- 以后进入 MLP/GNN 时，PyTorch 等依赖可能才变成必需；
- `RDLogger.DisableLog(...)` 只关闭 RDKit warning，不会关闭所有 DeepChem 导入提示。

## 10. 本单元易错点

- 导入模型类不等于已经训练模型。
- `SEED=42` 不能保证所有版本、平台永远逐位一致。
- `Path / "name"` 是拼路径，不是数学除法。
- 隐藏 RDKit warning 不等于数据没有问题。
- `get Git branch` 的 Notebook 输出可能已经过期。
- `GIT_DIRTY=True` 还可能来自 Git 读取失败。
- `.cache/deepchem` 是生成缓存，不是手写源代码。

---

# 代码单元 3：建立特征并加载 ESOL

## 11. 单元目的

本单元完成：

1. 定义如何把分子转为 1024 维 ECFP；
2. 加载 Delaney ESOL；
3. 使用 scaffold 划分；
4. 明确保留原始 logS；
5. 解包 train、valid、test。

## 12. 输入和输出

| 名称 | 类型 | 数量或形状 | 作用 |
|---|---|---:|---|
| `dc` | DeepChem 模块 | — | 提供特征和数据集工具 |
| `DATA_DIR` | `Path` | — | 数据与缓存目录 |
| `featurizer` | `CircularFingerprint` | 1024 维、半径 2 | 分子到数字的规则 |
| `tasks` | `list[str]` | 长度 1 | 目标名称 |
| `datasets` | 三元素序列 | 3 个 Dataset | train/valid/test |
| `transformers` | `list` | `[]` | 未使用标签转换器 |
| `train_dataset` | DeepChem Dataset | 902 行 | 训练数据 |
| `valid_dataset` | DeepChem Dataset | 113 行 | 验证数据 |
| `test_dataset` | DeepChem Dataset | 113 行 | 测试数据 |

## 13. 每个非空物理行

| 行号 | 原代码 | 结构、类型和作用 |
|---|---|---|
| `C3:L01` | `featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)` | 创建 ECFP 特征器对象；不是特征矩阵。每个分子输出 1024 维，邻域半径为 2。 |
| `C3:L03` | `tasks, datasets, transformers = dc.molnet.load_delaney(` | 开始调用 ESOL 加载函数，并准备把三个返回部分解包到三个变量。 |
| `C3:L04` | `featurizer=featurizer,` | 把上面创建的特征器作为关键字参数传入。 |
| `C3:L05` | `splitter="scaffold",` | 请求按分子骨架划分，而不是简单随机拆行。 |
| `C3:L06` | `transformers=[],` | 请求不转换标签，保留原始 logS；不代表没有 ECFP 特征化。 |
| `C3:L07` | `reload=True,` | 允许复用兼容缓存；布尔值 `True` 首字母大写。 |
| `C3:L08` | `data_dir=str(DATA_DIR),` | 把 `Path` 转成字符串，指定原始数据目录。 |
| `C3:L09` | `save_dir=str(DATA_DIR),` | 指定处理后缓存目录。 |
| `C3:L10` | `)` | 关闭从 L03 开始的 `load_delaney(...)` 调用。 |
| `C3:L11` | `train_dataset, valid_dataset, test_dataset = datasets` | 要求 `datasets` 正好有三个元素并按顺序解包。 |
| `C3:L13` | `assert transformers == [], "Day 1 requires labels in the original logS space."` | 要求返回的转换器列表为空；否则抛出带消息的 `AssertionError`。 |
| `C3:L14` | `print("Task:", tasks)` | 打印任务列表，当前只有一个回归目标。 |
| `C3:L15` | `print("Split sizes:", len(train_dataset), len(valid_dataset), len(test_dataset))` | 打印三个 Dataset 的样本数，不是特征数。 |
| `C3:L16` | `print("Transformers:", transformers)` | 打印空列表，确认标签未标准化。 |

## 14. `CircularFingerprint` 到底是什么

原始分子常以 SMILES 表示，例如一串字符。传统 sklearn 模型不能直接理解化学字符串，所以需要转成数字。

```text
一个分子
   ↓ CircularFingerprint
[0, 1, 0, 0, 1, ..., 0]
   共 1024 个位置
```

`size=1024` 表示输出列数。

`radius=2` 表示考虑原子周围最多两个化学键距离的局部环境，不表示“只有两个特征”。

这些特征通常不是：

- 第 1 列分子量；
- 第 2 列黏度；
- 第 3 列密度。

它们是局部结构片段经过哈希映射形成的位置，可能存在哈希碰撞。

## 15. 为什么使用 scaffold split

随机划分可能把高度相似的分子分到训练集和验证集，使验证结果过于乐观。

scaffold 划分尽量按核心骨架分开：

```text
某些骨架 → train
另一些骨架 → valid
另外骨架 → test
```

因此验证集更接近“模型遇到新骨架”的情况，通常比随机划分难。

但它仍然只是一个数据集划分协议，不能代表所有真实场景。

## 16. `transformers=[]` 的准确含义

这里的空列表表示不使用 DeepChem 的标签 transformer。

所以：

- `y` 保持原始 logS；
- 后面的 MAE/RMSE 也在 logS 空间；
- 分子仍然经过 ECFP 特征化；
- “标签不转换”和“输入不处理”是两回事。

## 17. 已保存输出解释

```text
Task: ['measured log solubility in mols per litre']
Split sizes: 902 113 113
Transformers: []
```

逐项含义：

| 输出 | 正确解释 |
|---|---|
| 方括号中的任务名称 | `tasks` 是列表，且只有一个目标 |
| `902` | 训练样本数 |
| 第一个 `113` | 验证样本数 |
| 第二个 `113` | 测试样本数 |
| `[]` | 没有标签转换器 |

## 18. 本单元易错点

- `featurizer` 是转换规则，不是 `X` 本身。
- 1024 是每个样本的特征数，不是样本数。
- `radius=2` 不是两个特征。
- `reload=True` 不是每次强制重新下载。
- `transformers=[]` 不表示完全没有数据处理。
- 检查测试集大小不等于用测试标签训练模型。

---

# 代码单元 4：数据完整性检查

## 19. 单元目的

本单元检查：

- 行数是否符合预期；
- 每个 `X` 是否有 1024 列；
- `X` 和 `y` 是否全为有限数字；
- 三个 split 的精确 ID/SMILES 是否重叠；
- 每个 split 的 logS 范围。

## 20. 输入和输出

| 输入对象 | 类型 | 形状 |
|---|---|---:|
| `train_dataset.X` | NumPy-like 二维数组 | `(902, 1024)` |
| `train_dataset.y` | NumPy-like 二维数组 | `(902, 1)` |
| `valid_dataset.X` | NumPy-like 二维数组 | `(113, 1024)` |
| `valid_dataset.y` | NumPy-like 二维数组 | `(113, 1)` |
| `test_dataset.X` | NumPy-like 二维数组 | `(113, 1024)` |
| `test_dataset.y` | NumPy-like 二维数组 | `(113, 1)` |

| 新对象 | 类型 | 内容 |
|---|---|---|
| `split_datasets` | `dict[str, Dataset]` | split 名称到 Dataset |
| `expected_sizes` | `dict[str, int]` | 预期行数 |
| `id_sets` | `dict[str, set]` | 各 split 的 ID 集合 |
| `overlaps` | `dict[str, int]` | 三组精确交集数量 |

## 21. 每个非空物理行

| 行号 | 原代码 | 结构、类型和作用 |
|---|---|---|
| `C4:L01` | `split_datasets = {` | 开始创建 split 名称到 Dataset 的字典。 |
| `C4:L02` | `"train": train_dataset,` | 键 `"train"` 对应训练 Dataset。 |
| `C4:L03` | `"valid": valid_dataset,` | 键 `"valid"` 对应验证 Dataset。 |
| `C4:L04` | `"test": test_dataset,` | 键 `"test"` 对应测试 Dataset。 |
| `C4:L05` | `}` | 关闭从 L01 开始的字典。 |
| `C4:L06` | `expected_sizes = {"train": 902, "valid": 113, "test": 113}` | 建立预期样本数字典，用于检测数据或划分意外变化。 |
| `C4:L08` | `for split_name, dataset in split_datasets.items():` | 循环解包每个 `(名称, Dataset)`；冒号后进入缩进循环体。 |
| `C4:L09` | `assert len(dataset) == expected_sizes[split_name]` | 检查实际样本数等于对应预期数。 |
| `C4:L10` | `assert dataset.X.shape == (expected_sizes[split_name], 1024)` | 检查二维形状为 `(样本数, 1024 特征)`。 |
| `C4:L11` | `assert np.isfinite(dataset.X).all()` | 对每个输入数检查非 NaN、非正负无穷，并要求全部通过。 |
| `C4:L12` | `assert np.isfinite(dataset.y).all()` | 对所有标签执行相同的有限值检查。 |
| `C4:L13` | `print(` | 开始跨四行的打印调用。 |
| `C4:L14` | `f"{split_name:>5}: X={dataset.X.shape}, y={dataset.y.shape}, "` | f-string 插入名称和形状；`:>5` 表示宽度 5 右对齐。 |
| `C4:L15` | `f"logS=[{dataset.y.min():.3f}, {dataset.y.max():.3f}]"` | 与上一行字符串自动连接；打印 min/max，`.3f` 保留三位小数。 |
| `C4:L16` | `)` | 关闭 L13 的 `print(...)`；本身没有独立数据操作。 |
| `C4:L18` | `id_sets = {name: set(dataset.ids) for name, dataset in split_datasets.items()}` | 字典推导式；把每个 split 的 ID 序列转成集合。 |
| `C4:L19` | `overlaps = {` | 开始创建三种两两交集计数字典。 |
| `C4:L20` | `"train_valid": len(id_sets["train"] & id_sets["valid"]),` | `&` 求集合交集，`len` 统计 train/valid 精确 ID 重叠数。 |
| `C4:L21` | `"train_test": len(id_sets["train"] & id_sets["test"]),` | 统计 train/test 精确 ID 重叠数。 |
| `C4:L22` | `"valid_test": len(id_sets["valid"] & id_sets["test"]),` | 统计 valid/test 精确 ID 重叠数。 |
| `C4:L23` | `}` | 关闭从 L19 开始的字典。 |
| `C4:L24` | `assert sum(overlaps.values()) == 0` | 三个重叠数求和，并要求总数为 0。 |
| `C4:L25` | `print("SMILES overlap counts:", overlaps)` | 打印精确 ID/SMILES 交集字典。 |

## 22. 循环中的形状检查

二维数组形状写成：

```text
(行数, 列数)
```

所以：

```text
(902, 1024)
```

表示：

- 902 个样本；
- 每个样本 1024 个 ECFP 特征。

不是 902 个特征和 1024 个样本。

`np.isfinite(dataset.X)` 返回一个同形状布尔数组：

```text
有限数字 → True
NaN/inf/-inf → False
```

随后 `.all()` 要求全部位置为 `True`。

这只是检查，不会自动填补或删除坏数据。

## 23. 已保存形状和范围输出

```text
train: X=(902, 1024), y=(902, 1), logS=[-11.600, 1.580]
valid: X=(113, 1024), y=(113, 1), logS=[-9.332, 1.100]
 test: X=(113, 1024), y=(113, 1), logS=[-8.804, 1.070]
```

解释：

| split | `X` | `y` | logS 范围 |
|---|---:|---:|---:|
| train | `(902,1024)` | `(902,1)` | `-11.600` 到 `1.580` |
| valid | `(113,1024)` | `(113,1)` | `-9.332` 到 `1.100` |
| test | `(113,1024)` | `(113,1)` | `-8.804` 到 `1.070` |

不同 split 的范围不完全相同，不自动意味着代码错误。

scaffold 划分会把不同骨架分开，因此数据分布可能发生变化，这也会使泛化更困难。

## 24. 字典推导式展开

这一行：

```python
id_sets = {name: set(dataset.ids) for name, dataset in split_datasets.items()}
```

大致等价于：

```python
id_sets = {}
for name, dataset in split_datasets.items():
    id_sets[name] = set(dataset.ids)
```

`set` 的特点：

- 元素不重复；
- 没有依赖原始顺序；
- 可以用 `&` 快速计算交集。

## 25. 精确 SMILES 无重叠的能力边界

已保存输出是：

```text
SMILES overlap counts: {'train_valid': 0, 'train_test': 0, 'valid_test': 0}
```

它只证明：

> `dataset.ids` 中没有跨 split 完全相同的字符串。

它不能单独证明：

- 不存在同一分子的不同 SMILES 写法；
- 不存在互变异构体、盐形式或标准化差异；
- 不存在高度相似结构；
- 不存在相同或接近的化学骨架；
- 不存在任何形式的数据泄漏。

核心骨架隔离主要依赖：

```python
splitter="scaffold"
```

当前集合检查只是额外的“精确 ID 重复检查”，不是完整的化学泄漏审计。

## 26. 本单元易错点

- `.shape` 先写样本行数，再写特征列数。
- `np.isfinite` 只检查，不修复数据。
- 负 logS 不表示负浓度。
- split 目标范围不同不一定是错误。
- 精确 SMILES 交集为 0 不等于没有相似分子。
- 查看测试集形状和有限值不等于用测试集训练。
- `assert` 是保护性检查，不是模型算法。

---

# 代码单元 6：建立 X 和 y

## 27. 单元目的

DeepChem Dataset 把输入和答案放在：

```text
dataset.X
dataset.y
```

本单元把 train 和 valid 转成 sklearn 常用 NumPy 形状。

## 28. 输入和输出

| 输入 | 形状 | 输出 | 输出形状 |
|---|---:|---|---:|
| `train_dataset.X` | `(902,1024)` | `X_train` | `(902,1024)` |
| `train_dataset.y` | `(902,1)` | `y_train` | `(902,)` |
| `valid_dataset.X` | `(113,1024)` | `X_valid` | `(113,1024)` |
| `valid_dataset.y` | `(113,1)` | `y_valid` | `(113,)` |

四个输出类型都是 `numpy.ndarray`。

## 29. 每个非空物理行

| 行号 | 原代码 | 结构、类型和作用 |
|---|---|---|
| `C6:L01` | `X_train = np.asarray(train_dataset.X)` | 确保训练特征是 NumPy 数组；形状 `(902,1024)`。 |
| `C6:L02` | `y_train = np.asarray(train_dataset.y).reshape(-1)` | 先转数组，再把单目标标签从 `(902,1)` 展平为 `(902,)`。 |
| `C6:L03` | `X_valid = np.asarray(valid_dataset.X)` | 确保验证特征是 NumPy 数组；形状 `(113,1024)`。 |
| `C6:L04` | `y_valid = np.asarray(valid_dataset.y).reshape(-1)` | 把验证标签从 `(113,1)` 展平为 `(113,)`。 |
| `C6:L06` | `print("Train arrays:", X_train.shape, y_train.shape)` | 打印训练输入和标签形状。 |
| `C6:L07` | `print("Validation arrays:", X_valid.shape, y_valid.shape)` | 打印验证输入和标签形状。 |
| `C6:L08` | `print("Test predictions are intentionally not computed on Day 1.")` | 打印实验协议提醒；它不会从技术上禁止后续调用测试集。 |

## 30. `np.asarray` 与 `.reshape(-1)`

`np.asarray(...)` 保证得到 NumPy 数组形式。

如果输入已经是合适的 NumPy 数组，它可能不复制底层数据。

`.reshape(-1)` 中的 `-1` 不表示负数，而表示：

> 根据元素总数自动推断这一维长度。

```text
(902, 1) 共有 902 个元素
reshape(-1)
(902,)    仍然有 902 个元素
```

所以它没有删除标签。

单目标 sklearn 回归通常希望：

```text
X.shape == (n_samples, n_features)
y.shape == (n_samples,)
```

如果未来是多目标任务，不能不加思考地把所有目标 `.reshape(-1)`，否则可能混合任务。

## 31. 已保存输出解释

```text
Train arrays: (902, 1024) (902,)
Validation arrays: (113, 1024) (113,)
Test predictions are intentionally not computed on Day 1.
```

它说明：

```text
X_train 行数 == y_train 长度 == 902
X_valid 行数 == y_valid 长度 == 113
```

这是监督学习最基本的配对关系：

```text
一行 X ↔ 一个 y
```

## 32. 为什么此时不计算测试预测

| 数据 | 用途 |
|---|---|
| train | 模型真正学习参数 |
| validation | 比较固定候选模型 |
| test | 最后的独立评估 |

如果不断看 test 结果并据此修改模型，test 会逐渐变成另一份 validation。

本项目还说明旧 Notebook 已经暴露过当前测试结果，所以它已经不再是严格完全未见的 holdout。

打印提示语本身不会阻止测试泄漏；真正的控制来自后续代码和实验纪律。

## 33. 本单元易错点

- `X` 是约定俗成的输入变量名，不是乘号。
- `y_train` 是真实答案，不是预测结果。
- `.reshape(-1)` 不会让数值变负。
- `(902,)` 仍有 902 个标签，没有丢一列数据。
- validation 标签不应传给 `.fit`。
- 同一模型要求 train 和 valid 的特征列数一致。
- 一句 `print` 不能自动禁止使用 test。

---

# 34. 把四个单元串起来

```text
代码单元 1
建立可追溯环境：路径、Git、版本、随机种子
        ↓
代码单元 3
SMILES → ECFP，并得到 train/valid/test
        ↓
代码单元 4
检查行数、列数、有限值和精确 ID 重叠
        ↓
代码单元 6
生成 sklearn 使用的 X_train/y_train/X_valid/y_valid
```

到这里仍然没有训练模型。

真正训练至少要出现：

```python
model.fit(X_train, y_train)
```

本部分的价值是先确保：

- 模型的输入是什么；
- 正确答案是什么；
- 数据形状是否匹配；
- 划分协议是否明确；
- 实验环境是否能追溯。

---

# 35. 学习检查题

1. `import deepchem as dc` 中 `as dc` 的作用是什么？
2. 为什么 `Path / "data"` 不是数学除法？
3. `SEED=42` 能否保证所有机器和版本永远完全一致？
4. 为什么要记录 commit 和依赖版本？
5. 为什么 `.ipynb` 中显示的分支可能不是当前分支？
6. `GIT_DIRTY=True` 除了本地修改，还可能是什么？
7. 缺少 PyTorch 为什么不阻止当前传统模型实验？
8. ECFP 的 `size=1024` 和 `radius=2` 分别表示什么？
9. `transformers=[]` 是否表示输入没有特征化？
10. `(902,1024)` 中哪个是样本数？
11. `np.isfinite(...).all()` 是检查还是修复？
12. 精确 SMILES 交集为 0 能否证明没有相似分子？
13. scaffold split 为什么通常比随机 split 更严格？
14. `.reshape(-1)` 对 `(902,1)` 做了什么？
15. `logS=-3` 是否代表负浓度？
16. logS 相差 1 对应原始浓度大约相差多少倍？
17. 为什么不应反复查看 test 并修改模型？
18. 到代码单元 6 为止，模型是否已经训练？

能够用自己的话回答这些问题，再进入模型定义、`.fit`、`.predict` 和评价指标会更稳。
