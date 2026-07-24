# Day 1 所需的 Python 语法预备

这不是完整 Python 教程，只解释 Day 1 Notebook 中实际出现的语法。阅读时不要求背诵，遇到原代码时能够回来查找即可。

## 1. Jupyter Notebook 是什么

Notebook 由两类单元组成：

- **Markdown 单元**：标题、说明和结论，不作为 Python 执行；
- **Code 单元**：真正的 Python 代码，按执行顺序共享变量。

如果先运行后面的单元，却没有运行前面的定义，可能出现 `NameError`。因此第一次学习时应选择“Restart Kernel and Run All”，让执行号从 1 连续增加。

只有最后一行写变量名时，Jupyter 会自动展示它：

```python
results
```

普通 `.py` 脚本不会自动显示，脚本通常需要写 `print(results)`。

## 2. 缩进决定代码属于谁

Python 不用大括号标记代码块，而使用缩进：

```python
def add_one(x):
    result = x + 1
    return result
```

- `def ...:` 后面的冒号表示代码块开始；
- 四个空格缩进的两行属于函数；
- 回到最左侧表示函数结束。

Day 1 中的 `for`、`if`、`try`、`except` 和函数定义都遵守这个规则。缩进错一层可能直接报错，也可能让代码在错误的循环层级运行。

## 3. 变量和赋值号 `=`

```python
SEED = 42
```

结构是：

```text
变量名 = 右侧表达式的结果
```

`=` 表示“把右侧结果绑定到左侧名字”，不是数学中的相等判断。判断是否相等使用 `==`：

```python
SEED == 42
```

常量通常使用全大写名字，例如 `SEED`；这是命名习惯，不是 Python 强制规定。

## 4. 常见数据类型

| 类型 | 示例 | 本 Notebook 中的用途 |
|---|---|---|
| `int` 整数 | `42`、`1024` | 随机种子、样本数、特征维数 |
| `float` 小数 | `1.0`、`0.2578` | 模型参数、时间、评价指标 |
| `str` 字符串 | `"train"` | 名称、路径片段、说明文字 |
| `bool` 布尔值 | `True`、`False` | 开关或逻辑结果 |
| `None` 空值 | `None` | 表示没有指定某项设置 |
| `list` 列表 | `["train", "valid"]` | 有顺序、可修改的一组值 |
| `tuple` 元组 | `(902, 1024)` | 固定组合、数组形状、多变量循环 |
| `dict` 字典 | `{"train": 902}` | 键到值的映射 |
| `set` 集合 | `{"train", "valid"}` | 去重和集合运算 |

检查类型可以临时运行：

```python
type(SEED)
type(X_train)
```

## 5. 导入 `import`

```python
import numpy as np
from pathlib import Path
```

- `import numpy as np`：导入整个 `numpy` 模块，并给它简称 `np`；
- `from pathlib import Path`：只从 `pathlib` 模块取出 `Path` 这个名字。

因此后面分别写：

```python
np.asarray(...)
Path.cwd()
```

`as` 只是设置简称，不会改变库本身。

## 6. 函数调用和关键字参数

```python
Ridge(alpha=1.0)
```

基本形式是：

```text
可调用对象(参数)
```

`alpha=1.0` 是关键字参数，明确把 `1.0` 传给名为 `alpha` 的设置。对比：

```python
print("Branch:", BRANCH)
```

这里两个值按位置传入，叫位置参数。

函数可以跨多行书写，只要外层圆括号还没有闭合：

```python
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
)
```

最后一个参数后的逗号允许保留，便于以后增加或调整参数。

## 7. 属性和方法中的点号 `.`

```python
dataset.X
dataset.X.shape
model.fit(X_train, y_train)
```

- `对象.属性`：读取对象保存的信息，例如 `dataset.X`；
- `对象.方法(...)`：让对象执行动作，例如 `model.fit(...)`；
- 可以连续访问，例如 `dataset.X.shape` 是先取得 `X`，再取得它的形状。

是否带圆括号很重要：

```python
dataset.y.min    # 方法本身，还没执行
dataset.y.min()  # 执行方法，得到最小值
```

## 8. 列表、元组、字典和集合

### 列表

```python
values = [1, 2, 3]
```

用方括号创建，可以用位置索引：`values[0]` 得到第一个值。Python 从 0 开始计数。

### 元组

```python
shape = (902, 1024)
```

这里表示 902 行、1024 列。元组通常用来表达不会随意改变的组合。

### 字典

```python
expected_sizes = {"train": 902, "valid": 113}
```

结构是 `键: 值`。读取训练集期望大小：

```python
expected_sizes["train"]
```

### 集合

```python
{"train", "valid"}
```

集合没有重复项。`&` 表示交集，Day 1 用它检查不同数据集是否出现同一个 SMILES。

## 9. 索引、负数索引与 pandas 的 `.loc`、`.iloc`

```python
tasks[0]
```

方括号在这里表示取第 0 个元素。

```python
.reshape(-1)
```

这里的 `-1` 不是“倒数第一个”，因为它出现在 `reshape` 参数里；它表示让 NumPy 自动计算这一维长度，将标签整理成一维数组。

pandas 中：

```python
table.loc[条件, "列名"]
table.iloc[0]
```

- `.loc` 主要按标签或布尔条件选择；
- `.iloc` 按整数位置选择；
- `.iloc[0]` 表示选中当前结果的第一行。

## 10. 函数定义、参数与返回值

```python
def regression_metrics(y_true, y_pred):
    return {"mae": some_metric(y_true, y_pred)}
```

- `def`：开始定义函数；
- `regression_metrics`：函数名；
- `y_true, y_pred`：调用时需要传入的参数；
- `return`：把结果交还给调用者；
- 函数在定义时不会立即计算，只有调用时才执行。

调用：

```python
scores = regression_metrics(y_valid, prediction)
```

## 11. `for` 循环和拆包

```python
for model_name, model in models.items():
    model.fit(X_train, y_train)
```

`models.items()` 每次给出一对 `(键, 值)`；左侧两个变量把这一对拆开。循环会对字典中的每个模型执行缩进代码。

类似地：

```python
train_dataset, valid_dataset, test_dataset = datasets
```

这叫序列拆包：右边必须能提供恰好三个值。

## 12. 条件、异常和 `assert`

```python
if condition:
    return value
```

只有 `condition` 为真时才执行缩进内容。

```python
try:
    risky_action()
except SomeError:
    fallback()
```

`try` 尝试执行，指定异常出现时进入 `except`，避免整个流程直接中断。

```python
assert len(dataset) == 902
```

`assert` 是自动检查：条件为真时继续，条件为假时抛出 `AssertionError`。它不是训练模型，而是防止错误数据悄悄进入后续步骤。

## 13. f-string

```python
RUN_ID = f"seed{SEED}"
```

字符串前面的 `f` 允许把 `{SEED}` 替换成变量值。

格式控制：

```python
f"{score:.3f}"
```

`.3f` 表示用固定小数形式显示 3 位小数，只影响显示，不改变变量内部精度。

```python
f"{name:>5}"
```

`>5` 表示在至少 5 个字符宽度中右对齐。

## 14. `*` 和 `**` 的不同含义

```python
def git_output(*args):
```

这里 `*args` 把任意数量的位置参数收集成一个元组。

```python
["git", *args]
```

这里 `*args` 把元组内容展开进列表。

```python
{"model": name, **scores}
```

这里 `**scores` 把一个字典的键值对展开并合并到新字典。

三个位置的符号相同，但作用由上下文决定。

## 15. 字典推导式

```python
id_sets = {name: set(dataset.ids) for name, dataset in split_datasets.items()}
```

它相当于：

```python
id_sets = {}
for name, dataset in split_datasets.items():
    id_sets[name] = set(dataset.ids)
```

推导式更短，但初学时先读懂展开版本即可。

## 16. 路径对象和 `/`

```python
DATA_DIR = REPO_ROOT / ".cache" / "deepchem"
```

这里 `/` 不是除法。`REPO_ROOT` 是 `Path` 对象，`Path` 重新定义了 `/`，用于安全拼接路径。

```python
DATA_DIR.mkdir(parents=True, exist_ok=True)
```

- `parents=True`：需要时连上层目录一起创建；
- `exist_ok=True`：目录已存在也不报错。

## 17. NumPy 数组和 `shape`

`X_train.shape == (902, 1024)` 表示：

- 902 行：902 个训练分子；
- 1024 列：每个分子的 1024 位 ECFP 特征。

`y_train.shape == (902,)` 表示 902 个标签的一维数组。逗号不能省略；`(902)` 只是整数 902，`(902,)` 才是一维形状元组。

## 18. 模型对象的统一接口

scikit-learn 模型通常遵循：

```python
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
parameters = model.get_params()
```

- `fit` 改变模型内部状态，让它从训练数据学习；
- `predict` 不再学习，只使用已经学到的状态产生预测；
- `get_params` 读取模型配置，不是读取学到的全部模型权重。

## 19. pandas 表格的常见动作

```python
results = pd.DataFrame.from_records(records)
```

把“由多个字典组成的列表”转换成表格，每个字典通常成为一行。

```python
results.loc[results["split"] == "valid"]
```

先产生一列 `True/False`，再仅保留为 `True` 的验证集行。

```python
results.pivot(index="model", columns="split", values=[...])
```

把长表整理成便于比较的宽表；它改变展示结构，不会重新训练模型。

## 20. JSON 与 Python 字典

Python 字典存在内存里；JSON 是可以写入文件的文本格式：

```python
text = json.dumps(dictionary, indent=2, ensure_ascii=False)
```

- `dumps` 中的 `s` 可以理解为生成 string；
- `indent=2` 让文本缩进易读；
- `ensure_ascii=False` 允许中文原样保存；
- `default=str` 在遇到 JSON 不认识的对象时尝试转成字符串。

## 21. 阅读一行复杂代码的方法

例如：

```python
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
```

从最里面向外读：

1. `mean_squared_error(y_true, y_pred)`：计算均方误差；
2. `np.sqrt(...)`：开平方，得到 RMSE；
3. `float(...)`：转换成普通 Python 小数；
4. `rmse = ...`：把结果保存为 `rmse`。

遇到复杂代码时，不要试图一次读完整行，先找到最里面的圆括号。
