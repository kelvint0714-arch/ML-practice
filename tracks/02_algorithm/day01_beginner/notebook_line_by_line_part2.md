# Notebook 逐行讲解（下）：指标、模型、训练、排名、保存与验收

本页覆盖原 Notebook 的代码 Cell 7、8、9、10、11、12。建议先完成[上半部分](notebook_line_by_line_part1.md)，至少确认你知道 `X_train`、`y_train`、`X_valid` 和 `y_valid` 分别是什么。

## 下半部分总数据流

```text
定义评价函数
    ↓
创建五个“尚未训练”的模型
    ↓
逐个执行 fit(X_train, y_train)
    ↓
分别 predict(X_train) 和 predict(X_valid)
    ↓
计算 MAE、RMSE、R²，追加到 records
    ↓
records → results DataFrame → 验证集排名
    ↓
保存 CSV/JSON → assert 自动检查 → 显示结论
```

---

## 代码 Cell 7：定义统一的回归指标函数

### 原代码

```python
def regression_metrics(y_true, y_pred):
    return {
        "mae_logS": float(mean_absolute_error(y_true, y_pred)),
        "rmse_logS": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
```

### 这一格的输入和输出

| 项目 | 类型/形状 | 含义 |
|---|---|---|
| 输入 `y_true` | 一维数组 `(n_samples,)` | 已知的真实 logS |
| 输入 `y_pred` | 一维数组 `(n_samples,)` | 模型预测的 logS |
| 输出 | `dict[str, float]`，3 项 | MAE、RMSE、R² |

执行这一格只是**定义函数**，还没有计算任何模型的指标。

### 逐行解释

#### C7:L01 `[核心]`

```python
def regression_metrics(y_true, y_pred):
```

- `def` 表示开始定义函数；
- `regression_metrics` 是作者起的函数名，意思是“回归指标”；
- 圆括号内是两个形式参数；
- `y_true` 表示真实答案，`y_pred` 表示预测答案；
- 结尾冒号 `:` 表示下面缩进的代码属于函数；
- 这行运行后，Python 记住函数规则，但不会立刻计算。

通用结构：

```python
def 函数名(参数1, 参数2):
    函数内部代码
```

#### C7:L02 `[核心]`

```python
    return {
```

- 四个空格说明它在函数内部；
- `return` 把结果交回调用函数的位置，并结束本次函数调用；
- `{` 开始创建字典；
- 后续三组“指标名称 → 指标数值”会装进同一个字典。

#### C7:L03 `[核心]`

```python
        "mae_logS": float(mean_absolute_error(y_true, y_pred)),
```

从最里面向外读：

1. `mean_absolute_error(y_true, y_pred)` 比较真实值与预测值并计算 MAE；
2. `float(...)` 把结果转成普通 Python 小数，方便写入 JSON；
3. 字符串 `"mae_logS"` 是字典键；
4. 冒号分隔字典的键和值；
5. 行末逗号表示字典后面还有项目。

MAE 是每个样本绝对误差的平均值，越低越好。这里处在 logS 数值尺度，不是百分比。

#### C7:L04 `[核心]`

```python
        "rmse_logS": float(np.sqrt(mean_squared_error(y_true, y_pred))),
```

从内向外分四层：

1. `mean_squared_error(...)`：误差平方后取平均；
2. `np.sqrt(...)`：开平方，得到 RMSE；
3. `float(...)`：转成 Python `float`；
4. 保存到字典键 `"rmse_logS"`。

平方使大错误被惩罚得更重，所以 RMSE 对极端错误比 MAE 更敏感。越低越好。

#### C7:L05 `[核心]`

```python
        "r2": float(r2_score(y_true, y_pred)),
```

- `r2_score(...)` 计算决定系数 R²；
- R² 越高越好，理想值是 1；
- R² 可以小于 0，表示在这批数据上甚至不如相应的均值参考；
- 负 R² 不是“负准确率”，也不表示 Python 出错。

#### C7:L06 `[核心]`

```python
    }
```

- `}` 关闭字典；
- 因为字典紧跟在 `return` 后面，所以它就是函数返回值；
- 函数每次调用会新建一个三项字典。

### 一个极小例子

如果真实值是 `[1, 3]`，预测值是 `[2, 2]`，两个误差都是 1，MAE 为 1，RMSE 也为 1。真实实验有更多样本，函数会对整列数组统一计算。

---

## 代码 Cell 8：创建五个候选模型

### 原代码

```python
models = {
    "dummy_mean": DummyRegressor(strategy="mean"),
    "ridge": Ridge(alpha=1.0),
    "decision_tree_overfit_diagnostic": DecisionTreeRegressor(
        random_state=SEED
    ),
    "decision_tree_regularized": DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=5, random_state=SEED
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    ),
}

print("Fixed candidates:", list(models))
```

### 最重要的区别

```text
创建模型对象 ≠ 已经训练模型
```

这一格只指定“要使用什么算法和参数”。直到下一格调用 `.fit(...)`，模型才从数据学习。

### 逐行解释

#### C8:L01 `[核心]`

```python
models = {
```

- 创建名为 `models` 的变量；
- `{` 开始字典；
- 字典的键是方便阅读的模型名称字符串；
- 字典的值是模型对象。

#### C8:L02 `[核心]`

```python
    "dummy_mean": DummyRegressor(strategy="mean"),
```

- `DummyRegressor(...)` 创建最简单的回归基线；
- `strategy="mean"` 要求它在 `.fit()` 时记住训练标签平均值；
- 预测时它基本忽略 `X`，对每行都给相同的训练均值；
- 它的作用不是追求最好，而是检查复杂模型是否真的比“猜平均值”强。

#### C8:L03 `[核心]`

```python
    "ridge": Ridge(alpha=1.0),
```

- 创建 Ridge 线性回归模型；
- `alpha=1.0` 是 L2 正则化强度，不是学习率、准确率或显著性水平；
- 当前只测试这一项固定设置；结果差不能自动证明“Ridge 算法永远不适合”。

#### C8:L04–L06 `[核心]`

```python
    "decision_tree_overfit_diagnostic": DecisionTreeRegressor(
        random_state=SEED
    ),
```

- L04 开始创建决策树，圆括号未关闭，因此参数放到下一行；
- L05 把随机状态设为 42，便于重复运行；
- 这里没有设置 `max_depth`，树可以继续分裂，特意用于观察过拟合；
- L06 第一个 `)` 关闭模型构造，后面的逗号分隔字典项目。

#### C8:L07–L09 `[核心]`

```python
    "decision_tree_regularized": DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=5, random_state=SEED
    ),
```

- L07 开始第二棵决策树；
- `max_depth=5` 限制树最多向下五层；
- `min_samples_leaf=5` 要求每个叶节点至少包含五个训练样本；
- 这些限制降低模型复杂度，尝试减少记忆训练集；
- `regularized` 在名称中表示受限制，并不表示它一定表现更好。

#### C8:L10 `[核心]`

```python
    "random_forest": RandomForestRegressor(
```

- 创建随机森林回归器；
- 随机森林不是一棵巨大的树，而是多棵带随机性的树共同预测并取平均。

#### C8:L11 `[核心]`

```python
        n_estimators=300,
```

- `n_estimators` 是森林中的树数量；
- 这里为 300 棵；
- 它不表示树深为 300，也不表示训练 300 轮神经网络。

#### C8:L12 `[核心]`

```python
        min_samples_leaf=2,
```

每个叶节点至少两个训练样本，稍微抑制单个样本形成独立叶子的情况。

#### C8:L13 `[工程]`

```python
        max_features="sqrt",
```

- 每次节点分裂只考虑总特征数平方根数量的候选特征；
- 1024 的平方根约为 32；
- 不同树看到不同特征子集，增加树之间差异，有助于集成平均。

#### C8:L14 `[核心]`

```python
        random_state=SEED,
```

固定随机森林的抽样和特征随机过程。固定种子提高可重复性，但不同软件版本或平台仍可能有差别。

#### C8:L15 `[工程]`

```python
        n_jobs=-1,
```

- 这是 scikit-learn 的约定；
- `-1` 在这里表示使用可用的全部 CPU 核心；
- 它不是“负一个任务”。

#### C8:L16–L17 `[工程]`

```python
    ),
}
```

- L16 关闭 `RandomForestRegressor(...)` 并结束该字典项目；
- L17 关闭整个 `models` 字典；
- 此时字典有五项，五个模型都尚未 `.fit()`。

#### C8:L19 `[核心]`

```python
print("Fixed candidates:", list(models))
```

- 遍历字典默认得到键；
- `list(models)` 把五个键名变成列表；
- `print` 显示固定候选名称；
- “固定”表示先决定比较对象，再看验证结果，减少看结果后随意增加模型的偏差。

### 运行后关键变量

| 变量 | 类型 | 长度 | 状态 |
|---|---|---:|---|
| `models` | `dict` | 5 | 全部未训练 |

---

## 代码 Cell 9：循环训练、预测并记录结果

这是 Day 1 最重要也最复杂的一格。先看结构，不要先陷入 JSON 参数。

### 原代码

```python
records = []
fitted_models = {}

for model_name, model in models.items():
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    fitted_models[model_name] = model
    params = json.dumps(model.get_params(deep=True), sort_keys=True, default=str)

    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
    ):
        prediction = model.predict(X_split)
        records.append({
            "run_id": RUN_ID,
            "model": model_name,
            "split": split_name,
            "n_samples": len(y_split),
            "seed": SEED,
            "featurizer": "ECFP radius=2 size=1024",
            "splitter": "scaffold",
            "target_transformer": "none",
            "fit_seconds": fit_seconds,
            "params": params,
            "source_commit": SOURCE_COMMIT,
            **regression_metrics(y_split, prediction),
        })

results = pd.DataFrame.from_records(records)
assert len(results) == len(models) * 2
results
```

### 两层循环在做什么

```text
外层循环：5个模型
  ├─ 训练当前模型一次
  └─ 内层循环：2个数据集
       ├─ train预测并记一行
       └─ valid预测并记一行

总记录数 = 5 × 2 = 10行
```

### 逐行解释

#### C9:L01 `[核心]`

```python
records = []
```

创建空列表。后面每得到一组“模型＋数据集”的指标，就用 `.append(...)` 增加一个字典。

#### C9:L02 `[工程]`

```python
fitted_models = {}
```

创建空字典，用于保存训练后的模型对象。这样循环结束后仍可按名称取得模型。

#### C9:L04 `[核心]`

```python
for model_name, model in models.items():
```

- `models.items()` 每次给出 `(键, 值)`；
- `model_name` 接收名称字符串；
- `model` 接收对应模型对象；
- 循环执行五次；
- 下面四个空格缩进的代码属于外层循环。

#### C9:L05 `[工程]`

```python
    started = time.perf_counter()
```

读取适合计算耗时的高精度计时器，把起始时间保存到 `started`。这个数本身不是当前日期时间。

#### C9:L06 `[核心]`

```python
    model.fit(X_train, y_train)
```

这是实际训练发生的地方：

- 输入 `X_train`：`(902, 1024)`；
- 输入 `y_train`：`(902,)`；
- 模型只看到训练数据；
- `.fit()` 会修改模型内部状态，学到系数、分裂规则或训练均值；
- 它通常返回模型自身，但这里不接收返回值；
- 它不会把 `X_train` 变成预测结果。

#### C9:L07 `[工程]`

```python
    fit_seconds = time.perf_counter() - started
```

训练结束时间减去开始时间，得到训练耗时秒数。不同电脑和后台负载会改变该值，所以耗时不是固定科学常数。

#### C9:L08 `[工程]`

```python
    fitted_models[model_name] = model
```

- 方括号表示按键写入字典；
- 把已经训练的模型保存到对应名称下；
- 例如 `fitted_models["random_forest"]` 会得到训练后的随机森林。

#### C9:L09 `[高级]`

```python
    params = json.dumps(model.get_params(deep=True), sort_keys=True, default=str)
```

从内向外：

1. `model.get_params(deep=True)` 取得模型配置/超参数字典；
2. 它不是模型学到的全部树节点或 Ridge 系数；
3. `json.dumps(...)` 把字典变成一段 JSON 字符串；
4. `sort_keys=True` 让键排序，便于比较；
5. `default=str` 遇到 JSON 不认识的对象时尝试转成字符串；
6. 最终字符串保存到 `params`，稍后写进 CSV 的一个单元格。

#### C9:L11–L14 `[核心]`

```python
    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
    ):
```

- 创建一个包含两个三元组的元组；
- 第一个三元组代表训练集，第二个代表验证集；
- 每次循环把名称、特征、标签拆给三个变量；
- 内层循环每个模型执行两次；
- 列表中故意没有 `test`，因此 Day 1 不评价测试集；
- L14 的 `):` 同时关闭外层元组并开始循环代码块。

#### C9:L15 `[核心]`

```python
        prediction = model.predict(X_split)
```

- 使用已经训练好的当前模型预测；
- 当 `split_name == "train"` 时输出形状为 `(902,)`；
- 当 `split_name == "valid"` 时输出形状为 `(113,)`；
- `.predict()` 不会再次训练模型；
- 训练集预测用来观察拟合程度，验证集预测用来比较泛化。

#### C9:L16 `[工程]`

```python
        records.append({
```

- `{` 开始构造一行结果字典；
- `records.append(...)` 把这行增加到列表末尾；
- L16–L29 共同构成一次 `append` 调用。

#### C9:L17 `[工程]`

```python
            "run_id": RUN_ID,
```

记录实验名称，用于区分不同运行。

#### C9:L18 `[核心]`

```python
            "model": model_name,
```

记录这一行属于哪个模型。

#### C9:L19 `[核心]`

```python
            "split": split_name,
```

记录这一行属于训练集还是验证集。解释指标时必须同时看这一列。

#### C9:L20 `[工程]`

```python
            "n_samples": len(y_split),
```

记录本次评价样本数：训练为 902，验证为 113。

#### C9:L21 `[工程]`

```python
            "seed": SEED,
```

记录随机种子 42，帮助复现。

#### C9:L22 `[工程]`

```python
            "featurizer": "ECFP radius=2 size=1024",
```

保存特征器的文字说明，不是把 1024 列特征复制进这一单元格。

#### C9:L23 `[工程]`

```python
            "splitter": "scaffold",
```

记录数据划分协议。

#### C9:L24 `[工程]`

```python
            "target_transformer": "none",
```

记录标签没有额外标准化。它不表示分子没有经过 ECFP 特征化。

#### C9:L25 `[工程]`

```python
            "fit_seconds": fit_seconds,
```

记录当前模型训练一次的耗时。同一个模型的 train 行和 valid 行会保存同一个训练耗时，因为模型只训练了一次。

#### C9:L26 `[高级]`

```python
            "params": params,
```

保存前面序列化后的参数字符串，便于以后知道本次模型怎么配置。

#### C9:L27 `[工程]`

```python
            "source_commit": SOURCE_COMMIT,
```

记录运行时 Git `HEAD`。如果当时工作区有未提交改动，仅有这个哈希并不能完全还原代码，因此还要查看 `GIT_DIRTY`。

#### C9:L28 `[高级但重要]`

```python
            **regression_metrics(y_split, prediction),
```

- 先调用指标函数，得到类似 `{"mae_logS": ..., "rmse_logS": ..., "r2": ...}`；
- `**` 把这三个键值对展开并合并进当前结果字典；
- 没有 `**` 时，整个指标字典会成为一个嵌套值，而不是三列；
- 真实标签是 `y_split`，预测标签是 `prediction`，顺序不能颠倒。

#### C9:L29 `[工程]`

```python
        })
```

`}` 关闭一行结果字典，`)` 关闭 `.append(...)`。内层循环一次至此产生一行记录。

#### C9:L31 `[核心]`

```python
results = pd.DataFrame.from_records(records)
```

- 把十个字典组成的列表转换成 pandas 表格；
- 每个字典成为一行；
- 相同字典键成为列；
- 当前结果形状为 `(10, 14)`。

#### C9:L32 `[工程]`

```python
assert len(results) == len(models) * 2
```

- `len(models)` 是 5；
- 每个模型有 train 和 valid 两行，因此乘 2；
- 如果结果不是 10 行就触发 `AssertionError`；
- 这项检查不能判断指标是否科学，只检查记录数量。

#### C9:L33 `[核心]`

```python
results
```

这是 Jupyter 特有的方便写法：代码单元最后一个表达式会自动显示。普通 `.py` 文件中单写变量名不会打印。

### 循环结束后的状态

| 变量 | 状态 |
|---|---|
| `fitted_models` | 五个已训练模型 |
| `records` | 十个字典 |
| `results` | 10 行 × 14 列表格 |

---

## 代码 Cell 10：只用验证集排名并制作汇总表

### 原代码

```python
valid_ranked = (
    results.loc[results["split"] == "valid"]
    .sort_values(["rmse_logS", "mae_logS"])
    .reset_index(drop=True)
)
best_model_name = valid_ranked.loc[0, "model"]

summary = results.pivot(
    index="model", columns="split", values=["mae_logS", "rmse_logS", "r2"]
).sort_values(("rmse_logS", "valid"))
display(summary.round(4))
print("Best model by validation RMSE:", best_model_name)
print("The test split remains unevaluated in this Day 1 run.")
```

### 逐行解释

#### C10:L01–L05 `[高级语法，核心逻辑]`

```python
valid_ranked = (
    results.loc[results["split"] == "valid"]
    .sort_values(["rmse_logS", "mae_logS"])
    .reset_index(drop=True)
)
```

按执行顺序拆开：

1. L01 用圆括号允许方法链跨多行；
2. L02 的 `results["split"] == "valid"` 产生十个 `True/False`；
3. `.loc[...]` 只保留五个验证集行；
4. L03 先按 `rmse_logS` 从小到大排，如相同再按 `mae_logS`；
5. L04 把旧索引重置为 0–4，`drop=True` 不保留旧索引列；
6. L05 关闭整个表达式；
7. 最终保存为 `valid_ranked`，形状 `(5, 14)`。

为什么不能按训练误差选择？复杂模型可能只记住训练数据，训练误差很低却不能泛化。

#### C10:L06 `[核心]`

```python
best_model_name = valid_ranked.loc[0, "model"]
```

- `.loc[行标签, 列标签]` 选择单个单元格；
- 排名重置后第 0 行就是验证 RMSE 最低者；
- 当前结果字符串为 `"random_forest"`；
- 严格表述应是“这五个固定候选模型、这一次验证划分、以 RMSE 为主指标时最好”。

#### C10:L08–L10 `[高级]`

```python
summary = results.pivot(
    index="model", columns="split", values=["mae_logS", "rmse_logS", "r2"]
).sort_values(("rmse_logS", "valid"))
```

- L08 开始 `pivot`，把长表变为宽表；
- L09 指定模型为行、train/valid 为列层级、三个指标为值；
- 得到多层列，例如 `("rmse_logS", "valid")`；
- L10 用这个元组指定“验证 RMSE”列并排序；
- 这里只重排和展示结果，不会重新训练。

#### C10:L11 `[核心]`

```python
display(summary.round(4))
```

- `.round(4)` 生成显示到四位小数的表；
- 不会降低 `results` 中保存的原始精度；
- `display` 在 Jupyter 中以富表格形式展示。

#### C10:L12 `[核心]`

```python
print("Best model by validation RMSE:", best_model_name)
```

打印当前选择结果。当前为随机森林，但不代表随机森林对所有数据都最好。

#### C10:L13 `[核心]`

```python
print("The test split remains unevaluated in this Day 1 run.")
```

打印证据边界提醒。本行文字本身不是强制锁；真正的保护来自训练/评价代码中没有放入测试数据，以及研究者遵守协议。

### 当前保存结果怎样读

| 模型 | Train R² | Valid RMSE | Valid R² | 初步解释 |
|---|---:|---:|---:|---|
| Dummy mean | 0.0000 | 2.1714 | -0.2065 | 最低参考 |
| Ridge (`alpha=1.0`) | 0.9343 | 2.3795 | -0.4489 | 这项设置训练拟合强、验证泛化差 |
| 不限深树 | 0.9799 | 1.9877 | -0.0111 | 明显过拟合 |
| 限制树 | 0.4935 | 2.1106 | -0.1399 | 限制降低训练拟合，但本次验证仍弱 |
| 随机森林 | 0.7574 | 1.7031 | 0.2578 | 当前固定候选中验证 RMSE 最低 |

Dummy 的验证 R² 可能为负，因为它学的是**训练标签均值**，而验证 R² 的参考与验证数据分布有关。这不是程序错误。

---

## 代码 Cell 11：把结果保存为 CSV 和 JSON

这一格属于 `[工程]`。第一次学习重点是知道“保存了什么”，不要求独立写出嵌套字典。

### 原代码

```python
metrics_path = RESULTS_DIR / "baseline_metrics.csv"
config_path = RESULTS_DIR / "run_config.json"

results.to_csv(metrics_path, index=False, encoding="utf-8")
run_config = {
    "run_id": RUN_ID,
    "branch": BRANCH,
    "source_commit": SOURCE_COMMIT,
    "working_tree_dirty_during_run": GIT_DIRTY,
    "seed": SEED,
    "task": tasks[0],
    "target_space": "original logS (mols per litre)",
    "featurizer": {"name": "ECFP", "radius": 2, "size": 1024},
    "splitter": "scaffold",
    "transformers": [],
    "split_sizes": expected_sizes,
    "selection_metric": "validation RMSE",
    "best_model_by_validation_rmse": best_model_name,
    "test_policy": "Not evaluated on Day 1; the legacy notebook already exposed this split.",
    "versions": VERSIONS,
    "model_params": {name: model.get_params(deep=True) for name, model in models.items()},
}
config_path.write_text(
    json.dumps(run_config, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)

print("Saved metrics:", metrics_path.relative_to(REPO_ROOT))
print("Saved config:", config_path.relative_to(REPO_ROOT))
```

### 逐行解释

#### C11:L01–L02 `[工程]`

```python
metrics_path = RESULTS_DIR / "baseline_metrics.csv"
config_path = RESULTS_DIR / "run_config.json"
```

- `RESULTS_DIR` 是 `Path`；这里 `/` 表示拼接路径，不是除法；
- `metrics_path` 指向模型指标 CSV；
- `config_path` 指向实验配置 JSON；
- 这两行只创建路径对象，还没有写文件。

#### C11:L04 `[工程]`

```python
results.to_csv(metrics_path, index=False, encoding="utf-8")
```

- 把 `results` DataFrame 写成 CSV；
- `index=False` 防止 pandas 行号被当成额外数据列；
- `encoding="utf-8"` 指定文字编码；
- 重复运行会覆盖同一路径的现有文件。

#### C11:L05 `[工程]`

```python
run_config = {
```

开始构造记录实验配置的字典。L05–L22 是一个整体。

#### C11:L06 `[工程]` — `"run_id": RUN_ID,`

记录本次运行名称。

#### C11:L07 `[工程]` — `"branch": BRANCH,`

记录代码执行时的 Git 分支。Notebook 中的保存输出可能是历史分支快照。

#### C11:L08 `[工程]` — `"source_commit": SOURCE_COMMIT,`

记录执行时 `HEAD` 提交哈希。

#### C11:L09 `[工程]` — `"working_tree_dirty_during_run": GIT_DIRTY,`

记录运行时是否有未提交改动。现有历史运行值为 `true`，因此仅凭 `source_commit` 不能精确还原当时代码。

此外当前实现有一个小边界问题：如果 Git 命令失败，辅助函数返回非空字符串 `"unavailable"`，`bool("unavailable")` 也会得到 `True`。因此这里的 `True` 原则上可能表示“有改动”或“Git 状态无法获取”；本次历史记录结合环境判断为工作区有改动。

#### C11:L10 `[工程]` — `"seed": SEED,`

记录随机种子。

#### C11:L11 `[工程]` — `"task": tasks[0],`

从任务列表取第 0 个名称。Python 从 0 开始计数；这里假设列表不为空。

#### C11:L12 `[核心概念]`

```python
    "target_space": "original logS (mols per litre)",
```

记录目标没有标准化。更严谨的理解是预测 `log10` 溶解度空间，原始溶解度以 mol/L 表达；不要把 logS 数值直接称为“mol/L 单位的小数”。

#### C11:L13 `[工程]`

```python
    "featurizer": {"name": "ECFP", "radius": 2, "size": 1024},
```

使用嵌套字典记录特征器设置。1024 维是哈希指纹位置，不是 1024 种可逐一命名的实验理化性质。

#### C11:L14 `[工程]` — `"splitter": "scaffold",`

记录请求的划分方法。

#### C11:L15 `[工程]` — `"transformers": [],`

保存空列表，表示没有标签转换。

#### C11:L16 `[工程]` — `"split_sizes": expected_sizes,`

复用前面字典，记录 902/113/113。

#### C11:L17 `[工程]` — `"selection_metric": "validation RMSE",`

声明选择模型的主指标，防止看到结果后临时更换标准。

#### C11:L18 `[工程]` — `"best_model_by_validation_rmse": best_model_name,`

保存当前验证 RMSE 最低模型名称。

#### C11:L19 `[工程]`

```python
    "test_policy": "Not evaluated on Day 1; the legacy notebook already exposed this split.",
```

把为什么本轮不用测试集写入配置，提醒未来读者证据边界。

#### C11:L20 `[工程]` — `"versions": VERSIONS,`

嵌套保存软件版本字典。

#### C11:L21 `[高级]`

```python
    "model_params": {name: model.get_params(deep=True) for name, model in models.items()},
```

这是字典推导式：遍历五个模型，生成“模型名称 → 参数字典”。它记录超参数，不是模型学到的全部权重和树结构。

#### C11:L22 `[工程]`

```python
}
```

关闭 `run_config` 字典。

#### C11:L23–L26 `[高级]`

```python
config_path.write_text(
    json.dumps(run_config, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
```

- L23 调用路径对象的写文本方法；
- L24 把 Python 字典序列化为有两格缩进的 JSON；
- `ensure_ascii=False` 允许中文原样保存；
- `default=str` 将不受支持对象转成字符串；
- `+ "\n"` 在文件结尾增加换行；
- L25 指定 UTF-8；
- L26 关闭调用；返回的字符数未保存。

#### C11:L28–L29 `[工程]`

```python
print("Saved metrics:", metrics_path.relative_to(REPO_ROOT))
print("Saved config:", config_path.relative_to(REPO_ROOT))
```

- `.relative_to(REPO_ROOT)` 把绝对路径缩短成仓库内相对路径；
- 只影响显示，不移动文件；
- 两个输出分别是 `baseline_metrics.csv` 和 `run_config.json`。

### CSV 和 JSON 分工

| 文件 | 保存内容 | 方便做什么 |
|---|---|---|
| CSV | 每个模型/数据集的指标行 | 用 Excel、pandas 查看和比较 |
| JSON | 数据、版本、参数、提交和测试策略 | 复现实验与审计 |

---

## 代码 Cell 12：自动检查并渲染结论

### 原代码

```python
assert metrics_path.exists() and config_path.exists()
assert set(results["split"]) == {"train", "valid"}
assert "test" not in set(results["split"])
assert results[["mae_logS", "rmse_logS", "r2"]].notna().all().all()

dummy_valid_rmse = float(valid_ranked.loc[
    valid_ranked["model"] == "dummy_mean", "rmse_logS"
].iloc[0])
best_valid_rmse = float(valid_ranked.loc[0, "rmse_logS"])
tree_rows = results.loc[results["model"] == "decision_tree_overfit_diagnostic"]
tree_train_r2 = float(tree_rows.loc[tree_rows["split"] == "train", "r2"].iloc[0])
tree_valid_r2 = float(tree_rows.loc[tree_rows["split"] == "valid", "r2"].iloc[0])

display(Markdown(
    f"""## 3. Day 1 可验收结论

- 验证 RMSE 最低的固定候选模型：`{best_model_name}`（{best_valid_rmse:.4f} logS）。
- 它相对 Dummy 的验证 RMSE 改善：`{dummy_valid_rmse - best_valid_rmse:.4f}` logS。
- 不限深决策树的 train/valid R² 为 `{tree_train_r2:.4f}` / `{tree_valid_r2:.4f}`，训练—验证差距表明明显过拟合。
- 本轮未生成测试集预测；结论只适用于当前一次 scaffold train/validation 比较。

**Day 1 automated checks passed.**
"""
))
```

### 逐行解释

#### C12:L01 `[工程]`

```python
assert metrics_path.exists() and config_path.exists()
```

- `.exists()` 分别检查两个文件；
- `and` 要求左右都为真；
- 任一个不存在就触发 `AssertionError`；
- 只证明文件存在，不证明内容一定正确。

#### C12:L02 `[工程]`

```python
assert set(results["split"]) == {"train", "valid"}
```

- 把 `split` 列转换成去重集合；
- `==` 是相等判断，不是赋值；
- 要求结果中恰好只有 train 和 valid。

#### C12:L03 `[核心边界]`

```python
assert "test" not in set(results["split"])
```

明确检查结果表中没有 test 行。它不能证明其他代码从未访问测试集，但能检查本流程没有保存测试指标。

#### C12:L04 `[高级]`

```python
assert results[["mae_logS", "rmse_logS", "r2"]].notna().all().all()
```

从左到右：

1. 双层方括号选择三列并保持 DataFrame；
2. `.notna()` 生成同形状真假表；
3. 第一个 `.all()` 检查每列所有行；
4. 第二个 `.all()` 再检查所有列；
5. 最终要求没有 NaN；
6. 它没有单独检查正负无穷。

#### C12:L06–L08 `[高级]`

```python
dummy_valid_rmse = float(valid_ranked.loc[
    valid_ranked["model"] == "dummy_mean", "rmse_logS"
].iloc[0])
```

- L06 开始从表中提取一个值；
- L07 用模型名称条件筛行，并选择 RMSE 列；
- `.loc[...]` 得到只含一项的 Series；
- L08 的 `.iloc[0]` 按位置取第一项；
- `float` 转成普通小数；
- 保存为 Dummy 验证 RMSE。

#### C12:L09 `[核心]`

```python
best_valid_rmse = float(valid_ranked.loc[0, "rmse_logS"])
```

从排序表第 0 行取得最佳验证 RMSE。当前为约 `1.7031`。

#### C12:L10 `[核心]`

```python
tree_rows = results.loc[results["model"] == "decision_tree_overfit_diagnostic"]
```

筛选不限深决策树的两行：一行 train，一行 valid。

#### C12:L11 `[核心]`

```python
tree_train_r2 = float(tree_rows.loc[tree_rows["split"] == "train", "r2"].iloc[0])
```

进一步选训练行的 R²，当前约 `0.9799`。

#### C12:L12 `[核心]`

```python
tree_valid_r2 = float(tree_rows.loc[tree_rows["split"] == "valid", "r2"].iloc[0])
```

选验证行的 R²，当前约 `-0.0111`。训练很好、验证很差的巨大落差是过拟合证据。

#### C12:L14 `[工程]`

```python
display(Markdown(
```

开始两个嵌套调用：`Markdown(...)` 把文本包装为 Markdown，`display(...)` 在 Notebook 中渲染。

#### C12:L15 `[高级语法]`

```python
    f"""## 3. Day 1 可验收结论
```

- `f` 开启变量替换；
- 三个引号允许字符串跨多行；
- `##` 位于字符串内部，是 Markdown 二级标题，不是 Python 注释。

#### C12:L17 `[核心结论]`

这一物理行生成第一条项目符号，把 `best_model_name` 和格式化到四位小数的 `best_valid_rmse` 插入文字。反引号属于 Markdown 行内代码样式。

#### C12:L18 `[核心结论]`

花括号内先计算 `dummy_valid_rmse - best_valid_rmse`，再用 `:.4f` 显示四位小数。它表示 RMSE 数值差，不是百分比改善。

#### C12:L19 `[核心结论]`

把训练与验证 R² 插入同一句，用差距说明不限深树过拟合。结论来自两个数字的对照，不是看到“决策树”名字就先判定。

#### C12:L20 `[核心边界]`

静态文字说明本轮没有生成测试预测，结论只针对当前一次 scaffold train/validation 比较。

#### C12:L22 `[工程]`

```text
**Day 1 automated checks passed.**
```

双星号是 Markdown 加粗。真正执行检查的是前面的 `assert`；这句话本身不会检查任何东西。

#### C12:L23 `[高级语法]`

```python
"""
```

关闭三引号 f-string。

#### C12:L24 `[高级语法]`

```python
))
```

第一个 `)` 关闭 `Markdown(...)`，第二个关闭 `display(...)`。

### 这一格没有证明什么

- 没证明随机森林是 ESOL 或粘合剂任务的普遍最佳模型；
- 没证明当前结果能迁移到真实粘合剂；
- 没证明验证集以后可以无限次调参；
- 没证明测试集仍是严格未见，因为旧 Notebook 历史已看过该 split；
- 没证明工作区能够只靠记录的旧提交完整复现，因为历史运行时 `GIT_DIRTY` 为真。

---

## 下半部分必须掌握的十句话

1. 指标函数接收真实值和预测值，返回 MAE、RMSE、R²。
2. 创建模型对象时还没有训练。
3. `.fit(X_train, y_train)` 才让模型学习。
4. `.predict(X_split)` 使用已经学到的模型，不会重新训练。
5. 外层五个模型、内层两个 split，所以结果有十行。
6. 训练指标说明对已见数据的拟合，验证指标用于当前候选比较。
7. 不限深树训练 R² 很高但验证 R² 很差，属于过拟合。
8. 随机森林只是在当前五个候选、当前验证划分下最好。
9. CSV 保存指标，JSON 保存运行配置和证据边界。
10. 测试集本轮不参与训练、选择和结果表。

完成后返回[练习与验收](exercises.md)，先不看答案区做第一遍。
