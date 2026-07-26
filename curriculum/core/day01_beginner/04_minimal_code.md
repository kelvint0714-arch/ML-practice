# Day 1.4：只运行能解释的最小代码

下面使用人工写出的极小数据，不下载 ESOL、不使用 DeepChem、不访问 Git，也不保存文件。

在同一个临时 Notebook 中，从上到下依次运行。后面的代码块会使用前面已经创建的变量，因此不能单独从第四步开始。每次只运行一个代码块；先读解释、猜结果，再运行。

## 第一步：准备训练答案

```python
import numpy as np

y_train = np.array([-3.0, -2.0, -1.0, -2.0])
mean_value = y_train.mean()

print(mean_value)
```

逐行解释：

| 代码 | 含义 |
|---|---|
| `import numpy as np` | 导入处理数字数组的 NumPy，并简称为 `np` |
| `y_train = ...` | 创建四个训练答案，并给它起名 `y_train` |
| `np.array(...)` | 把方括号里的数字转换成 NumPy 数组 |
| `.mean()` | 计算数组平均值 |
| `mean_value = ...` | 给计算结果起名 `mean_value` |
| `print(...)` | 把结果显示出来 |

预期输出是 `-2.0`。

## 第二步：让均值基线预测

```python
y_valid = np.array([-1.0, -3.0])
prediction = np.full(
    shape=len(y_valid),
    fill_value=mean_value,
)

print(prediction)
```

逐行解释：

| 代码 | 含义 |
|---|---|
| `y_valid = ...` | 保存两个验证样本的真实答案 |
| `len(y_valid)` | 计算验证答案有几个，这里是2 |
| `np.full(...)` | 创建一个全部填入同一数值的数组 |
| `shape=...` | 指定要创建几个预测值 |
| `fill_value=...` | 指定每个位置填入训练均值 |
| `prediction = ...` | 给预测结果起名 |

预期输出是：

```text
[-2. -2.]
```

## 第三步：计算 MAE 和 RMSE

```python
absolute_errors = np.abs(prediction - y_valid)
squared_errors = (prediction - y_valid) ** 2

mae = absolute_errors.mean()
rmse = np.sqrt(squared_errors.mean())

print("MAE:", mae)
print("RMSE:", rmse)
```

逐行解释：

| 代码 | 含义 |
|---|---|
| `prediction - y_valid` | 对应位置相减，得到每个样本的预测误差 |
| `np.abs(...)` | 把负误差变为正的距离 |
| `** 2` | 把每个误差平方 |
| `.mean()` | 求平均 |
| `np.sqrt(...)` | 开平方 |

这里的 MAE 和 RMSE 都应该是 `1.0`，与纸笔计算一致。

## 第四步：把手算算法换成 sklearn 模型

```python
from sklearn.dummy import DummyRegressor

X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
X_valid = np.array([[1.5], [2.5]])

model = DummyRegressor(strategy="mean")
model.fit(X_train, y_train)
sklearn_prediction = model.predict(X_valid)

print(sklearn_prediction)
```

逐行解释：

| 代码 | 含义 |
|---|---|
| `from ... import ...` | 从 sklearn 中取出均值基线算法 |
| `X_train` | 四个训练样本的输入，每行一个样本 |
| `X_valid` | 两个验证样本的输入 |
| `model = ...` | 创建尚未训练的模型 |
| `strategy="mean"` | 指定算法永远预测训练答案均值 |
| `model.fit(...)` | 让模型读取训练输入和训练答案 |
| `model.predict(...)` | 使用已经训练的模型预测验证输入 |

它应该同样输出 `[-2. -2.]`。这说明 sklearn 的均值基线与刚才的手算规则一致。

## 第五步：只增加一个新算法

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=2, random_state=42)
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_valid)

print(tree_prediction)
```

现在只需要理解：

- `tree` 是另一种候选算法训练得到的模型；
- 它不再永远预测同一个平均值；
- 它仍然使用同样的 `fit → predict` 骨架；
- `max_depth=2` 限制树的复杂度；
- 决策树怎样切分，会在 Day 4 专门学习。

## 今天真正要记住的代码骨架

下面是结构示意，`SomeModel` 和 `metric` 是占位名称，**不能直接复制运行**：

```python
model = SomeModel()
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
score = metric(y_valid, prediction)
```

不要背上面的类名。你应该能够指出每一行属于“创建、训练、预测、评价”中的哪一步。

下一步：[Day 1.5 ESOL Notebook 对照地图](05_esol_notebook_map.md)。
