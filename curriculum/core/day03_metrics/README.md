# Day 3：真正读懂回归评价指标

## 今天为什么学

模型输出一列预测值后，我们需要回答“预测得有多差”。

如果不理解评价指标，就会把 RMSE 说成准确率、把负 R² 当成程序报错，或者只挑对自己有利的数字。

今天专门学习 ESOL 回归任务使用的三个指标：

```text
MAE、RMSE、R²
```

## 前置条件

- 已完成 [Day 2 安全重跑](../day02_safe_rerun/README.md)；
- 知道 `y_true` 是真实答案，`y_pred` 是预测结果；
- 知道一维数组的形状写作 `(n_samples,)`；
- 能运行一个独立代码单元；
- 今天不比较新的复杂模型。

## 今日产出

今天应由你完成：

1. 一个可以重复调用的回归指标函数；
2. 一张真实值、预测值和误差的小表；
3. 一次手算 MAE 与代码结果的核对；
4. 对 MAE、RMSE、R² 方向的中文说明；
5. 对“负 R² 是否意味着代码错误”的回答。

不要预先填写结果数字，数字必须来自你自己的运行。

## 核心概念

### 1. MAE

MAE 是绝对误差的平均值。

- 越小越好；
- 单位与预测目标相同；
- 每个样本的误差按线性方式计入。

### 2. RMSE

RMSE 先平方误差、求平均，再开平方。

- 越小越好；
- 单位仍与预测目标相同；
- 大误差会受到更重惩罚；
- 不能称为“准确率”。

### 3. R²

R² 比较模型与“总是猜平均值”之间的相对表现。

- 越大通常越好；
- 1 表示在这些样本上预测完全一致；
- 0 附近表示与均值参考相近；
- 小于 0 表示比相应均值参考还差；
- 负数本身不代表 Python 出错。

## 分步骤任务

### 第一步：准备极小例子

先用三个或四个你能手算的真实值和预测值。

不要一开始就使用上千行数组。

### 第二步：建立误差表

创建 pandas DataFrame，并增加：

- `error`；
- `absolute_error`；
- `squared_error`。

### 第三步：手算 MAE

把绝对误差相加后除以样本数，再与 sklearn 输出核对。

### 第四步：定义统一函数

函数只接收真实值和预测值，并返回一个三项字典。

### 第五步：解释结果

使用完整句子，不只抄写数字。

例如：“RMSE 大于 MAE，说明较大的误差受到了更明显的惩罚。”

## 核心代码骨架

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_true = np.array([-3.0, -2.0, -1.0, 0.0])
y_pred = np.array([-2.5, -2.4, -0.2, -0.1])

error_table = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred,
})

error_table["error"] = error_table["y_pred"] - error_table["y_true"]
error_table["absolute_error"] = error_table["error"].abs()
error_table["squared_error"] = error_table["error"] ** 2

manual_mae = error_table["absolute_error"].mean()

def regression_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }

scores = regression_metrics(y_true, y_pred)

print(error_table)
print("manual MAE:", manual_mae)
print("function scores:", scores)

assert np.isclose(manual_mae, scores["mae"])
```

今天新增语法：

- `table["新列"] = ...`：给 DataFrame 增加一列；
- `.abs()`：逐项取绝对值；
- `** 2`：平方；
- `np.isclose(a, b)`：允许微小浮点误差地比较两个数。

## 常见错误

- 把 `y_true` 和 `y_pred` 的顺序写反；
- 一个数组有四项，另一个只有三项；
- 直接把 RMSE 称为百分比准确率；
- 认为 R² 的合法范围只能是 0 到 1；
- 把训练指标与验证指标放在同一列却不标 split；
- 忘记 RMSE 中的平方根；
- 用四舍五入后的显示值继续计算；
- 根据一个指标就断言模型已经适合真实粘合剂数据。

## 完成标准

- 能手算小例子的 MAE；
- 能解释 MAE 与 RMSE 为什么不同；
- 知道三个指标各自是越大还是越小越好；
- 能解释负 R²；
- 函数对同长度的一维数组正常工作；
- 手算 MAE 与函数 MAE 通过 `np.isclose`；
- 不使用 test 来选择指标或模型；
- 能用中文写出一段结果解释。

## 自测问题

1. MAE 和 RMSE 的单位是否与 logS 相同？
2. 为什么 RMSE 对大错误更敏感？
3. R² 小于 0 一定是代码错误吗？
4. 为什么 RMSE 不能叫准确率？
5. `error` 中的正负号表达什么？
6. `.abs()` 和 `** 2` 分别做什么？
7. 为什么用 `np.isclose` 而不是直接比较所有浮点数？
8. 比较两个模型前，需要固定哪些实验条件？

## 导航

- 上一天：[Day 2 安全重跑](../day02_safe_rerun/README.md)
- 下一天：[Day 4 决策树](../day04_decision_tree/README.md)
