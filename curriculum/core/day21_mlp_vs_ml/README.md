# Day 21：用同一协议比较 MLP 与传统机器学习

> 状态：待学习、待运行。今天只建立公平比较方法，不预设哪个模型获胜。

## 今天为什么学

导师提到“传统机器学习算法和神经网络结合”之前，
首先要知道传统模型和 MLP 各自单独能做到什么。

如果模型使用不同数据划分、不同评价指标或不同测试次数，
最后的分数不能公平比较。

今天先用 ESOL 练习数据完成同协议比较。
ESOL 的结果只证明流程能运行，不代表粘合剂实验结论。

## 前置条件

- 完成 [Day 20](../day20_regularization_early_stopping/README.md)；
- 知道训练集、验证集和测试集的职责；
- 能解释 RMSE、MAE 和 `R²`；
- 已冻结一个 MLP 设置；
- 今天不反复查看测试集。

## 今日产出

1. 一份写在运行之前的比较协议；
2. 传统模型与 MLP 的统一结果表；
3. 每个模型的训练时间和预测指标；
4. 一段关于“差异是否足够稳定”的说明；
5. 一个进入混合模型阶段的基准模型。

## 核心概念

### 1. 同一协议

至少保持以下内容相同：

- 相同样本和相同训练/验证索引；
- 相同输入特征和预测目标；
- 相同缺失值规则；
- 相同 RMSE、MAE、`R²` 定义；
- 相同随机种子集合；
- 相同的测试集封存规则。

### 2. 公平不等于所有步骤完全相同

Ridge 和 MLP 对特征尺度敏感，需要标准化。
树模型通常不依赖标准化。
可以为每类模型使用合理的预处理，但必须预先写明且放进 Pipeline。

### 3. 单次分数不是结论

小数据下，一个随机划分可能使排名发生变化。
今天先建立比较表，后续需要结合交叉验证和多种子结果解释。

### 4. 模型复杂不代表一定更好

表格小样本中，Ridge、随机森林或梯度提升可能胜过 MLP。
这不是神经网络“失败”，而是数据规模和特征形式共同决定的结果。

## 分步骤任务

1. 复制并记录 Day 20 冻结的数据索引。
2. 写下特征版本、目标列和评价指标。
3. 预先声明 Ridge、Random Forest、Gradient Boosting 和 MLP。
4. 为需要缩放的模型建立 Pipeline。
5. 确认所有模型只在同一个 `X_train, y_train` 上调用 `fit()`。
6. 对同一个 `X_valid` 生成预测。
7. 记录 RMSE、MAE、`R²` 和训练时间。
8. 检查结果表是否每个模型只有一行。
9. 不根据验证结果临时增加大量候选。
10. 写出最强传统基线和 MLP 之间的差值。
11. 说明差值是否小到可能由随机性造成。
12. 保存协议与结果，但不要写成粘合剂结论。

## 核心代码骨架

```python
from time import perf_counter
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "ridge": make_pipeline(SimpleImputer(strategy="median"),
                           StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, random_state=42)),
    "gradient_boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(random_state=42)),
    "mlp": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64,), alpha=0.001,
                     early_stopping=True, max_iter=500, random_state=42)),
}

rows = []
for name, model in models.items():
    start = perf_counter()
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    rows.append({
        "model": name,
        "rmse": root_mean_squared_error(y_valid, prediction),
        "mae": mean_absolute_error(y_valid, prediction),
        "r2": r2_score(y_valid, prediction),
        "fit_seconds": perf_counter() - start,
    })
comparison = pd.DataFrame(rows).sort_values("rmse")
print(comparison)
```

## 只解释今天新增的语法

- `perf_counter()` 返回适合计算耗时的高精度时间值。
- 两次时间值相减得到模型训练和预测所用的秒数。
- `rows.append({...})` 把本轮模型结果追加到列表。
- `DataFrame(rows)` 把“字典组成的列表”转换成表格。
- `sort_values("rmse")` 按 RMSE 从小到大排列。
- 多行 Pipeline 保证预处理只从训练数据学习。

## 与真实粘合剂项目的边界

今天使用的是练习数据和已准备好的特征。
真实粘合剂数据需要化学组确认：

- 粘合剂化学体系；
- 一行代表配方、试样还是重复实验；
- 预测性能、单位与测试标准；
- 配方、固化、基材和测试条件；
- 哪些记录可以用于算法、哪些不能公开。

在这些内容确定前，今天的模型排名不能迁移为实验结论。
结果只能写成“在本次固定协议和练习数据上，某模型验证 RMSE 较低”，
并同时报告数据版本、划分、参数、种子和测试集封存状态。

## 常见错误

| 错误 | 为什么不公平 | 正确处理 |
|---|---|---|
| 不同模型使用不同划分 | 样本难度不同 | 复用同一索引 |
| 先对全数据标准化 | 验证信息泄漏 | 在 Pipeline 内拟合 |
| MLP 调很多次而 Ridge 只调一次 | 搜索预算不同 | 预先限定预算 |
| 只报告最好分数 | 隐藏不稳定性 | 保留完整结果 |
| 用 ESOL 结果声称粘合剂有效 | 任务和标签不同 | 只说明流程学习 |

## 完成标准

- [ ] 比较协议在运行前写好；
- [ ] 所有模型使用相同训练和验证索引；
- [ ] 评价指标定义完全相同；
- [ ] 预处理位于各自 Pipeline 内；
- [ ] 结果表包含传统模型和 MLP；
- [ ] 测试集没有参与选择；
- [ ] 结论注明这是练习数据结果。

## 自测问题

1. 为什么树模型不缩放也可以被视为公平比较？
2. 如果 MLP 试了 100 组参数，Ridge 只试 1 组，会有什么问题？
3. 为什么验证 RMSE 只差一点时不能急着宣布赢家？
4. ESOL 上的最好模型能直接用于粘合剂吗？
5. 今天确定的是模型结论，还是比较协议？

## 上一天 / 下一天

- 上一天：[Day 20：MLP 正则化与早停](../day20_regularization_early_stopping/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 22：混合模型方案与泄漏风险](../day22_hybrid_risk/README.md)
