# Day 13：公平比较多个传统模型

> 状态：待学习。本文定义比较方法，不预告哪个模型会胜出。

## 今天为什么学

两个模型的分数只有在数据、划分、输入信息和指标一致时才能比较。
如果 Ridge 使用标准化数据、随机森林使用另一批样本，或者两个模型采用不同验证折，分数高低没有公平含义。

今天建立统一交叉验证表，让 Dummy、Ridge、随机森林和梯度提升在相同证据条件下比较。

## 前置条件

- 会使用 Pipeline；
- 知道调参不能偷看测试集；
- 能解释 RMSE、MAE 和 R² 的方向；
- 理解固定随机种子的作用。

## 今日产出

完成学习后，你应该得到：

1. 一张模型比较协议表；
2. 每个模型相同五折下的验证分数；
3. 均值、标准差和训练耗时汇总；
4. 一段不夸大“第一名”的结论。

## 核心概念

### 1. 公平比较的五个相同

| 项目 | 要求 |
|---|---|
| 样本 | 使用完全相同的训练行 |
| 输入 | 使用相同特征信息 |
| 划分 | 使用同一个 `cv` 对象 |
| 指标 | 使用相同 MAE、RMSE、R² |
|选择规则 | 运行前写明主指标 |

预处理可以因模型需要而不同，但不能让某个模型额外看到验证信息。

### 2. 为什么保留 Dummy

Dummy 不学习特征与目标的复杂关系。
如果复杂模型不能稳定超过 Dummy，首先应检查数据、特征和验证设计，而不是继续堆模型。

### 3. 均值不够

平均 RMSE 反映总体误差，标准差反映不同折之间的波动。
平均值略优但波动很大的模型，不一定是更可靠的选择。

下面的普通随机 `KFold` 只演示公平比较程序，不等于 scaffold 交叉验证。
对分子任务下结论时，还要根据研究问题决定是否按骨架或其他组别划分。

## 分步骤任务

1. 写下本轮唯一主指标，例如验证 RMSE。
2. 固定同一个五折 `KFold`。
3. 为需要缩放的 Ridge 建立 Pipeline。
4. 为所有模型保留相同的缺失值策略。
5. 用 `cross_validate` 一次记录多个指标。
6. 把每一折结果保留下来，不只抄平均值。
7. 汇总验证均值、验证标准差和拟合耗时。
8. 检查复杂模型是否稳定超过 Dummy。
9. 写明当前比较没有使用测试集。

## 核心代码骨架

```python
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "dummy": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor()),
    "ridge": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    ),
    "gradient_boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(random_state=42),
    ),
}
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append({
        "model": name,
        "mae_mean": -scores["test_mae"].mean(),
        "rmse_mean": -scores["test_rmse"].mean(),
        "rmse_std": scores["test_rmse"].std(),
        "r2_mean": scores["test_r2"].mean(),
        "fit_seconds_mean": scores["fit_time"].mean(),
    })
comparison = pd.DataFrame(rows).sort_values("rmse_mean")
print(comparison)
```

## 只解释今天新增的语法

- `make_pipeline(a, b, c)` 按顺序建立步骤，并自动生成步骤名。
- `scoring` 字典让一次交叉验证返回多种指标。
- `cross_validate` 返回一个“名称到数组”的字典，每个数组对应各折。
- `.mean()` 计算折间平均值，`.std()` 计算折间标准差。
- `sort_values("rmse_mean")` 只改变显示顺序，不证明第一名具有统计显著性。

## 常见错误

| 错误 | 为什么不公平 | 修正 |
|---|---|---|
| 每个模型重新随机划分 | 模型面对不同验证样本 | 共用一个 `cv` |
| 某模型额外获得目标相关特征 | 输入信息不同 | 先冻结特征集合 |
| 只汇报最好一折 | 隐藏波动 | 汇报全部折和均值±标准差 |
| 不与 Dummy 比较 | 缺少最低参照 | 始终保留简单基线 |
| 依据测试集排序 | 污染最终证据 | 排序只基于训练内验证 |
| 把随机 KFold 叫 scaffold 验证 | 证据边界错误 | 如实记录划分方式 |

## 完成标准

- [ ] 四个模型共用同一批训练数据和同一个 `cv`；
- [ ] 我保存了各折而不只保存均值；
- [ ] 我能解释负误差评分的符号；
- [ ] 结果表包含均值、标准差和耗时；
- [ ] 我没有写“第一名一定最优”；
- [ ] 测试集没有进入比较。

## 自测问题

1. 公平比较是否要求所有模型使用完全相同的预处理算法？
2. 为什么必须保留 Dummy？
3. RMSE 均值接近时，标准差能告诉我们什么？
4. 训练时间是否也是选择模型的证据？
5. “当前五折第一”与“普遍最好”有什么区别？

## 上一天 / 下一天

- 上一天：[Day 12：不偷看测试集的调参](../day12_tuning_without_test/README.md)
- 下一天：[Day 14：传统机器学习阶段报告](../day14_ml_stage_report/README.md)
