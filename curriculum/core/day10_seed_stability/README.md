# Day 10：多随机种子稳定性

## 今天为什么学

随机森林的抽样和随机特征选择会受到随机种子影响。

如果只运行一次，我们不知道当前结果是否恰好来自一个幸运种子。

今天学习把同一个实验重复多次，并用均值和标准差描述模型自身的随机波动。

## 前置条件

- 已完成 [Day 9 K 折交叉验证](../day09_cross_validation/README.md)；
- 能解释随机森林中的随机性来源；
- 会把多次结果追加到列表；
- 知道均值描述中心，标准差描述波动；
- 已固定 train 与 validation。

## 今日产出

今天应完成：

1. 一组看结果前固定的随机种子；
2. 每个种子对应的验证 MAE、RMSE、R²；
3. 一张均值、标准差、最小值、最大值汇总表；
4. 同一种子重复运行的一致性检查；
5. 对稳定性实验能力边界的说明。

本日不改变数据划分，因此不能把结果称为“划分稳定性”。

## 核心概念

### 1. 随机种子

随机种子让伪随机过程从指定状态开始。

相同环境、相同代码、相同数据和相同种子通常应得到一致结果。

它不能保证跨所有软件版本和硬件逐位相同。

### 2. 重复实验

重复实验时，除了随机种子，其他条件都保持一致：

- 数据；
- 特征；
- 模型参数；
- 指标；
- 软件环境。

### 3. 均值

均值用于概括多次运行的中心水平。

单独报告均值会隐藏波动，因此还要报告标准差。

### 4. 标准差

标准差越大，说明不同随机种子产生的结果差异越明显。

但它不能说明数据划分变化、实验测量误差或领域迁移误差。

## 分步骤任务

### 第一步：提前固定种子

例如：

```python
[0, 1, 2, 3, 4, 42, 100, 2026]
```

不要运行后删除成绩不好的种子。

### 第二步：固定模型参数

除 `random_state` 外，其他设置全部相同。

### 第三步：逐种子训练

每次记录验证 MAE、RMSE、R²。

如果需要训练指标，也应明确标记 split。

### 第四步：生成汇总统计

至少包括 mean、std、min 和 max。

### 第五步：解释边界

说明今天没有改变 scaffold 划分，也没有评估 test。

## 核心代码骨架

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

seed_values = [0, 1, 2, 3, 4, 42, 100, 2026]
records = []

for seed in seed_values:
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    valid_prediction = model.predict(X_valid)
    scores = regression_metrics(y_valid, valid_prediction)

    records.append({
        "seed": seed,
        "valid_mae": scores["mae"],
        "valid_rmse": scores["rmse"],
        "valid_r2": scores["r2"],
    })

seed_results = pd.DataFrame(records)

summary = seed_results[
    ["valid_mae", "valid_rmse", "valid_r2"]
].agg(["mean", "std", "min", "max"])

print(seed_results)
print(summary)
```

今天新增语法：

- `.agg(["mean", "std", "min", "max"])`：一次计算多种汇总统计；
- 双层方括号选择多列，并保持 DataFrame 类型；
- 每一行是一次运行，汇总表不能替代原始运行表。

## 常见错误

- 每个种子同时改变模型其他参数；
- 看完结果后删除较差种子；
- 只报告最好一次；
- 把标准差写成实验测量误差；
- 把固定 split 多种子称为数据划分稳定性；
- 不保存原始逐种子结果；
- 使用 test 计算每个种子成绩；
- 认为固定种子可以解决所有复现问题；
- 根据 ESOL 波动推断真实粘合剂实验误差。

## 完成标准

- 种子列表在运行前固定；
- 除随机种子外其他实验条件一致；
- 原始表保留每一次运行；
- 汇总表包含 mean、std、min、max；
- 能解释标准差的含义；
- 能区分模型随机性和划分随机性；
- 没有挑选性删除结果；
- 没有使用 test；
- 能用“均值 ± 标准差”写出规范描述；
- 不声称后续 Day 任务已经完成。

## 自测问题

1. 随机森林的随机性来自哪些步骤？
2. 相同种子是否解决所有复现问题？
3. 为什么不能只报告最好种子？
4. 均值和标准差分别表达什么？
5. 最小值和最大值有什么检查作用？
6. 今天改变了数据划分吗？
7. 固定 split 多种子结果能否代表实验测量误差？
8. 重复实验时哪些条件必须保持一致？

## 导航

- 上一天：[Day 9 K 折交叉验证](../day09_cross_validation/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 11 Pipeline 与数据泄漏](../day11_pipeline_leakage/README.md)
