# Day 25：用集成分歧做启发式不确定性

> 状态：待学习、待运行。集成标准差只是启发式分歧，不是校准后的预测区间。

## 今天为什么学

主动学习不能只给出一个预测值，
还需要某种方式判断模型对哪些候选更没有把握。

今天训练多个随机种子模型，
把它们对同一样本的预测标准差当作“模型分歧”。

这个数可以帮助排序候选，
但不能直接解释为真实误差范围或 95% 置信区间。

## 前置条件

- 完成 [Day 24](../day24_hybrid_ablation/README.md)；
- 能解释平均值和标准差；
- 知道随机种子会改变部分模型结果；
- 理解验证误差与单样本不确定性不同；
- 测试集仍然封存。

## 今日产出

1. 一个由多个种子模型组成的小型集成；
2. 每个验证样本的平均预测和预测标准差；
3. 分歧最高的若干样本编号；
4. 分歧与绝对误差的诊断图或表；
5. 一段明确的局限性说明。

## 核心概念

### 1. 集成分歧

对同一个样本，
多个模型会给出略有不同的预测。
这些预测的标准差越大，表示模型之间越不一致。

### 2. 它主要反映模型不稳定

分歧可能来自有限数据、随机抽样或模型结构。
它不完整描述实验测量噪声，
也不保证覆盖未知分布外风险。

### 3. 不等于预测区间

若要声称“95% 预测区间”，
需要专门的校准方法和覆盖率验证。
今天没有完成这些步骤。

## 分步骤任务

1. 固定模型类型和超参数。
2. 只改变预先声明的随机种子。
3. 每个模型只用训练数据拟合。
4. 对同一验证集生成预测。
5. 将预测按“模型 × 样本”堆叠。
6. 对每个样本计算平均值。
7. 对每个样本计算标准差。
8. 按标准差从高到低排序。
9. 计算验证样本的绝对误差。
10. 比较高分歧组与低分歧组的误差。
11. 记录分歧不能代表测量噪声。
12. 为 Day 26 冻结采集分数计算方法。

## 核心代码骨架

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

seeds = [11, 22, 33, 44, 55]
member_predictions = []

for seed in seeds:
    member = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=seed,
        ),
    )
    member.fit(X_train, y_train)
    member_predictions.append(member.predict(X_valid))

prediction_matrix = np.vstack(member_predictions)
mean_prediction = prediction_matrix.mean(axis=0)
disagreement = prediction_matrix.std(axis=0)

diagnosis = pd.DataFrame({
    "sample_id": valid_ids,
    "prediction": mean_prediction,
    "disagreement": disagreement,
    "absolute_error": np.abs(y_valid - mean_prediction),
}).sort_values("disagreement", ascending=False)
print(diagnosis.head(10))
```

## 只解释今天新增的语法

- `seeds` 列表保存预先声明的五个随机种子。
- `np.vstack(...)` 把多个一维预测堆成二维矩阵。
- `axis=0` 表示沿模型方向汇总每个样本。
- `.mean(axis=0)` 得到每个样本的集成平均预测。
- `.std(axis=0)` 得到每个样本的模型分歧。
- `np.abs(...)` 计算绝对值。
- `ascending=False` 让最大分歧排在最前面。

## 必须写在报告中的限制

可以写：

> 本项目暂以多个随机种子模型的预测标准差作为启发式模型分歧，
> 用于候选排序和流程研究。

不能写：

> 该标准差就是实验误差。

也不能写：

> 预测值加减两倍标准差就是可靠的 95% 区间。

## 与真实粘合剂数据的接口

真实项目至少需要：

- 可稳定读取的候选配方特征；
- 已测样本与未测候选的唯一编号；
- 实验测量误差或重复实验信息；
- 模型是否允许看到候选的全部工艺设置；
- 不可行或危险配方的化学约束。

算法分歧不能代替化学安全判断。
候选必须先通过化学组的可行性审核。

诊断时可把样本按分歧分成高、中、低三组并比较平均绝对误差。
若高分歧没有对应更大误差，应保留负结果，
并在 Day 26 把随机选择作为必要对照。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 把标准差称为置信区间 | 统计含义错误 | 称为启发式分歧 |
| 每个成员使用不同数据规则 | 分歧来源混杂 | 固定流程，只改种子 |
| 用验证标签参与分歧计算 | 采集规则泄漏 | 分歧只来自预测 |
| 忽略分布外样本 | 低分歧也可能错 | 增加适用域检查 |
| 不设随机选择对照 | 无法判断采集价值 | Day 26 同时模拟随机策略 |

## 完成标准

- [ ] 所有成员使用相同训练数据和参数；
- [ ] 唯一计划内差异是随机种子；
- [ ] 得到逐样本平均预测和标准差；
- [ ] 检查了分歧与绝对误差关系；
- [ ] 文档明确“不是校准预测区间”；
- [ ] 采集规则未使用候选标签；
- [ ] 没有将练习结果写成粘合剂结论。

## 自测问题

1. 集成标准差主要表示什么？
2. 为什么它不能直接称为 95% 预测区间？
3. `prediction_matrix` 的行和列分别代表什么？
4. 高分歧一定意味着高误差吗？
5. 为什么真实候选还要经过化学安全审核？

## 上一天 / 下一天

- 上一天：[Day 24：混合模型消融](../day24_hybrid_ablation/README.md)
- 下一天：[Day 26：主动学习池模拟](../day26_active_learning/README.md)
