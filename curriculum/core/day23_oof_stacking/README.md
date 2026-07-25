# Day 23：用 OOF 预测实现无泄漏 Stacking

> 状态：待学习、待运行。代码是流程骨架，分数必须实际运行后填写。

## 今天为什么学

昨天已经知道，第二层模型不能直接学习基础模型的训练内预测。
今天使用 `StackingRegressor`，
让 scikit-learn 在训练数据内部生成折外预测。

目标不是追求最复杂结构，
而是做出一个可以和单模型公平比较的最小 stacking 基线。

## 前置条件

- 完成 [Day 22](../day22_hybrid_risk/README.md)；
- 能解释 OOF 预测；
- 已冻结一个传统模型和一个 MLP；
- 已固定外部训练集与验证集；
- 测试集保持封存。

## 今日产出

1. 一个两基础模型的 `StackingRegressor`；
2. 一份内部 K 折配置；
3. 单模型、简单平均与 stacking 的结果表；
4. 一次泄漏检查记录；
5. 一段对性能和复杂度的谨慎解释。

## 核心概念

### 1. OOF 预测怎样产生

假设训练集分成 5 折。
每次用其中 4 折训练基础模型，
再预测没有参与训练的第 5 折。

循环 5 次后，
每个训练样本都有一条“没见过自己”的预测。
这些预测才适合训练第二层。

### 2. StackingRegressor 做了什么

它在内部交叉验证中为第二层生成 OOF 特征，
随后用全部训练数据重新拟合基础模型，
用于预测外部验证或未来样本。

### 3. `cv="prefit"` 的风险

零基础阶段不要使用 `cv="prefit"`。
如果预拟合基础模型看过了全部训练样本，
二层输入很可能变成训练内预测。

### 4. 外部验证仍然独立

内部 K 折只发生在外部训练集内部。
外部验证标签不能进入内部拟合、参数选择或二层训练。

## 分步骤任务

1. 读取 Day 22 的预注册方案。
2. 确认 Ridge/树模型/MLP 的预处理都封装在 Pipeline 中。
3. 选择两个互补的基础模型，不一次堆入很多候选。
4. 创建带 shuffle 和固定种子的 KFold。
5. 使用简单 Ridge 作为第二层模型。
6. 只对 `X_train, y_train` 调用 stacking 的 `fit()`。
7. 对同一个 `X_valid` 预测。
8. 计算相同的 RMSE、MAE 和 `R²`。
9. 计算两个单模型预测的简单平均。
10. 把单模型、平均和 stacking 放入同一结果表。
11. 检查代码中是否存在 `cv="prefit"`。
12. 记录耗时与结构复杂度。
13. 不因一次小幅改善就声称方法有效。

## 核心代码骨架

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tree = make_pipeline(
    SimpleImputer(strategy="median"),
    GradientBoostingRegressor(random_state=42),
)
mlp = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64,), alpha=0.001,
                 early_stopping=True, max_iter=500, random_state=42),
)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
stack = StackingRegressor(
    estimators=[("tree", tree), ("mlp", mlp)],
    final_estimator=Ridge(alpha=1.0),
    cv=inner_cv,
    passthrough=False,
)
stack.fit(X_train, y_train)
stack_pred = stack.predict(X_valid)

tree.fit(X_train, y_train)
mlp.fit(X_train, y_train)
mean_pred = (tree.predict(X_valid) + mlp.predict(X_valid)) / 2

for name, pred in {"mean": mean_pred, "stack": stack_pred}.items():
    print(name,
          root_mean_squared_error(y_valid, pred),
          mean_absolute_error(y_valid, pred),
          r2_score(y_valid, pred))
```

## 只解释今天新增的语法

- `StackingRegressor` 是回归任务的预测级组合器。
- `estimators=[("tree", tree), ...]` 给基础模型命名。
- `final_estimator` 指定第二层模型。
- `cv=inner_cv` 指定生成 OOF 特征的内部划分。
- `passthrough=False` 表示第二层只接收基础模型预测。
- `(a + b) / 2` 计算两个等权预测的简单平均。
- 字典的 `.items()` 同时取出方案名称和预测数组。

## 粘合剂数据需要额外处理

当真实数据包含同一配方的重复试样时，
普通 KFold 可能把相关样本拆到不同折。
此时应根据化学组提供的配方号、批次号或实验批次，
考虑 GroupKFold。

分组规则必须在看结果之前固定。
不要为了得到更好分数临时更换分组列。

即使使用封装类，也要能复述手工 OOF：
逐折训练、预测未参与训练的一折、写回原位置，
填满 OOF 后训练第二层，最后用全部训练数据重拟合基础模型。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 使用 `cv="prefit"` | 二层可能看到训练内预测 | 使用明确 KFold |
| 外部验证参与内部 CV | 选择偏乐观 | 内部 CV 只切训练集 |
| 基础模型没有 Pipeline | 每折预处理不完整 | 封装插补和缩放 |
| stacking 只与弱基线比较 | 无法证明价值 | 对比最强单模型和平均 |
| 重复配方跨折 | 近重复信息泄漏 | 获取真实分组字段 |

## 完成标准

- [ ] stacking 只在外部训练集上拟合；
- [ ] 内部 CV 明确固定折数和种子；
- [ ] 没有使用 `cv="prefit"`；
- [ ] 预处理包含在基础模型 Pipeline 中；
- [ ] 已与最强单模型和简单平均比较；
- [ ] 真实粘合剂接入前没有伪造分组字段；
- [ ] 结果未被写成已完成实验。

若 stacking 更好，只报告当前协议下的改善和稳定性；
若简单平均同样好，优先保留易审计的方案；
若没有改善，也要保留这个负结果。

## 自测问题

1. OOF 中的“out”指什么？
2. 第二层为什么不能使用基础模型训练内预测？
3. 外部验证集与内部 K 折有什么区别？
4. `passthrough=False` 时第二层看到哪些特征？
5. 为什么真实粘合剂可能需要 GroupKFold？

## 上一天 / 下一天

- 上一天：[Day 22：混合方案与泄漏风险](../day22_hybrid_risk/README.md)
- 下一天：[Day 24：混合模型消融实验](../day24_hybrid_ablation/README.md)
