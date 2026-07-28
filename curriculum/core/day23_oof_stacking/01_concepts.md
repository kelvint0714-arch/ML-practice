# Day 23 概念精讲：OOF Stacking

## 1. OOF 的定义

OOF（out-of-fold）预测指：某个训练样本的预测来自一个没有用该样本训练过的模型。
本日还要求产生该预测的模型没有见过相同 Bemis–Murcko scaffold 的样本。

5 折时，每轮用 4 折训练、预测剩余 1 折。5 轮后，每个训练样本恰好获得一条 OOF 预测。

## 2. OOF 矩阵的形状

若训练样本数为 `n`，基础模型数为 `m`：

```text
OOF matrix shape = (n, m)
```

Day 23 使用两个基础模型，因此第二层训练输入是 `(902, 2)`，不是 `(902, 1024)`。`passthrough=True` 时还会拼入原始特征，维度和泄漏审计都会更复杂。

## 3. StackingRegressor 的两阶段行为

调用 `stack.fit(X_train, y_train)` 时，概念上包括：

1. 内部交叉验证产生基础模型的 OOF 预测；
2. 用 OOF 矩阵训练 `final_estimator`；
3. 再用全部 `X_train, y_train` 拟合基础模型，供未来预测。

封装类减少手写错误，但不能替你决定正确的数据分组、外部验证和预处理边界。

## 4. `cv="prefit"` 为什么危险

`prefit` 表示基础模型已提前拟合。若它们看过全部训练数据，第二层将接收训练内预测，极易过拟合。零基础课程禁止使用该模式。

## 5. 内外两层验证

```text
外部 train
└── scaffold GroupKFold → OOF → 训练 meta model

外部 valid → 只评价冻结后的完整 stack
test → 继续封存
```

内部 CV 不是外部模型评价；它服务于第二层特征生成。

## 6. 为什么 stacking 不保证更好

- 基础模型误差可能高度相关；
- 第二层样本量有限；
- OOF 特征有噪声；
- 简单模型已足够；
- 复杂度带来更大方差。

因此必须同时报告两个单模型和简单平均。

## 7. ESOL scaffold 与真实业务 group

教程从 ESOL SMILES 生成 Bemis–Murcko scaffold；没有环系时统一记为
`__ACYCLIC__`，避免随机拆散这些样本。外部 train/valid 和每个内部折都要
断言 scaffold 集合互斥。

真实粘合剂重复试样、同配方和同批次也需要分组，但 ESOL scaffold
不能冒充业务 group。真实分组字段必须由化学组提供。

## 今日证据边界

若 Notebook 实际运行，只能描述固定 ESOL 协议的结果。不能把 OOF 的“无直接训练内预测泄漏”扩大成“没有任何可能泄漏”，更不能声称已完成真实粘合剂 stacking。
