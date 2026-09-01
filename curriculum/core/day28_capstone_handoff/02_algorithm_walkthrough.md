# Day 28.2：从结果表生成模型卡

## 输入、变换、输出

| 项目 | 内容 |
|---|---|
| 输入 | 模型结果、划分协议、配置和版本 |
| 变换 | 校验必需字段、排序、提取限制 |
| 输出 | 结果摘要、模型卡和下一步计划 |

## 步骤 1：建立统一结果表

每行至少包含 model、split、rmse_mean、rmse_std、n_repeats 和 input_set。

## 步骤 2：检查可比性

只有 split、metric、input_set 和 sample_scope 一致时，才直接比较数值。

## 步骤 3：选择候选

选择只看声明的 validation 或 CV 指标。Dummy 永远保留，用于解释复杂模型是否真的学到信号。

## 步骤 4：建立模型卡

~~~python
required = {
    "task",
    "data",
    "split_protocol",
    "models",
    "metrics",
    "results",
    "limitations",
    "reproducibility",
}
~~~

## 步骤 5：写限制

限制要具体：样本范围、分布偏移、缺失、标签噪声、未评估群体、额外输入和计算成本。

## 步骤 6：下一步决策

把下一步与限制对应。例如“分组外性能未知”对应 GroupKFold 或外部数据，而不是直接增加层数。

## 验收

- 每个结果可追溯；
- 选择规则没有使用 test；
- 有基线、有波动、有版本；
- 限制不是空泛免责声明；
- 下一步对应明确证据缺口。
