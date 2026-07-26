# Day 1.5：把算法映射到真实 ESOL Notebook

现在才打开 [ESOL 参考基线 E01 Notebook](../../../experiments/esol/day01_baseline/esol_baseline.ipynb)。

今天不要逐行读完整代码，也不要修改或重新运行。只在 Notebook 中找到算法主线对应的位置。

## 只找六个核心位置

| Notebook 代码单元 | 今天只看什么 | 对应概念 |
|---:|---|---|
| 2 | `featurizer`、`load_delaney`、三个 dataset | 准备特征、标签和数据划分 |
| 4 | `X_train`、`y_train`、`X_valid`、`y_valid` | 把数据整理成算法输入 |
| 5 | `regression_metrics` | 计算 MAE、RMSE、R² |
| 6 | `models = {...}` | 创建多个尚未训练的候选模型 |
| 7 | `.fit(...)`、`.predict(...)` | 训练和预测 |
| 8 | 按 validation RMSE 排序 | 使用同一标准比较候选模型 |

请在每个位置旁边写一句：

```text
输入是什么：
这一步做什么：
输出是什么：
```

## 今天暂时跳过

| Notebook 代码单元 | 暂时跳过的原因 |
|---:|---|
| 1 | 包含路径、Git、版本和目录等复现工程代码 |
| 3 | 包含完整性断言和重叠检查 |
| 9 | 保存 CSV 和 JSON |
| 10 | 自动验收与 Markdown 结果展示 |

这些代码不是不重要，而是不应该先于算法概念学习。Day 7 会重新回到完整 Notebook，届时再学习怎样安全运行和判断是否成功。

## 在真实代码中回答

1. 哪一行创建 ECFP？
2. 哪四个变量分别保存训练输入、训练答案、验证输入、验证答案？
3. 哪一行真正让模型学习？
4. 哪一行让模型产生预测？
5. 哪个函数计算三个评价指标？
6. 模型按训练 RMSE 还是验证 RMSE 排序？
7. 为什么今天不需要理解 `json.dumps` 和 `git_output`？

看不懂具体符号时，只查询 [Python 语法字典](python_basics.md) 对应小节，不需要从头读完整文件。

下一步：[Day 1.6 核心验收](exercises.md)。
