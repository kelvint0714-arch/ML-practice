# Day 22 参考答案

## A. 分类题

1. 特征级；
2. 预测级 stacking；
3. 残差式结合。

## B. 泄漏定位

1. valid 标签影响了选列，验证不再独立；
2. 第二层看到基础模型对已见样本的乐观预测；
3. 验证折通过近重复试样获得训练配方信息；
4. test 参与结构选择，已成为调参数据。

## C. 手工 OOF

```text
fold 1 train: [2, 3, 4, 5]
fold 2 train: [0, 1, 4, 5]
fold 3 train: [0, 1, 2, 3]
```

样本 3 由 fold 2 的基础模型预测；该模型没有用样本 2、3 训练。
由于样本 2、3 同属 group B，fold 2 的训练 group 为 A、C，
与 holdout group B 互斥。可检查：

```python
assert set(groups[fit_idx]).isdisjoint(set(groups[hold_idx]))
```

## D. 预注册示例

```text
输入：固定 ESOL ECFP 1024；目标为原始 logS
基础 A：Day 07 冻结随机森林
基础 B：插补 + 标准化 + 冻结 MLP(seed=42)
外部划分：固定 scaffold train/valid
group：由 ESOL SMILES 生成 Bemis–Murcko scaffold
内部 CV：GroupKFold(5)，仅切外部 train
第二层：Ridge(alpha=1.0)
指标：RMSE 主，MAE/R² 辅
对照：A、B、等权平均
失败：未稳定超过最强单模型或平均，或成本不合理
test：方案冻结前不预测
```

## E. 决策题

准确报告“当前一次验证上仅改善 0.005，成本约为 8 倍”。这种差异可能小于随机波动，应进行预先声明的多种子/多折比较；若不能稳定改善，优先保留简单平均。
