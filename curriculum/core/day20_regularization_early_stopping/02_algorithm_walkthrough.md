# Day 20 算法走读：预注册两个候选

## 步骤 0：先写候选表

| 名称 | `alpha` | 早停 | 其他设置 |
|---|---:|---:|---|
| weak_regularization | 0.00001 | 否 | 固定 |
| regularized_early_stop | 0.001 | 是 | 固定 |

运行后不临时增加“刚好更好”的第三组。若未来扩展，应建立新的实验编号。

## 步骤 1：用函数创建独立 Pipeline

```python
def make_candidate(alpha, early_stopping):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32,),
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        ),
    )
```

每个候选都创建新对象，避免重复 `fit()` 时旧状态引起误解。

这里必须披露一个实现限制：Pipeline 的插补器和缩放器先在全部 `X_train`
上拟合，之后 MLP 才切内部早停集。外部 valid/test 仍未参与，但内部早停
并非“预处理也严格隔离”的嵌套验证。严格实验应显式切 subtrain/internal-valid，
且预处理器只在 subtrain 上 `fit()`。

## 步骤 2：拟合与记录

```python
for name, model in candidates.items():
    model.fit(X_train, y_train)
    trained = model.named_steps["mlpregressor"]
```

记录：

- 训练和外部验证 RMSE；
- `n_iter_`；
- `len(loss_curve_)`；
- `best_validation_score_`（回归任务的内部最佳 `R²`；不开早停时记为“不适用”）；
- 是否捕获到 `ConvergenceWarning`。

`None` 表示该字段不适用，不应替换成 0。0 是一个真实数值，会造成错误解释。

## 步骤 3：捕获警告但不隐藏

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(X_train, y_train)

warning_messages = [str(item.message) for item in caught]
```

这样既能让 Notebook 保持整洁，也能把警告保存为证据。

## 步骤 4：检查曲线

```python
loss = trained.loss_curve_
assert len(loss) == trained.n_iter_
```

画图时横轴是迭代轮数，纵轴是训练损失。若另画
`validation_scores_`，纵轴必须写“internal validation R²”。
两条曲线都不能标成“外部验证 RMSE”。

## 步骤 5：解释而不是自动选最小值

按以下顺序阅读：

1. 候选是否收敛或早停；
2. 训练误差是否明显恶化；
3. 外部验证误差是否改善；
4. 改善是否大到值得进一步做多种子验证；
5. 冻结 Day 21 使用的设置。

若两个方案很接近，可以选择更简单、更快或更容易审计的一个，但要写明规则。

## 伪代码

```text
预先固定两个候选
对每个候选：
    新建 Pipeline
    只在外部训练集 fit
    记录警告、迭代数、训练损失和内部 R²
    披露内部预处理未严格隔离
    计算 train / external-valid 指标
并排比较
按事先规则冻结一个设置
不查看 test
```

## 结果记录模板

| 候选 | train RMSE | external-valid RMSE | n_iter | early stopped | 警告 |
|---|---:|---:|---:|---|---|
| weak | 待实际运行 | 待实际运行 | 待实际运行 | 否 | 待记录 |
| regularized + ES | 待实际运行 | 待实际运行 | 待实际运行 | 待判断 | 待记录 |
