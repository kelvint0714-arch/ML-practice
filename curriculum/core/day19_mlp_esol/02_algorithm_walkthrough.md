# Day 19 算法走读：第一条可复现 MLP Pipeline

## 运行前声明

固定数据版本、scaffold 划分、ECFP 1024 维、随机种子 42 和指标定义。今天不查看测试标签，也不因验证结果临时更换划分。

## 步骤 1：加载并检查数组

**输入：** DeepChem 的 train/valid 数据对象。

**动作：** 转为 NumPy，并把标签整理为一维。

**输出：** `X_train, y_train, X_valid, y_valid`。

```python
X_train = np.asarray(train_dataset.X)
y_train = np.asarray(train_dataset.y).reshape(-1)
X_valid = np.asarray(valid_dataset.X)
y_valid = np.asarray(valid_dataset.y).reshape(-1)

assert X_train.shape == (902, 1024)
assert y_train.shape == (902,)
```

`reshape(-1)` 表示让 NumPy 自动推断长度并整理成一维。它不会改变标签数值。

## 步骤 2：建立 Pipeline

```python
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(32,),
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=False,
        random_state=42,
    ),
)
```

即使当前 ECFP 没有缺失值，保留插补器也能固定未来输入规则。所有会“学习数据”的预处理都封装进 Pipeline。

## 步骤 3：只拟合训练数据

```python
model.fit(X_train, y_train)
```

这一行内部依次执行：

1. 用 `X_train` 学习每列中位数；
2. 用插补后的 `X_train` 学习均值和标准差；
3. 用处理后的训练特征学习 MLP 权重。

验证集只允许经过已经学好的 `transform()` 和 `predict()`。

## 步骤 4：统一计算指标

```python
def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
```

对训练集和验证集调用同一个函数，避免不同模型使用不同公式。

## 步骤 5：记录诊断信息

```python
trained_mlp = model.named_steps["mlpregressor"]
print(trained_mlp.n_iter_)
print(trained_mlp.loss_curve_[-5:])
```

检查：

- `n_iter_` 是否达到上限；
- 损失末尾是否仍快速下降；
- 是否出现 `ConvergenceWarning`；
- 训练与验证指标差距是否过大。

警告不是自动失败，也不是可以忽略的装饰。应把它和参数、损失走势一起记录。

## 步骤 6：在正确位置停止

今天保存 MLP 的 train/valid 指标、`n_iter_`、损失末端和警告后停止。
不要临时加入基线并宣布排名。Day 21 会读取 Day 07 的冻结配置，
用同一样本、标签、划分、指标和搜索预算完成正式比较。

## 伪代码

```text
固定协议
加载 ESOL train/valid
检查形状、缺失和标签空间
创建 imputer → scaler → MLP Pipeline
只在 train 上 fit
分别预测 train 和 valid
计算相同指标
记录 n_iter、损失和警告
注明正式基线比较留到 Day 21
写出结论边界
```

## 结果解释模板

```text
在固定的 ESOL scaffold 划分上，MLP 的验证 RMSE 为【实际运行值】。
训练 RMSE 与验证 RMSE 的差为【实际运行值】，表现出【谨慎判断】。
该结果只用于公开 logS 任务的流程学习；尚不能支持粘合剂性能结论。
```

不要在运行前填写数值，也不要从其他人的截图抄成自己的结果。
