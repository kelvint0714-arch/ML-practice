# Day 6.4：参考答案

> 先做练习，再展开。以下玩具数字不是 ESOL 或粘合剂实验结论。

<details>
<summary>A. 一轮修正</summary>

```text
当前残差 = y_true - current_prediction = [-2, 2]
缩放修正 = 0.2 × [-1.5, 1.0] = [-0.3, 0.2]
新预测 = [2,2] + [-0.3,0.2] = [1.7,2.2]
新残差 = [0,4] - [1.7,2.2] = [-1.7,1.8]
```

第二个样本残差为正，说明仍然预测偏低。

</details>

<details>
<summary>B. Bagging 与 Boosting</summary>

| 问题 | 随机森林 | Gradient Boosting |
|---|---|---|
| 树是否相对独立 | 是 | 否，后树依赖当前模型错误 |
| 每棵树主要学习什么 | Bootstrap 数据上的原目标 | 当前剩余错误/损失下降方向 |
| 最终怎样汇总 | 平均树预测 | 初始预测加逐阶段缩放修正 |
| `n_estimators` 的作用 | 森林树数量 | 顺序提升阶段数量 |

</details>

<details>
<summary>C. 概念判断</summary>

1. 错。学习率控制每个阶段修正幅度，不是准确率。
2. 对。每一步更谨慎时，往往需要更多阶段积累足够修正。
3. 错。后面的树学习当前组合模型留下的错误，存在顺序依赖。
4. 错。模型可能逐渐拟合训练噪声，validation 可能先改善后恶化。
5. 错。若输入或划分已经泄漏，算法本身不会纠正实验协议。

</details>

<details>
<summary>D. staged prediction</summary>

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

staged_predictions = list(model.staged_predict(X_valid))
assert len(staged_predictions) == 50

rows = []
for stage, prediction in enumerate(staged_predictions, start=1):
    rows.append({
        "stage": stage,
        "valid_rmse": float(np.sqrt(mean_squared_error(
            y_valid,
            prediction,
        ))),
    })

stage_table = pd.DataFrame(rows)
print(stage_table[
    stage_table["stage"].isin([1, 10, 25, 50])
])

assert np.allclose(
    staged_predictions[-1],
    model.predict(X_valid),
)
```

</details>

<details>
<summary>E. 控制变量设计</summary>

示例：

```text
learning_rate 候选：[0.03, 0.1, 0.3]
固定：train/valid 划分、特征、目标、n_estimators、max_depth、
      loss、random_state、评价函数
记录：learning_rate、train RMSE、valid MAE、valid RMSE、valid R²
```

test 必须保留给方案冻结后的最终评价；若用它选学习率，它就参与模型开发。

</details>

<details>
<summary>F. 假想学习过程</summary>

train RMSE 从 2.0 持续降至 0.1。validation RMSE 先从 2.2 降至 1.0，随后升至 1.4。阶段 50 到 100 表现出训练继续改善、验证反而恶化的过拟合信号。

开发阶段应使用 validation 或训练内交叉验证，而不是 test 选择阶段。Day 6 只观察逐阶段行为，正式早停在后续课程处理。

</details>

<details>
<summary>G. 找错题</summary>

代码同时改变 `learning_rate` 和 `n_estimators`，结果差异无法单独归因于学习率。修复方式是固定相同 `n_estimators`，循环只改变 `learning_rate`；或明确把问题改成“比较三组预先定义的组合”，但那不再是学习率的单变量实验。

</details>

完成核对后回到 [Day 6 任务卡](README.md)，口头复述一次“顺序修错”。
