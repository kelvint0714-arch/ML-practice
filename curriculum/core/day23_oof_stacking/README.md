# Day 23：用 OOF 预测实现无泄漏 Stacking

> 状态：待学习、待运行。代码是流程骨架，分数必须实际运行后填写。

## 本日完整学习包

1. [概念精讲](01_concepts.md)
2. [算法走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 是课程附带的 OOF 演示。保存的教程输出不代表学习者已完成实验，个人证据应另存到 `learning_outputs/`。

## 今天为什么学

昨天已经知道，第二层模型不能直接学习基础模型的训练内预测。
今天使用 `StackingRegressor`，
让 scikit-learn 按 ESOL 的 Bemis–Murcko scaffold groups
在训练数据内部生成折外预测。

目标不是追求最复杂结构，
而是做出一个可以和单模型公平比较的最小 stacking 基线。

## 前置条件

- 完成 [Day 22](../day22_hybrid_risk/README.md)；
- 能解释 Day 9 的 K 折轮换；
- 已冻结一个传统模型和一个 MLP；
- 已固定外部训练集与验证集；
- 测试集保持封存。

## 今日产出

1. 一张手工 OOF 折覆盖表；
2. 一个两基础模型的 `StackingRegressor`；
3. 一份 scaffold-aware GroupKFold 配置与组互斥检查；
4. 单模型、简单平均与 stacking 的结果表；
5. 一次泄漏检查记录和谨慎解释。

## 核心概念

### 1. OOF 预测怎样产生

假设训练集按 scaffold group 分成 5 折。
每次用其中 4 折训练基础模型，
再预测没有参与训练的第 5 折。

循环 5 次后，
每个训练样本都有一条“没见过自己、也没见过同 scaffold 样本”的预测。
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

1. 先用6个样本、3折在纸上标出每个样本在哪一轮作为 holdout。
2. 确认每个样本的二层训练特征都来自“没用它训练”的基础模型。
3. 读取 Day 22 的预注册方案。
4. 从 ESOL SMILES 生成 Bemis–Murcko scaffold；空环系统一为 `__ACYCLIC__`。
5. 选择两个互补的基础模型，不一次堆入很多候选。
6. 用 GroupKFold 创建 split 列表，并断言每折 scaffold 互斥。
7. 使用简单 Ridge 作为第二层模型。
8. 只对 `X_train, y_train` 调用 stacking 的 `fit()`。
9. 对同一个 `X_valid` 预测。
10. 计算相同的 RMSE、MAE 和 `R²`。
11. 计算两个单模型预测的简单平均。
12. 把单模型、平均和 stacking 放入同一结果表。
13. 检查代码中是否存在 `cv="prefit"`。
14. 记录耗时与结构复杂度，不因一次小幅改善就声称方法有效。

## 核心代码骨架

```python
import numpy as np
import json
import warnings
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def murcko_group(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or "__ACYCLIC__"

groups = np.array([murcko_group(s) for s in train_ids])
config_path = (
    REPO_ROOT
    / "curriculum/core/day07_integrated_baseline/reference_baseline/results/run_config.json"
)
if not config_path.exists():
    raise FileNotFoundError("缺少 Day 07 冻结配置")
rf_params = json.loads(config_path.read_text())[
    "model_params"
]["random_forest"]
tree = RandomForestRegressor(**rf_params)
mlp = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32,), alpha=0.001,
                 early_stopping=True, max_iter=300,
                 n_iter_no_change=10, random_state=42),
)
inner_splits = list(
    GroupKFold(5).split(X_train, y_train, groups=groups)
)
for fit_idx, hold_idx in inner_splits:
    assert set(groups[fit_idx]).isdisjoint(set(groups[hold_idx]))
stack = StackingRegressor(
    estimators=[("tree", tree), ("mlp", mlp)],
    final_estimator=Ridge(alpha=1.0),
    cv=inner_splits,
    passthrough=False,
)
with warnings.catch_warnings(record=True) as stack_caught:
    warnings.simplefilter("always", ConvergenceWarning)
    stack.fit(X_train, y_train)
print("stack convergence warnings:", [
    str(item.message) for item in stack_caught
    if issubclass(item.category, ConvergenceWarning)
])
stack_pred = stack.predict(X_valid)

tree.fit(X_train, y_train)
with warnings.catch_warnings(record=True) as mlp_caught:
    warnings.simplefilter("always", ConvergenceWarning)
    mlp.fit(X_train, y_train)
print("MLP convergence warnings:", [
    str(item.message) for item in mlp_caught
    if issubclass(item.category, ConvergenceWarning)
])
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
- `cv=inner_splits` 传入已按 scaffold 生成的 OOF 划分。
- `passthrough=False` 表示第二层只接收基础模型预测。
- `(a + b) / 2` 计算两个等权预测的简单平均。
- 字典的 `.items()` 同时取出方案名称和预测数组。
- `catch_warnings(record=True)` 让收敛警告成为可见记录；不要全局忽略。

## 下游任务数据需要额外处理

本日 ESOL 已使用 SMILES 衍生 scaffold group，而不是普通 KFold。
当真实数据包含同一配方的重复试样时，
scaffold 不能替代真实实验分组。
此时应根据领域团队提供的配方号、批次号或实验批次，
考虑 GroupKFold。

分组规则必须在看结果之前固定。
不要为了得到更好分数临时更换分组列。

即使使用封装类，也要能复述手工 OOF：
逐折训练、预测未参与训练的一折、写回原位置，
填满 OOF 后训练第二层，最后用全部训练数据重拟合基础模型。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 使用 `cv="prefit"` | 二层可能看到训练内预测 | 使用明确 GroupKFold splits |
| 外部验证参与内部 CV | 选择偏乐观 | 内部 CV 只切训练集 |
| 基础模型没有 Pipeline | 每折预处理不完整 | 封装插补和缩放 |
| stacking 只与弱基线比较 | 无法证明价值 | 对比最强单模型和平均 |
| 重复配方跨折 | 近重复信息泄漏 | 获取真实分组字段 |

## 完成标准

- [ ] stacking 只在外部训练集上拟合；
- [ ] 外部 train/valid 与每个内部折都通过 scaffold 互斥断言；
- [ ] 没有使用 `cv="prefit"`；
- [ ] 预处理包含在基础模型 Pipeline 中；
- [ ] 已与最强单模型和简单平均比较；
- [ ] 真实下游任务接入前没有伪造分组字段；
- [ ] 结果未被写成已完成实验。

若 stacking 更好，只报告当前协议下的改善和稳定性；
若简单平均同样好，优先保留易审计的方案；
若没有改善，也要保留这个负结果。

## 自测问题

1. OOF 中的“out”指什么？
2. 第二层为什么不能使用基础模型训练内预测？
3. 外部验证集与内部 K 折有什么区别？
4. `passthrough=False` 时第二层看到哪些特征？
5. 为什么真实下游任务可能需要 GroupKFold？

## 上一天 / 下一天

- 上一天：[Day 22：混合方案与泄漏风险](../day22_hybrid_risk/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 24：混合模型消融实验](../day24_hybrid_ablation/README.md)
