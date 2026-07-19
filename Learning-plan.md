# 机器学习与图神经网络算法学习方案（修订版）

## 一、当前任务定位

当前课题组的材料数据仍在采集和整理阶段，因此现阶段的重点不是围绕某一篇材料论文、某一种材料体系或某一个具体预测目标开展研究，而是：

> 通过公开数据集、经典论文、官方教程和开源代码，系统熟悉决策树、随机森林、SVM、XGBoost、MLP、GCN、GraphSAGE、GAT、GIN 等算法，并掌握完整的机器学习实验流程。

现阶段的核心训练内容包括：

```text
算法原理学习
→ 查找公开数据与代码
→ 原样跑通
→ 修改关键参数
→ 加入基线模型
→ 比较实验结果
→ 分析过拟合与稳定性
→ 逐步整理公共代码
→ 后续迁移到课题数据
```

当前不应将学习范围限制在深共晶溶剂、CO₂ 捕集、分子性质预测、某一篇综述或某一个材料数据集。材料与分子任务可以作为后期练习方向，但不应成为当前唯一主线。

---

# 二、基本原则

## 2.1 先学习算法，再开发框架

正确顺序：

```text
先分别跑通算法
→ 理解不同任务的输入输出
→ 找出共同代码
→ 提取工具函数
→ 拆分模型与训练模块
→ 最后形成通用框架
```

不建议一开始就设计完整统一框架，因为：

- scikit-learn 使用 `fit()` 和 `predict()`；
- PyTorch 模型使用 `forward()`；
- PyTorch Geometric 的输入通常是 `Data` 或 `Batch`；
- 表格数据是二维矩阵；
- 图节点分类和图分类的数据结构不同；
- 分类任务与回归任务的损失函数和指标不同。

因此，不应强行把传统机器学习和深度学习完全统一成同一种模型接口。

## 2.2 先使用标准数据集

推荐使用：

### 表格分类
- Iris
- Wine
- Breast Cancer Wisconsin
- Digits

### 表格回归
- California Housing
- Diabetes

### 图节点分类
- Karate Club
- Cora
- Citeseer

### 图分类
- MUTAG
- PROTEINS

### 后期可选
- ESOL
- FreeSolv
- Lipophilicity
- 其他分子或材料性质预测数据

## 2.3 不预设模型必须达到某个指标

模型性能取决于数据质量、特征工程、数据划分、随机种子、超参数、是否存在数据泄漏、样本规模和任务难度。

正确目标是：

> 在相同数据划分和评价标准下，比较不同模型的性能、稳定性、过拟合程度和计算成本。

---

# 三、学习路线总览

## 第一阶段：传统机器学习

重点算法：

- 线性回归
- 逻辑回归
- K近邻
- 决策树
- 随机森林
- 支持向量机
- XGBoost

目标：

- 熟悉分类与回归；
- 掌握训练集、验证集和测试集；
- 掌握常见评价指标；
- 理解过拟合与欠拟合；
- 掌握基础调参方法；
- 学会建立基线模型。

## 第二阶段：基础神经网络

重点模型：

- MLP
- CNN

目标：

- 理解前向传播；
- 理解损失函数；
- 理解反向传播；
- 掌握优化器；
- 理解 Batch、Iteration 和 Epoch；
- 掌握 Dropout 和 Early Stopping；
- 学会绘制训练曲线。

## 第三阶段：图神经网络

重点模型：

- GCN
- GraphSAGE
- GAT
- GIN

目标：

- 理解节点、边和邻接关系；
- 理解节点特征矩阵；
- 理解消息传递；
- 理解邻居聚合；
- 理解节点分类与图分类；
- 理解图池化与 Readout；
- 掌握 PyTorch Geometric。

## 第四阶段：论文复现与领域迁移

完成基础算法后，再选择一个方向：

- 分子性质预测；
- 材料性质预测；
- 社交网络；
- 推荐系统；
- 交通网络；
- 知识图谱。

完成论文阅读、代码复现、数据获取、参数实验、基线比较、结果分析和后续课题迁移。

---

# 四、四周执行计划

## 第一周：决策树与随机森林

### 数据集

| 数据集 | 任务 | 主要目的 |
|---|---|---|
| Iris | 多分类 | 熟悉分类树 |
| Breast Cancer Wisconsin | 二分类 | 熟悉分类指标 |
| California Housing | 回归 | 熟悉回归树 |

### 实验内容

1. 加载并查看数据；
2. 划分训练集和测试集；
3. 训练默认决策树；
4. 修改 `max_depth`；
5. 修改 `min_samples_split`；
6. 修改 `min_samples_leaf`；
7. 比较训练集和测试集性能；
8. 可视化决策树；
9. 查看特征重要性；
10. 在相同数据上训练随机森林；
11. 修改 `n_estimators`；
12. 修改 `max_features`；
13. 比较决策树与随机森林。

### 重点问题

- 决策树如何选择切分特征？
- Gini 指数和信息增益有什么作用？
- 为什么树太深容易过拟合？
- 回归树与分类树有何区别？
- 随机森林为什么比单棵树稳定？
- Bagging 的作用是什么？

### 本周产出

```text
01_tree_models/
├── decision_tree_classification.ipynb
├── decision_tree_regression.ipynb
├── random_forest.ipynb
├── results.csv
└── notes.md
```

## 第二周：SVM、XGBoost 与公共评价模块

### 数据集

- Breast Cancer Wisconsin
- Digits
- California Housing

### SVM 实验

1. 数据标准化；
2. 训练线性 SVM；
3. 训练 RBF 核 SVM；
4. 修改参数 `C`；
5. 修改参数 `gamma`；
6. 比较不同核函数；
7. 比较标准化前后的结果。

### XGBoost 实验

1. 在 California Housing 上完成回归；
2. 在 Breast Cancer 上完成分类；
3. 修改 `n_estimators`；
4. 修改 `learning_rate`；
5. 修改 `max_depth`；
6. 修改 `subsample`；
7. 使用 Early Stopping；
8. 与随机森林比较。

### 公共模块

```text
common/
├── metrics.py
├── plots.py
└── seed.py
```

- `metrics.py`：分类与回归评价指标；
- `plots.py`：混淆矩阵、真实值—预测值图、训练曲线；
- `seed.py`：统一随机种子。

## 第三周：GCN 节点分类

### 数据集

- Karate Club
- Cora

### Karate Club

1. 使用 NetworkX 加载图；
2. 查看节点数和边数；
3. 可视化图；
4. 查看节点邻居；
5. 理解邻接矩阵；
6. 观察节点嵌入。

目标：

> 理解图神经网络如何利用邻居信息更新节点表示。

### Cora 节点分类

模型顺序：

```text
Logistic Regression
→ MLP
→ GCN
→ GraphSAGE
→ GAT
```

完成：

1. 查看节点特征矩阵；
2. 查看 `edge_index`；
3. 理解训练、验证、测试 mask；
4. 训练 MLP 基线；
5. 训练两层 GCN；
6. 修改 GCN 层数；
7. 修改隐藏维度；
8. 修改 Dropout；
9. 训练 GraphSAGE；
10. 训练 GAT；
11. 比较各模型；
12. 使用多个随机种子重复实验。

## 第四周：图分类与第一次论文复现

### 数据集

- MUTAG
- PROTEINS

### 模型

```text
GCN
→ GraphSAGE
→ GIN
```

### 实验内容

1. 查看每张图的节点数和边数；
2. 理解一个 Batch 中的多张图；
3. 使用 Global Mean Pooling；
4. 使用 Global Add Pooling；
5. 训练 GCN；
6. 训练 GraphSAGE；
7. 训练 GIN；
8. 比较不同池化方式；
9. 使用交叉验证；
10. 记录均值和标准差。

### 第一次论文复现

论文选择标准：

- 数据公开；
- 代码公开；
- README 清晰；
- 单机可以运行；
- 训练时间合理；
- 依赖不复杂；
- 有基准结果；
- 模型规模适中。

完成要求：

1. 阅读论文摘要、方法和实验部分；
2. 获取代码和数据；
3. 配置环境；
4. 原样跑通；
5. 记录原始指标；
6. 修改至少一个超参数；
7. 加入一个基线模型；
8. 分析与论文结果的差异；
9. 写简短复现报告。

---

# 五、统一评价指标

## 5.1 分类任务

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 混淆矩阵

类别不平衡时不能只看 Accuracy。

## 5.2 回归任务

- MAE
- MSE
- RMSE
- \(R^2\)

\[
MAE=rac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
\]

\[
RMSE=\sqrt{rac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}
\]

\[
R^2=1-rac{\sum_{i=1}^{N}(y_i-\hat y_i)^2}
{\sum_{i=1}^{N}(y_i-ar y)^2}
\]

## 5.3 AARD% 的使用原则

\[
AARD\%=
rac{100\%}{N}
\sum_{i=1}^{N}
\left|
rac{\hat y_i-y_i}{y_i}

ight|
\]

AARD% 适用于部分物性预测或化工任务，但不是所有任务的通用指标。当真实值接近零时，AARD% 可能异常放大。

当前阶段优先使用：

```text
MAE + RMSE + R²
```

---

# 六、每个算法的固定学习流程

## 6.1 原理学习

需要回答：

1. 该算法解决什么任务？
2. 输入是什么？
3. 输出是什么？
4. 核心计算过程是什么？
5. 损失函数是什么？
6. 主要超参数是什么？
7. 优势是什么？
8. 局限是什么？

## 6.2 资源查找

每个算法至少寻找：

- 官方文档；
- 经典论文；
- 可运行代码。

优先选择有清晰 README、环境说明、数据获取方式、训练代码、评价代码且单机可运行的项目。

## 6.3 原样跑通

记录：

- Python 版本；
- 包版本；
- 数据集；
- 数据规模；
- 模型参数；
- 训练时间；
- 原始指标；
- 报错信息；
- 报错解决方式。

## 6.4 自己重写关键部分

至少独立完成：

- 数据加载；
- 数据划分；
- 模型定义；
- 损失函数；
- 优化器；
- 训练循环；
- 验证循环；
- 测试评价；
- 模型保存；
- 结果保存。

## 6.5 控制变量实验

### 决策树
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### 随机森林
- `n_estimators`
- `max_depth`
- `max_features`

### SVM
- `kernel`
- `C`
- `gamma`

### XGBoost
- `learning_rate`
- `n_estimators`
- `max_depth`
- `subsample`

### MLP
- 隐藏层数量
- 隐藏维度
- 学习率
- Dropout
- Batch Size

### GCN
- 层数
- 隐藏维度
- Dropout
- 学习率

### GAT
- 注意力头数
- 隐藏维度
- Dropout
- 层数

## 6.6 基线模型

### 表格分类

```text
Logistic Regression
Decision Tree
Random Forest
SVM
XGBoost
```

### 表格回归

```text
Linear Regression
Decision Tree
Random Forest
XGBoost
```

### Cora 节点分类

```text
Logistic Regression
MLP
GCN
GraphSAGE
GAT
```

### 图分类

```text
图统计特征 + Random Forest
GCN
GraphSAGE
GIN
```

## 6.7 实验总结

每个实验至少回答：

1. 哪个模型最好？
2. 是否存在过拟合？
3. 参数变化有什么影响？
4. 模型提升是否稳定？
5. 不同随机种子的结果差异多大？
6. 计算成本如何？
7. 模型为什么可能表现更好？
8. 结果是否符合预期？
9. 后续如何改进？
10. 将来如何迁移到课题数据？

---

# 七、工程化重构路线

## 第一级：Notebook 跑通

```text
加载数据
→ 建立模型
→ 训练
→ 评价
```

## 第二级：提取公共工具函数

```text
common/
├── metrics.py
├── plots.py
└── seed.py
```

## 第三级：拆分模块

```text
data/
models/
trainers/
evaluation/
```

## 第四级：引入配置文件

```text
configs/
├── decision_tree.yaml
├── xgboost.yaml
├── gcn_cora.yaml
└── gin_mutag.yaml
```

## 第五级：统一实验入口

```bash
python main.py --config configs/gcn_cora.yaml
```

统一入口应在多个实验已经跑通后再实现。

---

# 八、推荐目录结构

## 8.1 当前阶段

```text
ml-algorithm-practice/
├── 01_decision_tree/
├── 02_random_forest/
├── 03_svm/
├── 04_xgboost/
├── 05_mlp/
├── 06_gcn/
├── 07_graphsage_gat/
├── 08_graph_classification/
├── common/
│   ├── metrics.py
│   ├── plots.py
│   └── seed.py
├── results/
└── README.md
```

## 8.2 后期框架

```text
algorithm-framework/
├── configs/
├── data/
│   ├── tabular_dataset.py
│   ├── graph_dataset.py
│   └── molecular_dataset.py
├── models/
│   ├── tree_models.py
│   ├── mlp.py
│   └── gnn_models.py
├── trainers/
│   ├── sklearn_trainer.py
│   └── torch_trainer.py
├── evaluation/
│   ├── classification_metrics.py
│   ├── regression_metrics.py
│   └── visualization.py
├── experiments/
├── results/
└── main.py
```

传统机器学习和深度学习建议分别使用 `SklearnTrainer` 与 `TorchTrainer`。

---

# 九、实验记录模板

## 9.1 基本信息

| 项目 | 内容 |
|---|---|
| 实验编号 |  |
| 算法名称 |  |
| 任务类型 | 分类 / 回归 / 节点分类 / 图分类 |
| 数据集 |  |
| 数据规模 |  |
| 特征维度 |  |
| 代码来源 |  |
| 论文来源 |  |
| 运行环境 |  |
| 随机种子 |  |

## 9.2 模型参数

| 参数 | 设置 |
|---|---|
| 学习率 |  |
| Batch Size |  |
| Epoch |  |
| 隐藏维度 |  |
| 层数 |  |
| Dropout |  |
| 优化器 |  |
| 其他参数 |  |

## 9.3 实验结果

| 模型 | Train | Validation | Test | 训练时间 | 备注 |
|---|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |
| Model 1 |  |  |  |  |  |
| Model 2 |  |  |  |  |  |

## 9.4 实验分析

- 最优模型：
- 是否过拟合：
- 关键参数影响：
- 多随机种子稳定性：
- 运行中出现的问题：
- 与参考结果的差异：
- 可能原因：
- 后续改进方向：
- 对未来课题可迁移的部分：

---

# 十、当前执行顺序

```text
1. Iris：决策树分类
2. California Housing：决策树回归
3. 在相同数据上加入随机森林
4. Breast Cancer：SVM 分类
5. California Housing：XGBoost 回归
6. Fashion-MNIST：MLP
7. Fashion-MNIST 或 CIFAR-10：CNN
8. Karate Club：理解图结构
9. Cora：MLP 与 GCN
10. Cora：GCN、GraphSAGE 与 GAT
11. MUTAG：GCN、GraphSAGE 与 GIN
12. 选择一篇难度适中的论文进行完整复现
13. 等课题数据完善后迁移现有流程
```

---

# 十一、阶段验收标准

一个算法只有同时满足以下条件，才算完成学习：

- 能解释算法解决的问题；
- 能解释输入与输出；
- 能说明核心计算过程；
- 能独立运行代码；
- 能修改主要超参数；
- 能建立基线模型；
- 能输出规范评价指标；
- 能判断是否过拟合；
- 能比较多个模型；
- 能分析实验结果；
- 能保存模型和结果；
- 能总结算法优势与局限；
- 能说明未来如何迁移到课题数据。

---

# 十二、暂缓内容

以下内容保留为长期方向，但当前不作为四周主线：

- RDKit 分子描述符；
- Morgan 指纹；
- DES 数据集；
- CO₂ 捕集预测；
- CatBoost 材料模型；
- SHAP 材料特征解释；
- MPNN 分子图回归；
- GNN 与 CatBoost 混合模型；
- Stacking；
- Cascading；
- 量子化学描述符；
- COSMO-RS 特征；
- 完整 YAML 配置系统；
- 大型统一材料机器学习框架。

---

# 十三、向老师汇报时的概括

> 目前材料数据仍在采集阶段，因此我准备先不限定具体材料体系，而是利用公开标准数据集、经典论文和开源代码，系统熟悉决策树、随机森林、SVM、XGBoost、MLP，以及 GCN、GraphSAGE、GAT、GIN 等算法。每个算法都会完成原理学习、代码运行、参数实验、基线比较和结果总结。代码框架也会在实验跑通后逐步重构，而不是一开始就进行过度抽象。后续等课题数据完善后，再把已经建立的数据处理、训练和评价流程迁移到实际课题中。

---

# 十四、最终原则

当前阶段的主线是：

\[
oxed{
	ext{标准数据集}

ightarrow
	ext{单个算法跑通}

ightarrow
	ext{参数实验}

ightarrow
	ext{模型比较}

ightarrow
	ext{逐步重构}
}
\]

而不是：

\[
	ext{先搭大型材料框架}

ightarrow
	ext{再学习算法}
\]

框架应从实际实验需求中逐步形成，而不是在尚未理解各类任务之前进行过度抽象。
