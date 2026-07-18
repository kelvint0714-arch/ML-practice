# 化学与材料机器学习算法学习流程

> 目标：结合真实化学或材料数据，学习论文中与课题最相关的算法。  
> 原则：不单独安排数学预备；在具体算法中遇到不会的公式、概念或代码时再补。  
> 核心要求：每学习一个算法，都必须在真实数据集上完成一次完整实验。

---

# 一、总体路线

```text
化学数据库与公开数据集
        ↓
SMILES 与分子结构
        ↓
RDKit 描述符与分子指纹
        ↓
传统机器学习模型
        ↓
分子图与 GNN
        ↓
混合物与双组分建模
        ↓
模型评价与解释
        ↓
迁移到 DES / 离子液体 / 材料数据
```

真正需要掌握的不是单独的模型名称，而是：

```text
数据获取
→ 分子表示
→ 数据划分
→ 模型训练
→ 指标评价
→ 模型解释
→ 迁移到课题数据
```

---

# 二、第一阶段：建立完整分子机器学习流程

## 推荐数据集

优先使用：

- ESOL / Delaney：水溶解度回归
- Lipophilicity：脂溶性回归
- FreeSolv：水合自由能回归
- QM7 / QM8 / QM9：分子量子性质预测

第一套任务建议使用：

```text
ESOL：根据分子结构预测水溶解度
```

## 需要完成

1. 使用 DeepChem 加载数据集
2. 查看样本数量
3. 查看 SMILES
4. 查看预测标签
5. 查看训练集、验证集和测试集
6. 建立一个最简单的基线模型
7. 输出 RMSE、MAE 和 \(R^2\)
8. 保存预测结果
9. 画真实值—预测值散点图
10. 画残差图

## 必须理解

- 一个样本是什么
- 输入是什么
- 标签是什么
- 模型预测什么
- 数据集如何划分
- 指标如何评价模型
- 训练集表现好但测试集表现差意味着什么

## 验收标准

能够完整说清：

```text
数据集
→ 输入表示
→ 模型
→ 损失函数
→ 预测结果
→ 评价指标
```

---

# 三、第二阶段：学习分子数据表示

分子机器学习的核心不是直接把分子名称送进模型，而是先把分子转成算法可以处理的表示。

总路线：

```text
SMILES
 ├── RDKit 分子描述符 → 传统表格模型
 ├── Morgan / ECFP 指纹 → 传统机器学习
 └── 分子图 → GNN
```

---

## 1. SMILES

### 需要学习

- SMILES 是什么
- 原子如何表示
- 键如何表示
- 支链如何表示
- 环如何表示
- 芳香原子如何表示
- 离子、电荷和立体结构如何表示
- Canonical SMILES
- Isomeric SMILES

### 需要完成

- 使用 RDKit 读取 SMILES
- 检查 SMILES 是否有效
- 将 SMILES 转成分子对象
- 将分子对象画出来
- 从 PubChem 获取标准 SMILES

### 最小代码接口

```python
from rdkit import Chem

mol = Chem.MolFromSmiles(smiles)
```

---

## 2. RDKit 分子描述符

### 重点描述符

- Molecular Weight
- LogP
- TPSA
- H-bond Donor Count
- H-bond Acceptor Count
- Rotatable Bond Count
- Ring Count
- Aromatic Ring Count
- Heavy Atom Count
- Fraction Csp3
- Formal Charge
- Molar Refractivity

### 数据形式

```text
SMILES
  ↓
RDKit
  ↓
数值描述符表格
  ↓
MLR / RF / XGBoost / MLP
```

示例：

| MolWt | LogP | TPSA | HBD | HBA | RotBonds | Target |
|---:|---:|---:|---:|---:|---:|---:|

### 需要完成

- 批量读取 SMILES
- 批量生成描述符
- 处理无效分子
- 处理缺失值
- 保存为 CSV
- 查看描述符分布
- 删除常数特征
- 删除高度相关特征

---

## 3. Morgan Fingerprint / ECFP

### 需要学习

- 分子指纹是什么
- Morgan Fingerprint
- ECFP
- Radius
- Bit Vector
- Fingerprint Size
- 子结构编码
- 稀疏向量
- Tanimoto Similarity

### 数据形式

\[
x\in\{0,1\}^{n}
\]

常见设置：

```text
radius = 2
nBits = 1024 或 2048
```

### 需要完成

- 从 SMILES 生成 Morgan 指纹
- 比较 1024 位与 2048 位
- 比较 radius=2 与 radius=3
- 计算两个分子的 Tanimoto 相似度
- 用指纹训练 Random Forest
- 用指纹训练 XGBoost

### 最小代码接口

```python
from rdkit.Chem import AllChem

fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=2048
)
```

---

## 4. 分子图

### 图结构

- 原子：节点
- 化学键：边
- 原子属性：节点特征
- 化学键属性：边特征

### 节点特征可以包括

- 原子序数
- 元素种类
- 原子度
- 形式电荷
- 芳香性
- 杂化方式
- 是否在环中
- 氢原子数

### 边特征可以包括

- 单键
- 双键
- 三键
- 芳香键
- 是否共轭
- 是否在环中

### 数据流程

```text
SMILES
  ↓
原子与化学键解析
  ↓
节点特征 + 边特征
  ↓
分子图
  ↓
GNN
```

---

# 四、第三阶段：传统机器学习模型

所有传统模型优先在同一个数据集上比较，避免因为数据不同导致无法判断算法差异。

固定比较：

```text
ESOL 数据集
+ RDKit 描述符
+ Morgan 指纹
```

---

## 1. 多元线性回归 MLR

### 用途

作为最基础的线性基线。

### 需要理解

- 多个特征的线性组合
- 回归系数
- 截距
- 残差
- 线性假设
- 外推风险

### 实验

```text
RDKit 描述符 + MLR
```

### 需要记录

- 训练集指标
- 验证集指标
- 测试集指标
- 回归系数
- 是否出现异常预测

---

## 2. Ridge

### 用途

缓解多重共线性和过拟合。

### 实验

```text
RDKit 描述符 + StandardScaler + Ridge
```

### 需要比较

- 不同 `alpha`
- 系数收缩
- 测试误差
- 与 MLR 的差异

---

## 3. Lasso

### 用途

正则化与特征选择。

### 实验

```text
RDKit 描述符 + StandardScaler + Lasso
```

### 需要比较

- 不同 `alpha`
- 非零系数数量
- 被删除的特征
- 与 Ridge 的差异

---

## 4. Decision Tree

### 用途

学习树模型的基础结构。

### 需要理解

- 节点
- 特征切分
- 切分阈值
- 叶节点
- 树深
- 过拟合
- 剪枝

### 实验

```text
RDKit 描述符 + Decision Tree
```

比较：

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### 必须观察

- 训练集是否接近完美
- 测试集是否明显变差
- 树深对过拟合的影响

---

## 5. Random Forest

### 用途

论文和导师明确要求重点学习。

### 需要理解

- Bootstrap
- Bagging
- 随机特征子集
- 多棵树平均
- OOB Error
- 特征重要性

### 两组实验

```text
RDKit 描述符 + Random Forest
Morgan 指纹 + Random Forest
```

### 主要参数

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`

### 必须比较

- 单棵树与随机森林
- 描述符与指纹
- 训练误差与测试误差
- 内置特征重要性与置换重要性

---

## 6. GBDT

### 用途

学习 Boosting 的基本思想。

### 需要理解

- 串行训练
- 弱学习器
- 残差
- 负梯度
- 学习率
- 树数量

### 实验

```text
RDKit 描述符 + GBDT
Morgan 指纹 + GBDT
```

### 主要参数

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

---

## 7. XGBoost

### 用途

材料和分子表格数据中的重要模型。

### 需要理解

- GBDT
- 一阶梯度
- 二阶梯度
- 正则化
- 行采样
- 列采样
- 分裂增益
- 早停

### 两组实验

```text
RDKit 描述符 + XGBoost
Morgan 指纹 + XGBoost
```

### 主要参数

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

---

## 8. LightGBM

### 需要理解

- Histogram
- Leaf-wise
- Level-wise
- GOSS
- EFB
- 训练效率
- 小数据过拟合风险

### 实验

```text
RDKit 描述符 + LightGBM
```

### 重点比较

- 与 XGBoost 的训练速度
- 与 XGBoost 的预测性能
- 叶子数对过拟合的影响

---

## 9. CatBoost

### 需要理解

- 类别特征
- Target Statistics
- Ordered Statistics
- Ordered Boosting
- Oblivious Tree
- 目标泄漏

### 适用场景

后续 DES 或离子液体数据可能包含：

- HBA 类型
- HBD 类型
- 阳离子类型
- 阴离子类型
- 材料类别

因此 CatBoost 优先级较高。

### 实验

在普通分子数据上完成基础实验后，再在包含类别变量的数据上比较：

```text
One-Hot + XGBoost
CatBoost 原生类别特征
```

---

# 五、第四阶段：模型评价与数据划分

化学机器学习中，数据划分往往比模型本身更重要。

---

## 1. Random Split

随机把样本分成：

- 训练集
- 验证集
- 测试集

### 用途

快速建立基线。

### 局限

相似分子可能同时出现在训练集和测试集，使测试结果偏乐观。

---

## 2. Scaffold Split

按照分子骨架划分。

### 需要理解

- Bemis–Murcko Scaffold
- 骨架相同的分子放在同一集合
- 测试模型对新化学骨架的泛化能力

### 重要性

对于分子性质预测，Scaffold Split 通常比 Random Split 更严格。

### 必须比较

```text
Random Split
vs
Scaffold Split
```

---

## 3. Group Split

后续双组分或混合物任务中使用。

例如：

- 同一个 HBA 只能出现在一个集合
- 同一个 HBD 只能出现在一个集合
- 同一种 DES 配方只能出现在一个集合
- 同一种离子液体只能出现在一个集合

### 目的

防止模型只记住材料身份。

---

## 4. 评价指标

回归任务至少使用：

- \(R^2\)
- MAE
- RMSE

必要时增加：

- MSE
- AARD%
- MAPE

### 必须检查

- 训练集指标
- 验证集指标
- 测试集指标
- 多个随机种子
- 交叉验证均值
- 交叉验证标准差
- 异常预测
- 是否出现负值等物理不合理结果

---

# 六、第五阶段：图神经网络

导师明确提到 GNN，因此必须与传统模型在同一任务上比较。

总流程：

```text
SMILES
  ↓
分子图
  ↓
Message Passing
  ↓
Graph Pooling
  ↓
全连接层
  ↓
性质预测
```

---

## 1. GraphConv / GCN

### 需要理解

- 节点特征
- 邻居节点
- 消息传递
- 聚合
- 节点更新
- 图级池化
- 图级回归

### 实验

```text
ESOL + GraphConv / GCN
```

### 必须比较

```text
RDKit 描述符 + Random Forest
Morgan 指纹 + Random Forest
分子图 + GCN
```

---

## 2. MPNN

### 需要理解

- Message Function
- Update Function
- Readout Function
- 边特征
- 多轮消息传递

### 实验

```text
ESOL + MPNN
```

### 重点比较

- GCN 与 MPNN
- 是否使用边特征
- 消息传递层数
- 图池化方式

---

## 3. GAT

### 需要理解

- 图注意力
- 邻居权重
- 多头注意力
- 不同邻居贡献不同

### 实验

```text
ESOL + GAT
```

GAT 不需要优先于 GCN 和 MPNN，但应了解其基本结构。

---

## 4. GNN 训练重点

必须记录：

- 节点特征
- 边特征
- 消息传递层数
- Hidden Dimension
- Pooling 方法
- Batch Size
- Learning Rate
- Epoch
- Early Stopping
- 验证集最佳轮次

---

# 七、第六阶段：相似数据集复现

导师明确要求：

> 找类似的数据和任务跑一下算法。

因此不能只完成一个 ESOL 示例。

---

## 任务 1：ESOL

目标：

```text
分子结构 → 水溶解度
```

模型：

- MLR
- Random Forest
- XGBoost
- GCN
- MPNN

---

## 任务 2：Lipophilicity

目标：

```text
分子结构 → 脂溶性
```

模型：

- Random Forest
- XGBoost
- GCN
- MPNN

目的：

检验是否真正掌握完整流程，而不是只会复制 ESOL 代码。

---

## 任务 3：FreeSolv

目标：

```text
分子结构 → 水合自由能
```

重点：

- 小样本
- 数据划分敏感
- 模型稳定性
- 多随机种子
- 交叉验证

---

## 任务 4：QM 数据

可选：

- QM7
- QM8
- QM9

目标：

```text
分子结构 → 量子化学性质
```

重点：

- 多任务学习
- GNN
- 大规模分子数据
- 图表示

---

# 八、第七阶段：从单分子迁移到双组分与混合物

你们课题最终不是普通单分子预测，而更可能是：

```text
分子 A + 分子 B + 配比 + 实验条件 → 材料性质
```

因此需要从单分子模型迁移到双组分或混合物模型。

---

## 1. 描述符拼接

对两个组分分别生成描述符：

\[
x_A,\quad x_B
\]

再拼接条件变量：

\[
x=
[x_A,x_B,r,T,P]
\]

其中：

- \(x_A\)：组分 A 描述符
- \(x_B\)：组分 B 描述符
- \(r\)：摩尔比
- \(T\)：温度
- \(P\)：压力

可使用：

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP

---

## 2. 指纹拼接

分别生成：

\[
f_A,\quad f_B
\]

组合方式：

```text
直接拼接
按比例加权
求和
求平均
加入交互项
```

最终输入：

\[
x=[f_A,f_B,r,T,P]
\]

可用于：

- Random Forest
- XGBoost
- MLP

---

## 3. 双塔神经网络

结构：

```text
组分 A 表示 → Encoder A ┐
                         ├→ Fusion → 回归器 → 预测
组分 B 表示 → Encoder B ┘
```

输入可以是：

- 描述符
- 指纹
- 图嵌入

---

## 4. 双 GNN / Mixture GNN

结构：

```text
组分 A 分子图 → GNN A → 表示 A
组分 B 分子图 → GNN B → 表示 B
                            ↓
                        表示融合
                            ↓
                   拼接配比和实验条件
                            ↓
                         回归网络
                            ↓
                          预测值
```

需要比较：

- GNN 是否共享参数
- 表示拼接
- 表示求和
- 注意力融合
- 按摩尔比加权
- 加入条件特征的位置

---

## 5. 顺序不变性

对于某些混合物，组分顺序不应改变预测：

\[
f(A,B)=f(B,A)
\]

需要考虑：

- 特征排序
- 对称融合
- 求和或平均
- DeepSets
- 共享编码器

但对于具有明确 HBA/HBD 角色的体系，也可能保留角色顺序：

```text
HBA 编码器
HBD 编码器
```

需根据任务定义决定。

---

# 九、第八阶段：迁移到 DES、离子液体或材料数据

目标数据形式：

| HBA | HBD | Ratio | Temperature | Pressure | Target |
|---|---|---:|---:|---:|---:|

或者：

| Cation | Anion | Temperature | Pressure | Property |
|---|---|---:|---:|---:|

---

## 数据获取流程

```text
论文正文与补充材料
        ↓
提取材料名称和实验条件
        ↓
统一名称
        ↓
PubChem 查询 SMILES
        ↓
RDKit 标准化
        ↓
生成描述符、指纹和分子图
        ↓
构建机器学习数据集
```

---

## 数据清洗重点

- 同义名称
- 缩写
- 化学式与名称对应
- SMILES 标准化
- 盐与离子的表示
- 重复实验
- 单位统一
- 温度单位
- 压力单位
- 目标值单位
- 摩尔比表示
- 缺失值
- 异常值
- 数据来源记录

---

## 第一轮模型

优先训练：

1. MLR
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost
6. MLP

---

## 第二轮模型

数据量和质量允许后训练：

1. GCN
2. MPNN
3. 双 GNN
4. Mixture GNN
5. Attention Fusion

---

# 十、模型解释

---

## 1. 传统特征重要性

适用于：

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

需要学习：

- Split Importance
- Gain Importance
- Impurity Importance
- 特征重要性偏差

---

## 2. Permutation Importance

流程：

```text
训练完成的模型
   ↓
打乱某个特征
   ↓
重新计算性能
   ↓
性能下降越大，特征越重要
```

优点：

- 模型无关
- 解释直观

问题：

- 相关特征会影响结果
- 计算量较大

---

## 3. SHAP

### 必须掌握

- 基准预测
- SHAP Value
- 正贡献
- 负贡献
- 局部解释
- 全局解释
- TreeSHAP

### 必须会看

- Bar Plot
- Summary Plot
- Dependence Plot
- Waterfall Plot
- Force Plot

### 必须能够回答

- 哪些描述符最重要
- 高值使预测增大还是减小
- 是否存在阈值
- 是否存在特征交互
- 某个样本为什么得到当前预测

---

## 4. GNN 解释

可学习：

- GNNExplainer
- 节点掩码
- 边掩码
- 子图解释
- 原子贡献
- 化学键贡献

需要注意：

> 模型解释只能说明模型依赖了什么，不自动等于化学因果关系。

---

# 十一、统一实验设计

每个数据集都按同一模板执行。

## 1. 数据说明

记录：

- 数据集名称
- 数据来源
- 样本数量
- 输入字段
- 目标变量
- 单位
- 缺失值
- 重复值

## 2. 数据划分

至少比较：

- Random Split
- Scaffold Split 或 Group Split

## 3. 分子表示

至少比较：

- RDKit 描述符
- Morgan 指纹
- 分子图

## 4. 模型

至少比较：

- MLR
- Decision Tree
- Random Forest
- XGBoost
- GNN

## 5. 指标

统一报告：

- \(R^2\)
- MAE
- RMSE

## 6. 稳定性

使用：

- 多个随机种子
- K 折交叉验证
- 平均值
- 标准差

## 7. 图表

至少包含：

- 真实值—预测值图
- 残差图
- 模型指标对比图
- 特征重要性图
- SHAP Summary Plot

---

# 十二、最终项目结构

```text
materials-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_rdkit_descriptors.ipynb
│   ├── 03_morgan_fingerprints.ipynb
│   ├── 04_tree_models.ipynb
│   ├── 05_gnn_models.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_shap_analysis.ipynb
│   └── 08_mixture_models.ipynb
├── src/
│   ├── data_loader.py
│   ├── standardize_smiles.py
│   ├── descriptors.py
│   ├── fingerprints.py
│   ├── graph_data.py
│   ├── splits.py
│   ├── train_tree.py
│   ├── train_gnn.py
│   ├── evaluate.py
│   └── explain.py
├── results/
│   ├── metrics/
│   ├── figures/
│   └── models/
├── requirements.txt
└── README.md
```

---

# 十三、最终学习顺序

严格按下面顺序推进：

```text
DeepChem 数据加载
→ MoleculeNet ESOL
→ SMILES
→ PubChem
→ RDKit
→ 分子描述符
→ Morgan Fingerprint
→ MLR
→ Decision Tree
→ Random Forest
→ GBDT
→ XGBoost
→ LightGBM
→ CatBoost
→ Random Split
→ Scaffold Split
→ 分子图
→ GCN / GraphConv
→ MPNN
→ GAT
→ SHAP
→ Lipophilicity
→ FreeSolv
→ 双组分描述符拼接
→ 双组分指纹拼接
→ 双塔神经网络
→ 双 GNN / Mixture GNN
→ DES / 离子液体 / 材料数据
```

---

# 十四、优先级

## 第一优先级

必须优先完成：

- DeepChem
- MoleculeNet
- SMILES
- PubChem
- RDKit
- 分子描述符
- Morgan Fingerprint
- Decision Tree
- Random Forest
- XGBoost
- GCN
- MPNN
- Random Split
- Scaffold Split
- SHAP

---

## 第二优先级

完成第一优先级后学习：

- Ridge
- Lasso
- GBDT
- LightGBM
- CatBoost
- GAT
- Group Split
- Permutation Importance
- 双组分特征拼接
- 双 GNN
- Mixture GNN

---

## 暂缓内容

当前不优先：

- LSSVM
- PSO
- GA
- CSA
- LSTM
- LSTM-AE
- VAE
- GAN
- PINN
- Active Learning
- Bayesian Optimization
- LRP

这些内容等真实数据和具体研究任务明确后再决定是否学习。

---

# 十五、最终验收标准

完成本流程后，应能够：

1. 从公开数据库获取分子数据
2. 根据名称查询标准 SMILES
3. 使用 RDKit 生成描述符和指纹
4. 将分子表示成图
5. 使用决策树和随机森林完成回归
6. 使用 XGBoost、LightGBM 和 CatBoost 完成表格建模
7. 使用 GCN 或 MPNN 完成图级回归
8. 比较 Random Split 与 Scaffold Split
9. 比较描述符、指纹和分子图
10. 使用 SHAP 解释树模型
11. 构建双组分或混合物输入
12. 将完整流程迁移到 DES、离子液体或材料性质预测任务

最终应当能够独立完成：

```text
找到数据
→ 清洗数据
→ 标准化分子结构
→ 生成三类分子表示
→ 训练传统模型
→ 训练 GNN
→ 比较划分方法
→ 评价模型
→ 解释模型
→ 形成实验报告
```
