# 论文算法学习路径

> 适用目标：只学习论文中涉及的算法，不展开化学背景。  
> 建议周期：8 周。  
> 建议投入：每天 2～3 小时，每周学习 6 天。  
> 最终目标：能够解释论文中的主要算法、写出核心公式、使用 Python 完成基础实现，并说明各算法之间的关系。

---

# 一、总体学习路线

```text
数学与机器学习基础
        ↓
线性回归与正则化
        ↓
支持向量机与核方法
        ↓
决策树与集成学习
        ↓
神经网络基础
        ↓
序列模型与自编码器
        ↓
图神经网络
        ↓
优化算法与混合模型
        ↓
特征选择、降维与模型解释
        ↓
主动学习、贝叶斯优化、PINN、VAE、GAN
```

整个过程分成八个阶段：

1. 基础知识
2. 线性模型
3. 支持向量机
4. 树模型与集成学习
5. 神经网络
6. 图神经网络
7. 优化与混合模型
8. 特征处理、解释与未来算法

---

# 二、第一阶段：基础知识

## 学习时间

3～5 天。

## 学习目标

理解所有后续算法共同使用的概念。

## 必须掌握的数学基础

### 1. 线性代数

- 向量
- 矩阵
- 矩阵乘法
- 转置
- 逆矩阵
- 秩
- 特征值
- 特征向量
- 正交
- 二次型

### 2. 微积分

- 导数
- 偏导数
- 梯度
- 链式法则
- 泰勒展开
- 极值问题

### 3. 概率统计

- 均值
- 方差
- 协方差
- 正态分布
- 条件概率
- 最大似然估计
- 偏差与方差
- 过拟合

## 必须掌握的机器学习概念

- 监督学习
- 回归
- 分类
- 特征
- 标签
- 模型参数
- 超参数
- 损失函数
- 训练集
- 验证集
- 测试集
- 插值
- 外推
- 泛化能力
- 数据泄漏
- 欠拟合
- 过拟合

## 代码任务

使用 Python 完成：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

学习：

- NumPy 数组
- Pandas DataFrame
- 数据读取
- 缺失值检查
- 列选择
- 简单绘图

## 验收标准

能够回答：

1. 参数和超参数有什么区别？
2. 训练集、验证集、测试集分别做什么？
3. 什么是过拟合？
4. 梯度是什么？
5. 为什么要设置随机种子？

---

# 三、第二阶段：线性模型

## 学习时间

1 周。

## 学习顺序

```text
简单线性回归
    ↓
多元线性回归 MLR
    ↓
多项式回归 MPR
    ↓
Ridge
    ↓
Lasso
```

---

## 3.1 多元线性回归 MLR

### 核心公式

\[
\hat y=\beta_0+\beta_1x_1+\cdots+\beta_px_p
\]

矩阵形式：

\[
\hat y=X\beta
\]

最小二乘目标：

\[
\min_\beta \|y-X\beta\|_2^2
\]

正规方程：

\[
\hat\beta=(X^\top X)^{-1}X^\top y
\]

### 必须掌握

- 线性假设
- 回归系数含义
- 截距
- 残差
- 最小二乘法
- 正规方程
- 梯度下降
- MSE
- 多重共线性

### 代码任务

```python
from sklearn.linear_model import LinearRegression
```

完成：

- 训练模型
- 输出系数
- 输出截距
- 计算预测值
- 画真实值与预测值散点图
- 画残差图

---

## 3.2 多项式回归 MPR

### 核心思想

将原始特征扩展为：

\[
x,\quad x^2,\quad x^3,\quad x_1x_2
\]

### 必须掌握

- 多项式特征
- 交互项
- 阶数选择
- 维度爆炸
- 高阶模型过拟合
- 多项式回归仍是线性参数模型

### 代码任务

```python
from sklearn.preprocessing import PolynomialFeatures
```

比较：

- 一阶模型
- 二阶模型
- 三阶模型

观察训练误差和测试误差。

---

## 3.3 Ridge Regression

### 目标函数

\[
\min_\beta
\left[
\|y-X\beta\|_2^2+\lambda\|\beta\|_2^2
\right]
\]

### 必须掌握

- \(L_2\) 正则化
- 系数收缩
- 正则化强度
- 偏差—方差权衡
- 为什么 Ridge 通常不会将系数压到 0

### 代码任务

```python
from sklearn.linear_model import Ridge
```

测试：

```text
alpha = 0
alpha = 0.1
alpha = 1
alpha = 10
alpha = 100
```

观察系数变化。

---

## 3.4 Lasso Regression

### 目标函数

\[
\min_\beta
\left[
\|y-X\beta\|_2^2+\lambda\|\beta\|_1
\right]
\]

### 必须掌握

- \(L_1\) 正则化
- 稀疏解
- 自动特征选择
- Ridge 与 Lasso 的差异

### 代码任务

```python
from sklearn.linear_model import Lasso
```

观察不同 `alpha` 下有多少系数变为 0。

---

## 本阶段验收标准

能够清楚解释：

1. MLR 为什么是线性模型？
2. 多项式回归为什么可以拟合非线性？
3. Ridge 和 Lasso 的正则项有什么区别？
4. 为什么标准化会影响正则化模型？
5. 为什么高阶多项式容易过拟合？

---

# 四、第三阶段：支持向量机与核方法

## 学习时间

1 周。

## 学习顺序

```text
最大间隔分类
    ↓
软间隔 SVM
    ↓
核技巧
    ↓
SVR
    ↓
LSSVM
```

---

## 4.1 SVM 基础

### 必须掌握

- 超平面
- 函数间隔
- 几何间隔
- 最大间隔
- 支持向量
- 硬间隔
- 软间隔
- 松弛变量
- 惩罚参数 \(C\)

### 不要求

暂时不需要完整推导 KKT 条件，但要理解：

- 原始问题
- 对偶问题
- 为什么最终预测只依赖支持向量

---

## 4.2 核技巧

### 必须掌握

核函数的意义：

\[
K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)
\]

重点学习：

- 线性核
- 多项式核
- RBF 核
- Sigmoid 核

RBF 核：

\[
K(x_i,x_j)=
\exp\left(-\gamma\|x_i-x_j\|^2\right)
\]

理解：

- \(\gamma\) 很大时模型更局部
- \(\gamma\) 很小时模型更平滑

---

## 4.3 SVR

### 核心概念

\[
|y_i-f(x_i)|\leq \varepsilon
\]

误差落在 \(\varepsilon\)-tube 内时不惩罚。

### 必须掌握

- \(\varepsilon\)-不敏感损失
- 参数 \(C\)
- 参数 \(\varepsilon\)
- 参数 \(\gamma\)
- 支持向量
- 核回归
- 特征标准化

### 代码任务

```python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

比较：

- Linear SVR
- Polynomial SVR
- RBF SVR

---

## 4.4 LSSVM

### 必须掌握

- 平方损失
- 等式约束
- 线性方程求解
- 与普通 SVM/SVR 的区别
- 对异常值更敏感

### 学习要求

能够说清：

> LSSVM 用平方误差和等式约束简化了优化问题，但失去了普通 SVM 的部分稀疏性和稳健性。

---

## 本阶段验收标准

能够回答：

1. 什么是支持向量？
2. 核技巧解决了什么问题？
3. \(C\)、\(\varepsilon\)、\(\gamma\) 分别控制什么？
4. 为什么 SVR 前通常要标准化？
5. LSSVM 与 SVR 的主要区别是什么？

---

# 五、第四阶段：决策树与集成学习

## 学习时间

2 周。

## 学习顺序

```text
决策树
   ↓
Bagging
   ↓
Random Forest
   ↓
Boosting
   ↓
GBDT
   ↓
XGBoost
   ↓
LightGBM
   ↓
CatBoost
```

这是整篇论文最重要的算法部分之一。

---

## 5.1 决策树 DT

### 必须掌握

- 根节点
- 内部节点
- 叶节点
- 递归二分
- 切分特征
- 切分阈值
- 回归树
- MSE 切分准则
- 树深
- 剪枝
- 过拟合

### 代码任务

```python
from sklearn.tree import DecisionTreeRegressor
```

比较：

```text
max_depth = 2
max_depth = 5
max_depth = 10
max_depth = None
```

---

## 5.2 Bagging

### 核心公式

\[
\hat y=
\frac{1}{M}\sum_{m=1}^{M}f_m(x)
\]

### 必须掌握

- Bootstrap
- 有放回抽样
- 基学习器
- 多模型平均
- 降低方差
- OOB 样本

---

## 5.3 Random Forest

### 必须掌握

- Bootstrap 样本
- 随机特征子集
- 多棵树平均
- OOB 误差
- 特征重要性
- 方差降低

### 主要超参数

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`

### 代码任务

```python
from sklearn.ensemble import RandomForestRegressor
```

---

## 5.4 Boosting

### 核心思想

串行训练多个弱学习器，后一个模型修正前一个模型的错误。

### 必须掌握

- 弱学习器
- 串行训练
- 前向加法模型
- 残差
- 学习率
- 降低偏差

---

## 5.5 GBDT

### 核心公式

\[
F_m(x)=F_{m-1}(x)+\eta h_m(x)
\]

负梯度：

\[
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]
\]

### 必须掌握

- 负梯度拟合
- 残差拟合
- 学习率
- 树数量
- 树深
- Shrinkage
- Subsample

---

## 5.6 XGBoost

### 必须掌握

- 一阶梯度
- 二阶梯度
- 二阶泰勒展开
- 正则化
- 分裂增益
- 行采样
- 列采样
- 缺失值处理
- 学习率
- 树复杂度

### 代码任务

```python
from xgboost import XGBRegressor
```

---

## 5.7 LightGBM

### 必须掌握

- Histogram
- Leaf-wise 生长
- Level-wise 生长
- GOSS
- EFB
- 大规模数据训练
- 过拟合风险

### 代码任务

```python
from lightgbm import LGBMRegressor
```

---

## 5.8 CatBoost

### 必须掌握

- 类别特征
- Target Statistics
- Ordered Target Statistics
- Ordered Boosting
- Oblivious Tree
- 防止目标泄漏
- 类别特征组合

### 代码任务

```python
from catboost import CatBoostRegressor
```

---

## 本阶段必须完成的对比表

| 模型 | 核心思想 | 并行或串行 | 主要降低 | 类别特征处理 |
|---|---|---|---|---|
| Decision Tree | 单棵递归树 | 单模型 | 无 | 需编码 |
| Random Forest | Bagging | 并行 | 方差 | 需编码 |
| GBDT | 梯度提升 | 串行 | 偏差 | 需编码 |
| XGBoost | 正则化 GBDT | 串行 | 偏差 | 需编码 |
| LightGBM | 高效 GBDT | 串行 | 偏差 | 部分支持 |
| CatBoost | 类别特征 Boosting | 串行 | 偏差 | 原生支持 |

---

## 本阶段验收标准

能够回答：

1. 决策树为什么容易过拟合？
2. Bagging 为什么能降低方差？
3. Random Forest 的两种随机性是什么？
4. Boosting 与 Bagging 有什么区别？
5. GBDT 为什么拟合负梯度？
6. XGBoost 相比 GBDT 增加了什么？
7. LightGBM 为什么训练快？
8. CatBoost 如何避免类别特征目标泄漏？

---

# 六、第五阶段：神经网络

## 学习时间

1～2 周。

## 学习顺序

```text
神经元
  ↓
MLP
  ↓
反向传播
  ↓
优化器
  ↓
RNN
  ↓
LSTM
  ↓
Autoencoder
  ↓
Self-Attention
```

---

## 6.1 神经元与 MLP

### 核心公式

\[
z=Wx+b
\]

\[
a=\sigma(z)
\]

多层网络：

\[
h^{(l)}=
\sigma\left(W^{(l)}h^{(l-1)}+b^{(l)}\right)
\]

### 必须掌握

- 权重
- 偏置
- 线性变换
- 激活函数
- 隐藏层
- 输出层
- 参数数量
- 前向传播

### 激活函数

- Sigmoid
- Tanh
- ReLU

---

## 6.2 反向传播 BPNN

### 必须掌握

- 链式法则
- 损失函数
- 梯度
- 输出层梯度
- 隐藏层梯度
- 权重更新
- 学习率

### 必须手算

至少手算一次：

- 单隐藏层网络
- 单样本
- MSE 损失
- 一次前向传播
- 一次反向传播
- 一次参数更新

---

## 6.3 优化器

建议学习：

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Momentum
- Adam

---

## 6.4 RNN

### 核心公式

\[
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
\]

### 必须掌握

- 隐藏状态
- 参数共享
- 时间展开
- BPTT
- 梯度消失
- 梯度爆炸

---

## 6.5 LSTM

### 必须掌握

- 遗忘门
- 输入门
- 输出门
- 细胞状态
- 隐藏状态

核心公式：

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
\]

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
\]

\[
C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t
\]

\[
h_t=o_t\odot\tanh(C_t)
\]

---

## 6.6 Autoencoder

### 结构

\[
x\rightarrow z\rightarrow \hat x
\]

### 必须掌握

- Encoder
- Latent Representation
- Decoder
- Reconstruction Loss
- 非线性降维
- 表示学习

---

## 6.7 Self-Attention

### 核心公式

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

### 必须掌握

- Query
- Key
- Value
- 注意力分数
- Softmax
- 加权求和
- 全局依赖

---

## 代码路线

先使用：

```python
from sklearn.neural_network import MLPRegressor
```

再学习 PyTorch：

```python
import torch
import torch.nn as nn
```

建议实现：

1. MLP 回归
2. 简单 RNN
3. LSTM
4. Autoencoder
5. Self-Attention 模块

---

## 本阶段验收标准

能够回答：

1. MLP 为什么可以拟合非线性？
2. 反向传播为什么需要链式法则？
3. ReLU 与 Sigmoid 有什么区别？
4. RNN 为什么会梯度消失？
5. LSTM 如何缓解长期依赖问题？
6. Autoencoder 学到的潜在变量是什么？
7. Attention 中 Q、K、V 分别是什么？

---

# 七、第六阶段：图神经网络

## 学习时间

1 周。

## 学习顺序

```text
图结构
  ↓
节点特征与边特征
  ↓
消息传递
  ↓
图卷积
  ↓
图级 Readout
  ↓
GNN-MG
```

---

## 7.1 图基础

### 必须掌握

- 节点
- 边
- 邻接矩阵
- 度矩阵
- 节点特征
- 边特征
- 有向图
- 无向图

---

## 7.2 消息传递神经网络

### 核心过程

消息聚合：

\[
m_v^{(k)}
=
\operatorname{AGGREGATE}
\left(
\{h_u^{(k-1)}:u\in\mathcal N(v)\}
\right)
\]

节点更新：

\[
h_v^{(k)}
=
\operatorname{UPDATE}
\left(
h_v^{(k-1)},m_v^{(k)}
\right)
\]

图级输出：

\[
h_G=
\operatorname{READOUT}
\left(
\{h_v^{(K)}\}
\right)
\]

### 必须掌握

- Message
- Aggregate
- Update
- Readout
- 节点级任务
- 边级任务
- 图级任务
- 过平滑

---

## 7.3 GNN-MG

### 学习目标

理解多个分子或组分如何组成混合物图。

需要掌握：

- 单分子图编码
- 多组分表示
- 组分嵌入融合
- 条件变量拼接
- 图级回归

---

## 代码任务

建议使用：

```python
import torch_geometric
```

完成：

1. 构造简单图
2. 实现 GCN
3. 图分类示例
4. 图回归示例
5. 多图嵌入融合

---

## 本阶段验收标准

能够回答：

1. 分子为什么适合表示成图？
2. GNN 的消息传递是什么？
3. 聚合函数为什么需要置换不变？
4. 节点嵌入如何变成图嵌入？
5. 多组分体系如何用 GNN 表示？

---

# 八、第七阶段：优化算法与混合模型

## 学习时间

1 周。

---

## 8.1 遗传算法 GA

### 学习顺序

```text
编码
 ↓
初始化种群
 ↓
适应度
 ↓
选择
 ↓
交叉
 ↓
变异
 ↓
新一代
```

### 必须掌握

- 染色体
- 基因
- 种群
- 适应度
- 选择
- 交叉
- 变异
- 精英保留

### 论文组合

```text
GA + MLR
```

---

## 8.2 粒子群优化 PSO

### 核心公式

\[
v_i^{t+1}
=
\omega v_i^t
+c_1r_1(p_i-x_i^t)
+c_2r_2(g-x_i^t)
\]

\[
x_i^{t+1}=x_i^t+v_i^{t+1}
\]

### 必须掌握

- 粒子
- 位置
- 速度
- 个体最优
- 全局最优
- 惯性权重
- 探索
- 开发

### 论文组合

```text
PSO + ANN
```

---

## 8.3 模拟退火 SA

### 核心公式

\[
P=\exp\left(-\frac{\Delta E}{T}\right)
\]

### 必须掌握

- 邻域搜索
- 温度
- 降温
- 接受较差解
- 跳出局部最优

---

## 8.4 耦合模拟退火 CSA

### 必须掌握

- 多条搜索链
- 链之间耦合
- 全局搜索
- 接受概率

### 论文组合

```text
CSA + LSSVM
```

---

## 8.5 Stacking

### 核心结构

```text
原始输入
  ├── 模型 1
  ├── 模型 2
  └── 模型 3
        ↓
   Meta Learner
        ↓
      最终预测
```

### 必须掌握

- Base Learner
- Meta Learner
- Out-of-fold Prediction
- 数据泄漏
- 多模型互补

---

## 8.6 Cascading

### 核心结构

```text
输入
 ↓
特征提取模型
 ↓
新的表示
 ↓
预测模型
 ↓
输出
```

### 论文组合

```text
LSTM-AE + SVM
```

---

## 8.7 Feature Fusion

### 必须掌握

- Early Fusion
- Intermediate Fusion
- Late Fusion
- 特征拼接
- 多模态表示
- 异质特征标准化

---

## 本阶段验收标准

能够回答：

1. GA 和 PSO 有什么区别？
2. SA 为什么会接受更差的解？
3. PSO 如何平衡探索和开发？
4. Stacking 为什么必须使用 OOF 预测？
5. Cascading 与 Stacking 有什么区别？
6. LSTM-AE + SVM 中各部分负责什么？

---

# 九、第八阶段：特征处理、解释与未来算法

## 学习时间

1 周。

---

## 9.1 特征选择

### Filter

学习：

- Pearson
- Spearman
- 方差过滤
- 相关性筛选

### Wrapper

学习：

- Forward Selection
- Backward Elimination
- RFE

### Embedded

学习：

- Lasso
- Tree Importance
- Boosting Importance

---

## 9.2 PCA

### 核心流程

```text
数据中心化
    ↓
协方差矩阵
    ↓
特征值与特征向量
    ↓
选取主成分
    ↓
投影
```

### 核心公式

\[
Z=XW
\]

### 必须掌握

- 协方差矩阵
- 特征值
- 特征向量
- 方差解释率
- 降维
- 信息损失

---

## 9.3 K 折交叉验证

### 必须掌握

- K-fold
- 每折训练
- 每折验证
- 指标均值
- 指标标准差
- 超参数选择
- Nested CV

---

## 9.4 Early Stopping

### 必须掌握

- 验证损失
- Patience
- 最佳轮次
- 过拟合控制

---

## 9.5 特征重要性

学习：

- Impurity-based Importance
- Split Importance
- Gain Importance
- Permutation Importance

---

## 9.6 SHAP

### 核心公式

\[
f(x)=\phi_0+\sum_{j=1}^{M}\phi_j
\]

### 必须掌握

- Shapley Value
- 边际贡献
- 局部解释
- 全局解释
- TreeSHAP
- Interaction Value

### 必须会看

- Bar Plot
- Summary Plot
- Dependence Plot
- Waterfall Plot
- Force Plot

---

## 9.7 GNNExplainer

学习：

- 节点掩码
- 边掩码
- 子图解释
- 特征掩码
- 互信息

---

## 9.8 LRP

学习：

- Relevance
- 逐层传播
- 相关性守恒
- 神经网络解释

---

## 9.9 Active Learning

### 核心流程

```text
已有数据
  ↓
训练模型
  ↓
选择最有价值样本
  ↓
获取标签
  ↓
加入训练集
  ↓
重新训练
```

学习：

- Uncertainty Sampling
- Query by Committee
- Exploration–Exploitation

---

## 9.10 Bayesian Optimization

### 核心结构

```text
目标函数
  ↓
代理模型
  ↓
采集函数
  ↓
选择下一个点
  ↓
更新代理模型
```

学习：

- Gaussian Process
- Expected Improvement
- Probability of Improvement
- Upper Confidence Bound

---

## 9.11 PINN

### 核心损失

\[
L=
L_{\text{data}}+\lambda L_{\text{physics}}
\]

学习：

- 数据损失
- 物理残差
- 自动微分
- 初始条件
- 边界条件

---

## 9.12 VAE

### 核心损失

\[
L=
L_{\text{reconstruction}}
+
D_{\mathrm{KL}}
\left(
q_\phi(z|x)\Vert p(z)
\right)
\]

学习：

- Encoder
- Decoder
- 潜变量分布
- KL 散度
- 重参数化技巧

---

## 9.13 GAN

### 核心目标

\[
\min_G\max_D
\left[
\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb E_{z\sim p(z)}
\log(1-D(G(z)))
\right]
\]

学习：

- Generator
- Discriminator
- 对抗训练
- 极小极大博弈
- 模式崩溃

---

# 十、八周执行安排

## 第 1 周：基础与线性模型

### 学习内容

- 机器学习基本概念
- MLR
- MPR
- Ridge
- Lasso
- 训练集、验证集、测试集
- MSE、RMSE、MAE、\(R^2\)

### 本周输出

- 一个线性回归 Notebook
- 一个 Ridge/Lasso 对比 Notebook
- 一张线性模型总结表

---

## 第 2 周：SVM 与 SVR

### 学习内容

- 最大间隔
- 软间隔
- 核技巧
- SVR
- LSSVM
- 标准化
- 网格搜索

### 本周输出

- Linear SVR
- Polynomial SVR
- RBF SVR
- 参数 \(C,\varepsilon,\gamma\) 对比实验

---

## 第 3 周：决策树与随机森林

### 学习内容

- 决策树
- Bagging
- Bootstrap
- Random Forest
- OOB
- 特征重要性

### 本周输出

- 单棵树与随机森林对比
- 不同树深实验
- 不同树数量实验
- 特征重要性图

---

## 第 4 周：GBDT 系列

### 学习内容

- Boosting
- GBDT
- XGBoost
- LightGBM
- CatBoost

### 本周输出

- 五种树模型对比表
- 训练时间对比
- 测试指标对比
- 超参数总结表

---

## 第 5 周：神经网络

### 学习内容

- MLP
- BPNN
- 优化器
- RNN
- LSTM
- Autoencoder
- Self-Attention

### 本周输出

- MLP 回归
- 手算一次反向传播
- 简单 LSTM
- Autoencoder
- Attention 计算示例

---

## 第 6 周：GNN

### 学习内容

- 图表示
- Message Passing
- Graph Convolution
- Readout
- GNN-MG

### 本周输出

- 简单图卷积实现
- 图级回归模型
- 一张 GNN 消息传递流程图

---

## 第 7 周：优化与混合模型

### 学习内容

- GA
- PSO
- SA
- CSA
- Stacking
- Cascading
- Feature Fusion

### 本周输出

- GA 简单实现
- PSO 简单实现
- Stacking 回归
- LSTM-AE + SVM 结构图

---

## 第 8 周：特征处理与解释

### 学习内容

- Filter
- Wrapper
- Embedded
- PCA
- Cross-validation
- SHAP
- GNNExplainer
- Active Learning
- Bayesian Optimization
- PINN
- VAE
- GAN

### 本周输出

- PCA 降维实验
- SHAP Summary Plot
- SHAP Dependence Plot
- 单样本 Waterfall Plot
- 全部算法总表

---

# 十一、每个算法统一学习模板

学习每个算法时，都按照以下模板记录。

## 1. 算法名称

中文名、英文名、缩写。

## 2. 算法解决什么问题

分类、回归、降维、优化、解释或生成。

## 3. 输入与输出

输入数据是什么，输出是什么。

## 4. 核心思想

用一段话解释算法。

## 5. 数学公式

写出核心目标函数或更新公式。

## 6. 训练流程

按步骤描述训练过程。

## 7. 主要超参数

列出参数及其作用。

## 8. 优点

说明算法适用条件。

## 9. 缺点

说明局限和风险。

## 10. 与其他算法的区别

至少比较两个相近算法。

## 11. Python 实现

使用 `scikit-learn`、`PyTorch` 或对应库。

## 12. 验收问题

为每个算法准备 3～5 个问题。

---

# 十二、学习优先级

## A 级：必须真正掌握

- MLR
- MPR
- Ridge
- Lasso
- SVM
- SVR
- LSSVM
- Decision Tree
- Random Forest
- Bagging
- Boosting
- GBDT
- XGBoost
- LightGBM
- CatBoost
- MLP
- BPNN
- LSTM
- Self-Attention
- GNN
- PCA
- Cross-validation
- SHAP

---

## B 级：掌握流程与结构

- RNN
- Autoencoder
- LSTM Autoencoder
- GNN-MG
- GA
- PSO
- SA
- CSA
- Stacking
- Cascading
- Feature Fusion
- GNNExplainer
- Bayesian Optimization
- Active Learning

---

## C 级：先理解概念

- LRP
- PINN
- VAE
- GAN

---

# 十三、最终验收标准

完成这条路线后，应当达到以下水平。

## 理论方面

能够：

- 写出主要算法的核心公式
- 解释损失函数
- 解释训练流程
- 说明主要超参数
- 比较相近算法
- 分析算法优缺点
- 判断算法适用场景

## 编程方面

能够独立使用：

- `LinearRegression`
- `Ridge`
- `Lasso`
- `SVR`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `XGBRegressor`
- `LGBMRegressor`
- `CatBoostRegressor`
- `MLPRegressor`
- PyTorch MLP
- PyTorch LSTM
- PyTorch Geometric
- `PCA`
- `KFold`
- `GridSearchCV`
- `shap`

## 表达方面

能够在 10～15 分钟内讲清：

1. 论文中算法如何分类
2. 线性模型、核方法、树模型、神经网络的区别
3. Random Forest 与 Boosting 的区别
4. XGBoost、LightGBM、CatBoost 的区别
5. LSTM、Autoencoder 与 Self-Attention 的作用
6. GNN 如何表示分子
7. GA、PSO、CSA 如何优化模型
8. SHAP 如何解释预测结果

---

# 十四、最终建议

不要同时学习所有算法。

严格按照以下主线推进：

```text
MLR
→ Ridge / Lasso
→ SVR
→ Decision Tree
→ Random Forest
→ GBDT
→ XGBoost
→ LightGBM
→ CatBoost
→ MLP
→ LSTM
→ Autoencoder
→ Self-Attention
→ GNN
→ GA / PSO / CSA
→ Stacking
→ PCA
→ SHAP
→ Active Learning / Bayesian Optimization
→ PINN / VAE / GAN
```

优先完成 A 级算法，再进入 B 级和 C 级。  
每学完一个算法，都要同时完成：

- 一页理论笔记
- 一个最小代码示例
- 一组超参数实验
- 一次算法对比
- 三个口头问题
