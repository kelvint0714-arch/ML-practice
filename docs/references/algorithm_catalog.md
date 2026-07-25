# 论文涉及算法完整学习清单

> 目标：只学习论文中涉及的算法、模型、优化方法、特征处理方法、验证方法与解释方法，不展开化学背景。

---

# 一、线性回归类

## 1.1 多元线性回归 MLR

**全称：** Multiple Linear Regression

\[
\hat y=\beta_0+\sum_{j=1}^{p}\beta_jx_j
\]

### 需要学习

- 线性回归模型
- 最小二乘法
- MSE 损失
- 矩阵表达
- 回归系数
- 正规方程
- 梯度下降求解
- 多重共线性
- 线性模型的插值与外推

**学习深度：熟练掌握。**

---

## 1.2 多元多项式回归 MPR

**全称：** Multivariate Polynomial Regression

在线性模型中加入：

\[
x_i^2,\quad x_i^3,\quad x_ix_j
\]

例如：

\[
\hat y
=
\beta_0+\beta_1x_1+\beta_2x_2
+\beta_3x_1^2+\beta_4x_1x_2
\]

### 需要学习

- 多项式特征
- 交互项
- 阶数选择
- 特征维度爆炸
- 多项式回归过拟合
- 多项式回归本质上仍是线性参数模型

**学习深度：熟练掌握。**

---

## 1.3 Ridge Regression

**中文：** 岭回归

\[
\min_{\beta}
\left[
\sum_{i=1}^{n}(y_i-\hat y_i)^2
+
\lambda\sum_{j=1}^{p}\beta_j^2
\right]
\]

### 需要学习

- \(L_2\) 正则化
- 正则化系数 \(\lambda\)
- 系数收缩
- 多重共线性
- 偏差—方差权衡
- 特征标准化

**学习深度：熟练掌握。**

---

## 1.4 Lasso Regression

\[
\min_{\beta}
\left[
\sum_{i=1}^{n}(y_i-\hat y_i)^2
+
\lambda\sum_{j=1}^{p}|\beta_j|
\right]
\]

### 需要学习

- \(L_1\) 正则化
- 稀疏解
- 自动特征选择
- 与 Ridge 的区别
- Lasso 在相关特征下的不稳定性

**学习深度：熟练掌握。**

---

# 二、支持向量机类

## 2.1 支持向量机 SVM

**全称：** Support Vector Machine

### 需要学习

- 最大间隔思想
- 支持向量
- 超平面
- 软间隔
- 松弛变量
- 惩罚参数 \(C\)
- 对偶问题
- 核技巧

**学习深度：理解原理。**

---

## 2.2 支持向量回归 SVR

**全称：** Support Vector Regression

\[
f(x)=\sum_{i=1}^{N}w_iK(x,x_i)+b
\]

### 需要学习

- \(\varepsilon\)-不敏感损失
- 支持向量
- 参数 \(C\)
- 参数 \(\varepsilon\)
- 参数 \(\gamma\)
- 核函数
- 特征标准化
- 网格搜索

### 核函数

1. 线性核
2. 多项式核
3. Sigmoid 核
4. 径向基函数核 RBF

\[
K(x_i,x_j)
=
\exp\left(-\gamma\|x_i-x_j\|^2\right)
\]

**学习深度：熟练掌握。**

---

## 2.3 最小二乘支持向量机 LSSVM

**全称：** Least-Squares Support Vector Machine

### 需要学习

- 平方误差损失
- 等式约束
- 线性方程组求解
- SVR 与 LSSVM 的区别
- 核函数
- 对异常值的敏感性

**学习深度：掌握基本原理。**

---

# 三、基础神经网络类

## 3.1 人工神经网络 ANN

**全称：** Artificial Neural Network

### 需要学习

- 神经元
- 权重和偏置
- 前向传播
- 损失函数
- 反向传播
- 梯度下降
- 激活函数
- 优化器
- 正则化
- 早停
- Batch
- Epoch

**学习深度：熟练掌握。**

---

## 3.2 多层感知机 MLP

**全称：** Multilayer Perceptron

\[
h=g(W_1x+b_1)
\]

\[
\hat y=W_2h+b_2
\]

### 需要学习

- 全连接层
- 输入层、隐藏层、输出层
- 线性变换
- 非线性激活
- 网络深度与宽度
- 回归输出层
- 参数量计算
- 万能逼近定理的基本含义

### 激活函数

- ReLU
- Tanh
- Sigmoid

**学习深度：熟练掌握。**

---

## 3.3 反向传播神经网络 BPNN

**全称：** Backpropagation Neural Network

### 需要学习

- 链式法则
- 输出层误差
- 隐藏层梯度
- 权重更新
- 学习率
- 梯度消失
- 参数初始化

**学习深度：熟练掌握。**

---

## 3.4 自注意力神经网络 SANN

**全称：** Self-Attention Neural Network

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

### 需要学习

- Query
- Key
- Value
- 缩放点积注意力
- 特征全局依赖
- 注意力权重
- 自注意力与 MLP 的区别

**学习深度：理解并能实现简单版本。**

---

# 四、序列与表示学习算法

## 4.1 循环神经网络 RNN

\[
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
\]

### 需要学习

- 隐藏状态
- 时间展开
- 参数共享
- 序列依赖
- BPTT
- 梯度消失与梯度爆炸

**学习深度：理解原理。**

---

## 4.2 长短期记忆网络 LSTM

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
\]

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
\]

\[
\tilde C_t=\tanh(W_C[h_{t-1},x_t]+b_C)
\]

\[
C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t
\]

\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
\]

\[
h_t=o_t\odot\tanh(C_t)
\]

### 需要学习

- 遗忘门
- 输入门
- 输出门
- 细胞状态
- 隐藏状态
- 长期依赖

**学习深度：掌握结构和前向传播。**

---

## 4.3 自编码器 AE

\[
x\xrightarrow{\text{Encoder}}z
\xrightarrow{\text{Decoder}}\hat x
\]

\[
\min \|x-\hat x\|^2
\]

### 需要学习

- 编码器
- 潜在变量
- 解码器
- 重构损失
- 降维
- 表示学习
- 欠完备自编码器
- 过完备自编码器

**学习深度：理解并能实现基本 AE。**

---

## 4.4 LSTM Autoencoder

### 结构

- LSTM Encoder 将输入编码成潜在表示
- LSTM Decoder 重构输入
- 潜在表示可交给 SVM 等模型预测

**学习深度：理解结构。**

---

## 4.5 LSTM-AE + SVM

\[
X
\rightarrow
\text{LSTM Autoencoder}
\rightarrow
Z
\rightarrow
\text{SVM}
\rightarrow
\hat y
\]

### 需要理解

- LSTM-AE 负责特征提取
- SVM 负责最终预测
- 属于 Cascading
- 是表示学习与核方法的结合

**学习深度：重点理解。**

---

# 五、图神经网络类

## 5.1 图神经网络 GNN

### 消息传递

\[
m_v^{(k)}
=
\operatorname{AGGREGATE}
\left(
\{h_u^{(k-1)}:u\in\mathcal N(v)\}
\right)
\]

### 节点更新

\[
h_v^{(k)}
=
\operatorname{UPDATE}
\left(
h_v^{(k-1)},m_v^{(k)}
\right)
\]

### 图级读出

\[
h_G
=
\operatorname{READOUT}
\left(
\{h_v^{(K)}:v\in G\}
\right)
\]

### 需要学习

- 图表示
- 邻接关系
- 消息传递
- 聚合函数
- 节点嵌入
- 图嵌入
- Readout
- 图级回归

**学习深度：重点掌握。**

---

## 5.2 GNN Mixture Graph，GNN-MG

### 需要学习

- 单个分子图表示
- 多组分图融合
- HBA 与 HBD 的组合表示
- 条件特征与图嵌入拼接
- 图级回归

**学习深度：理解模型架构。**

---

# 六、决策树与集成树算法

## 6.1 决策树 DT

### 需要学习

- 回归树
- 递归二分
- 特征选择
- 切分点
- 节点
- 叶节点
- MSE 切分准则
- 最大深度
- 剪枝
- 过拟合

**学习深度：熟练掌握。**

---

## 6.2 Bagging

\[
\hat y
=
\frac1M\sum_{m=1}^{M}f_m(x)
\]

### 需要学习

- Bootstrap
- 有放回抽样
- 基学习器
- 并行训练
- 降低方差
- 袋外样本 OOB

**学习深度：熟练掌握。**

---

## 6.3 随机森林 RF

### 需要学习

- Bootstrap
- 随机子特征
- 决策树基学习器
- 多树平均
- OOB 评价
- 特征重要性

### 主要超参数

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`

**学习深度：熟练掌握。**

---

## 6.4 Boosting

### 需要学习

- 串行学习
- 弱学习器
- 前向加法模型
- 残差
- 负梯度
- 降低偏差
- 学习率

**学习深度：熟练掌握。**

---

## 6.5 GBDT

\[
F_m(x)
=
F_{m-1}(x)+\eta h_m(x)
\]

\[
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
\]

### 需要学习

- 前向分步加法
- 负梯度
- 残差拟合
- 决策树弱学习器
- 学习率
- 树数量
- 过拟合控制

**学习深度：熟练掌握。**

---

## 6.6 GradientBoosting / GBoost

### 需要学习

- `GradientBoostingRegressor`
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

**学习深度：能使用和解释。**

---

## 6.7 XGBoost

\[
\mathcal L^{(t)}
=
\sum_i
l(y_i,\hat y_i^{(t-1)}+f_t(x_i))
+
\Omega(f_t)
\]

### 需要学习

- 一阶梯度
- 二阶梯度
- 泰勒展开
- 正则化目标
- 树结构复杂度
- 列采样
- 行采样
- 缺失值处理
- Shrinkage
- 分裂增益

**学习深度：熟练掌握。**

---

## 6.8 LightGBM

### 需要学习

- Histogram 算法
- Leaf-wise 生长
- Level-wise 与 Leaf-wise
- 最大叶子数
- GOSS
- EFB
- 过拟合控制

**学习深度：掌握核心机制并会使用。**

---

## 6.9 CatBoost

### 需要学习

- 类别特征
- Target Statistics
- Ordered Target Statistics
- Ordered Boosting
- 对称树 Oblivious Tree
- 类别组合特征
- 防止目标泄漏
- 小数据适应性

**学习深度：熟练掌握。**

---

# 七、优化算法

## 7.1 粒子群优化 PSO

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

### 需要学习

- 粒子
- 位置
- 速度
- 个体最优
- 全局最优
- 惯性权重
- 探索与开发
- 超参数优化

### 论文组合

\[
\text{PSO-ANN}
\]

**学习深度：掌握算法流程。**

---

## 7.2 模拟退火 SA

\[
P=\exp\left(-\frac{\Delta E}{T}\right)
\]

### 需要学习

- 当前解
- 邻域搜索
- 温度
- 降温策略
- Metropolis 接受准则
- 跳出局部最优

**学习深度：理解原理。**

---

## 7.3 耦合模拟退火 CSA

### 需要学习

- 多个候选搜索链
- 链之间的信息耦合
- 接受概率控制
- 全局搜索能力

### 论文组合

\[
\text{CSA-LSSVM}
\]

**学习深度：掌握框架。**

---

## 7.4 遗传算法 GA

### 基本流程

1. 编码
2. 初始化种群
3. 适应度计算
4. 选择
5. 交叉
6. 变异
7. 生成下一代
8. 终止判断

### 论文用途

- 特征选择
- 筛选 MLR 输入变量
- 优化模型参数

### 论文组合

\[
\text{GA + MLR}
\]

**学习深度：掌握流程并能实现简单版本。**

---

# 八、混合与集成架构

## 8.1 Stacking

\[
x
\rightarrow
\begin{cases}
f_1(x)\\
f_2(x)\\
f_3(x)
\end{cases}
\rightarrow
g(f_1(x),f_2(x),f_3(x))
\]

### 需要学习

- Base Learner
- Meta Learner
- Out-of-fold Prediction
- 防止数据泄漏
- 第一层与第二层模型
- 回归 Stacking

**学习深度：熟练掌握。**

---

## 8.2 Cascading

\[
x\rightarrow f_1(x)\rightarrow f_2(f_1(x))
\]

### 需要学习

- 串行模型
- 前一模型输出作为后一模型输入
- 误差传播
- 特征提取器与预测器组合

### 论文实例

\[
\text{LSTM-AE}+\text{SVM}
\]

**学习深度：掌握概念。**

---

## 8.3 Feature Fusion

\[
z=[z_1,z_2,\ldots,z_k]
\]

\[
\hat y=f(z)
\]

### 需要学习

- 早期融合
- 中期融合
- 后期融合
- 特征拼接
- 不同尺度标准化
- 异质表示融合

**学习深度：掌握概念。**

---

# 九、特征选择与降维算法

## 9.1 Filter Feature Selection

### 需要学习

- Pearson 相关系数
- Spearman 相关系数
- 方差过滤
- 相关特征删除
- 单变量统计检验

**学习深度：掌握。**

---

## 9.2 Wrapper Feature Selection

### 典型方法

- Forward Selection
- Backward Elimination
- Recursive Feature Elimination，RFE

**学习深度：掌握。**

---

## 9.3 Embedded Feature Selection

### 典型方法

- Lasso
- 决策树特征重要性
- Random Forest 特征重要性
- Boosting 特征重要性

**学习深度：掌握。**

---

## 9.4 主成分分析 PCA

\[
Z=XW
\]

### 需要学习

- 数据中心化
- 协方差矩阵
- 特征值
- 特征向量
- 主成分
- 方差解释率
- 投影
- 降维
- 多重共线性缓解

**学习深度：熟练掌握。**

---

# 十、模型训练与验证方法

## 10.1 Hold-out Split

### 需要学习

- 训练集
- 验证集
- 测试集
- 随机种子
- 数据泄漏

---

## 10.2 K 折交叉验证

\[
\text{CV Score}
=
\frac1K\sum_{k=1}^{K}s_k
\]

### 需要学习

- K-fold
- 每折训练与验证
- 均值和标准差
- 超参数选择
- 嵌套交叉验证

**学习深度：熟练掌握。**

---

## 10.3 Early Stopping

### 需要学习

- 验证损失
- Patience
- 最佳轮次
- 防止过拟合
- ANN、XGBoost、CatBoost 中的早停

**学习深度：掌握。**

---

# 十一、模型解释算法

## 11.1 特征重要性

### 需要学习

- Impurity-based Importance
- Split Importance
- Gain Importance
- Permutation Importance
- 特征重要性的偏差

---

## 11.2 SHAP

**全称：** SHapley Additive exPlanations

\[
f(x)=\phi_0+\sum_{j=1}^{M}\phi_j
\]

### 需要学习

- Shapley Value
- 边际贡献
- 特征联盟
- 局部解释
- 全局解释
- TreeSHAP

### 常见图

1. Summary Plot
2. Dependence Plot
3. Force Plot
4. Bar Plot
5. Waterfall Plot
6. Interaction Value

**学习深度：熟练掌握。**

---

## 11.3 GNNExplainer

### 需要学习

- 关键节点
- 关键边
- 关键子图
- 节点特征
- 子图掩码
- 特征掩码
- 互信息思想

**学习深度：理解原理。**

---

## 11.4 层级相关性传播 LRP

**全称：** Layer-wise Relevance Propagation

### 需要学习

- Relevance
- 守恒思想
- 逐层传播
- 神经网络解释

**学习深度：了解。**

---

## 11.5 Attention Mechanism Analysis

### 需要学习

- 注意力权重分析
- 特征贡献
- 原子贡献
- 组分关系
- 注意力不等于因果解释

**学习深度：理解。**

---

# 十二、论文展望中的算法

## 12.1 主动学习 Active Learning

\[
\text{训练模型}
\rightarrow
\text{选择样本}
\rightarrow
\text{实验标注}
\rightarrow
\text{更新模型}
\]

### 需要学习

- Uncertainty Sampling
- Query by Committee
- Expected Improvement
- Exploration–Exploitation
- Pool-based Active Learning

**学习深度：理解框架。**

---

## 12.2 贝叶斯优化 Bayesian Optimization

\[
x_{t+1}
=
\arg\max_x a(x\mid\mathcal D_t)
\]

### 需要学习

- 代理模型
- Gaussian Process
- Acquisition Function
- Expected Improvement
- Probability of Improvement
- Upper Confidence Bound
- 迭代搜索

**学习深度：重点理解。**

---

## 12.3 物理信息神经网络 PINN

\[
L
=
L_{\text{data}}
+
\lambda L_{\text{physics}}
\]

### 需要学习

- 数据损失
- 物理方程残差
- 自动微分
- 边界条件
- 初始条件
- 软约束

**学习深度：理解框架。**

---

## 12.4 变分自编码器 VAE

\[
L
=
L_{\mathrm{reconstruction}}
+
D_{\mathrm{KL}}
\left(
q_\phi(z|x)\Vert p(z)
\right)
\]

### 需要学习

- 概率编码器
- 潜变量分布
- 均值和方差
- 重参数化技巧
- KL 散度
- 重构损失
- 潜空间采样

**学习深度：理解原理。**

---

## 12.5 生成对抗网络 GAN

\[
\min_G\max_D
\left[
\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb E_{z\sim p(z)}
\log(1-D(G(z)))
\right]
\]

### 需要学习

- Generator
- Discriminator
- 对抗训练
- 极小极大博弈
- 模式崩溃

**学习深度：理解原理。**

---

# 十三、完整算法名单

## 预测模型

1. MLR
2. MPR
3. Ridge Regression
4. Lasso Regression
5. SVM
6. SVR
7. LSSVM
8. ANN
9. MLP
10. BPNN
11. SANN
12. RNN
13. LSTM
14. Autoencoder
15. LSTM Autoencoder
16. GNN
17. GNN-MG
18. Decision Tree
19. Random Forest
20. GBDT
21. GBoost
22. XGBoost
23. LightGBM
24. CatBoost

## 优化算法

25. PSO
26. Simulated Annealing
27. CSA
28. Genetic Algorithm
29. Bayesian Optimization

## 集成和混合架构

30. Bagging
31. Boosting
32. Stacking
33. Cascading
34. Feature Fusion
35. PSO-ANN
36. CSA-LSSVM
37. GA-MLR
38. LSTM-AE + SVM

## 特征与降维算法

39. Filter Feature Selection
40. Wrapper Feature Selection
41. Embedded Feature Selection
42. Correlation Screening
43. PCA

## 训练与验证方法

44. Hold-out Split
45. K-fold Cross-validation
46. Early Stopping

## 解释算法

47. Feature Importance
48. Permutation Importance
49. SHAP
50. TreeSHAP
51. GNNExplainer
52. Layer-wise Relevance Propagation
53. Attention Mechanism Analysis

## 未来算法

54. Active Learning
55. PINN
56. VAE
57. GAN

---

# 十四、推荐学习顺序

## 第一阶段：机器学习基础

1. MLR
2. MPR
3. Ridge
4. Lasso
5. 数据划分
6. K-fold
7. 评价指标
8. PCA

## 第二阶段：核方法

9. SVM
10. SVR
11. 核函数
12. LSSVM

## 第三阶段：树模型

13. Decision Tree
14. Bagging
15. Random Forest
16. Boosting
17. GBDT
18. XGBoost
19. LightGBM
20. CatBoost

## 第四阶段：神经网络

21. ANN
22. MLP
23. BPNN
24. RNN
25. LSTM
26. Autoencoder
27. SANN

## 第五阶段：图神经网络

28. 图论基础
29. Message Passing
30. GNN
31. GNN-MG

## 第六阶段：优化与混合模型

32. GA
33. PSO
34. SA
35. CSA
36. Stacking
37. Cascading
38. Feature Fusion
39. PSO-ANN
40. CSA-LSSVM
41. LSTM-AE + SVM

## 第七阶段：模型解释

42. 特征重要性
43. Permutation Importance
44. SHAP
45. GNNExplainer
46. LRP

## 第八阶段：未来算法

47. Bayesian Optimization
48. Active Learning
49. PINN
50. VAE
51. GAN

---

# 十五、最核心学习范围

至少真正掌握以下内容：

\[
\boxed{
\begin{aligned}
&\text{MLR, MPR, Ridge, Lasso}\\
&\text{SVM, SVR, LSSVM}\\
&\text{DT, RF, GBDT, XGBoost, LightGBM, CatBoost}\\
&\text{MLP/BPNN, SANN, LSTM, GNN}\\
&\text{SHAP}
\end{aligned}
}
\]

需要理解工作流程：

- PSO
- CSA
- GA
- Autoencoder
- Stacking
- Cascading
- PCA
- 主动学习

暂时只需认识原理：

- VAE
- GAN
- PINN
