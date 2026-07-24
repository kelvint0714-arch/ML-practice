# Day 1 零基础学习入口

这套材料面向“刚开始学 Python、还不能独立看懂机器学习 Notebook”的学习者。它不会把代码能运行当成已经学会，而是要求你能用自己的话说明：代码在接收什么、产生什么、为什么这样写。

## 今天的目标

完成 Day 1 后，你不需要会自己写完整模型，但应该能够回答：

1. Notebook、代码单元和运行顺序分别是什么；
2. `X`、`y`、训练集和验证集分别是什么；
3. 一行 Python 中哪里是变量、函数、参数、返回值；
4. `fit`、`predict` 和评价指标分别做什么；
5. 为什么训练成绩很好、验证成绩很差可能是过拟合；
6. 为什么当前不能反复查看测试集。

## 阅读顺序

| 顺序 | 材料 | 阅读目的 |
|---|---|---|
| 1 | [Python 语法预备](python_basics.md) | 先认识 Notebook 中反复出现的符号和句型 |
| 2 | [Notebook 逐行讲解](notebook_line_by_line.md) | 对照原 Notebook，从第一个代码单元读到最后一个 |
| 3 | [练习与验收](exercises.md) | 不看答案复述概念，并做很小的代码观察 |
| 4 | [原始 Day 1 Notebook](../../../practice/01_load_esol/01_load_esol.ipynb) | 回到真实代码，确认自己能找到每一部分 |
| 5 | [实验结果解释](../../../practice/01_load_esol/notes.md) | 把代码输出翻译成机器学习结论 |

建议分三遍阅读：

- **第一遍只看流程**：知道每个代码单元大概负责什么，不纠结所有细节；
- **第二遍看语法**：对照逐行讲解，标记不认识的符号；
- **第三遍遮住解释**：只看原代码，尝试说出输入、动作和输出。

## 整个 Notebook 的十步结构

| 代码单元 | 作用 | 主要产物 |
|---:|---|---|
| 1 | 导入工具、固定随机种子、定位仓库、记录环境 | `REPO_ROOT`、`DATA_DIR`、`VERSIONS` |
| 2 | 构造 ECFP 特征器并加载 ESOL | `train_dataset`、`valid_dataset`、`test_dataset` |
| 3 | 检查样本数、数组形状、有限值和数据重叠 | 通过或触发 `AssertionError` |
| 4 | 把 DeepChem 数据转换成 sklearn 使用的数组 | `X_train`、`y_train`、`X_valid`、`y_valid` |
| 5 | 定义统一评价函数 | `regression_metrics` |
| 6 | 创建五个固定候选模型 | `models` 字典 |
| 7 | 循环训练、预测并记录结果 | `results` 表格 |
| 8 | 排序和整理结果 | `valid_ranked`、`summary`、`best_model_name` |
| 9 | 保存 CSV 与 JSON | 两个结果文件 |
| 10 | 自动验收并生成结论 | 断言通过及结论文本 |

## 先记住机器学习的固定骨架

不同模型的名字会变，但 scikit-learn 最基本的使用结构通常相同：

```python
model = SomeModel(parameter=value)  # 1. 创建模型
model.fit(X_train, y_train)         # 2. 用训练数据学习
prediction = model.predict(X_valid) # 3. 对验证数据预测
score = metric(y_valid, prediction) # 4. 比较真实值和预测值
```

Day 1 的主要代码只是把这个骨架扩展为五个模型，并把每次结果规范地保存下来。

## 你暂时不用背的内容

- DeepChem 和 RDKit 每个类的内部实现；
- ECFP 位向量的全部化学细节；
- Ridge、决策树和随机森林的数学推导；
- Git 命令和 JSON 序列化的高级选项；
- GNN、主动学习和贝叶斯优化。

先做到“看见代码知道它属于哪一步”，再学习模型原理。

## 关于导入时的红色或黄色提示

DeepChem 可能提示缺少 PyTorch、TensorFlow、JAX 或 PyTorch Geometric。Day 1 使用的是 ECFP 与 scikit-learn，不需要这些深度学习可选依赖。只要后续打印出仓库、数据规模和模型结果，这些提示不代表 Day 1 失败。

真正需要停止检查的是：出现 Python traceback、`AssertionError`、数据无法下载，或者后面的代码单元没有继续执行。
