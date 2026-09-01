# Day 7.3：完整 ESOL 基线练习

除非题目明确允许读取已有产物，否则先自己定位。做完再看 [参考答案](04_reference_answers.md)。

## A. 代码分类

把下列动作分类为“算法核心”“数据准备”“工程支持”或“结果解释”。有些可写两个分类，但要说明主要职责。

1. `model.fit(X_train, y_train)`
2. 查找 `.git` 所在的仓库根目录
3. `CircularFingerprint(size=1024, radius=2)`
4. `model.predict(X_valid)`
5. 记录 Python 和 sklearn 版本
6. 检查 ID 集合交集
7. 计算 RMSE
8. 保存 `run_config.json`
9. 按 validation RMSE 排序
10. 断言结果中没有 test 行

## B. 五个核心动作定位

不看 walkthrough 表格，打开真实 Notebook，写出：

| 动作 | 代码单元编号 | 关键变量或函数 |
|---|---:|---|
| 形成 X/y |  |  |
| 创建模型 |  |  |
| fit |  |  |
| predict |  |  |
| metric |  |  |

## C. 运行状态判断

判断下列情况能否称为“从空内核完整复现”，并说明原因：

1. 打开 Notebook 看见旧输出，没有运行；
2. 只运行最后三格且没有报错；
3. Restart 后 Run All，执行号连续 1–10，无 error；
4. Run All 中第 4 格 traceback，但第 10 格显示旧的通过文字；
5. 执行号为 `[1,2,4,5,...]`，结果文件存在。

## D. warning、traceback、断言

分别回答：

1. warning 是否一定意味着失败？
2. traceback 后为什么不能继续引用后方旧输出？
3. `AssertionError` 与普通打印有何区别？
4. 缺少可选深度学习后端的 warning，是否一定阻止当前 sklearn 模型？

## E. 结果契约

五个模型、两个 split。回答：

1. 长表应有几行？
2. `split` 应有哪些唯一值？
3. 为什么不应出现 test？
4. 每个模型为什么必须有相同的指标列？
5. 为什么 CSV 与 JSON 都需要？

## F. 证据边界

改写下面四句，使其符合当前证据：

1. “随机森林已经证明最适合所有分子任务。”
2. “不限深树训练 R² 高，所以它最好。”
3. “ESOL 结果说明下游任务目标可以准确预测。”
4. “test 以前看过，但现在重新运行就又变成未见数据。”

## G. 不运行训练的审计

使用 [tutorial.ipynb](tutorial.ipynb) 或自己写代码，检查：

- 真实 Notebook 有 10 个代码单元；
- 执行号连续；
- 无保存的 error output；
- CSV 有 10 行；
- CSV 只有 train/valid；
- JSON 声明的最好候选与 CSV 最小 validation RMSE 一致。

解释为什么“静态审计通过”仍不能证明你本机刚完成重跑。

## 验收清单

- [ ] 能区分算法主线与工程支持；
- [ ] 能在真实 Notebook 定位五个动作；
- [ ] 能从空内核完整执行；
- [ ] 能分辨 warning、traceback 和断言；
- [ ] 能核对 CSV/JSON 契约；
- [ ] 能解释不限深树的过拟合信号；
- [ ] 没有产生 test 预测；
- [ ] 没有把 ESOL 写成下游任务证据。

完成后再查看：[折叠参考答案](04_reference_answers.md)。
