# Unit 09：主动学习综合演练与论文复现交接

> 状态：先用人工候选池组合前八个 Unit；公开数据论文复现是下一阶段。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [项目结构与运行走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

真正的研究交付不是“Notebook 跑了”，而是问题、数据、版本、基线、逐轮日志、指标和结论边界都能由别人检查。本单元先用人工候选池演练最小研究包，再选择 P01/P02 进入论文复现仓库。因此课程 Notebook 不是公开数据基线，也不是可投稿结果。

## 前置条件

- 完成 Unit 01–08；
- 能运行多种子、多策略实验；
- 能借助 [Python 语法速查](../shared/python_patterns.md)解释函数参数、回调和 `groupby/agg`；
- 理解课程教材与个人实验的区别；
- 阅读 [统一实验协议](../shared/experiment_protocol.md)。

## 今日产出

1. 一份实验配置；
2. 固定 GP 下 Random/Greedy/UCB/EI 的公平实验矩阵；
3. 逐轮 query log；
4. 策略指标汇总和学习曲线；
5. 能处理策略并列的结构化中文报告；
6. 真实化学组 Oracle 接口字段。

## 核心概念

最小研究包：

```text
config
→ source/data version
→ one-command run
→ raw query log
→ metrics summary
→ generated figures
→ limitations and handoff
```

课程 Notebook 只在内存中生成这些对象。个人项目应保存到自己的 `experiments/active_learning/unit09_capstone/results/`。

## 分步骤任务

1. 写清优化目标、候选行含义和 Oracle。
2. 冻结数据、特征、初始点、预算和种子。
3. 选择简单代理模型与 Random 基线。
4. 运行 Greedy/UCB/EI，保存逐轮 query。
5. 检查标签权限和预算一致性。
6. 汇总 regret、best-so-far 和运行数量。
7. 用 Greedy 对照 UCB/EI，检查探索项，并保留并列或无优势的负结果。
8. 写结果能说明与不能说明什么，明确这里只是人工演练。
9. 选择 P01 或 P02 进入精确论文复现。

## 核心文档骨架

```text
实验问题：
候选与目标：
代理模型：
不确定性：
采集函数：
初始样本与预算：
种子：
指标：
主要结果：
失败/负结果：
证据边界：
接入真实实验还缺什么：
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 只保存截图 | 无法复算 | 保存长表和配置 |
| 改参数不记版本 | 结果不可追踪 | 固定 config/commit |
| 课程人工数据称材料结果 | 证据不实 | 明确教学 Oracle |
| 未完成基线就追复杂模型 | 无贡献归因 | Random + 简单模型先行 |

## 完成标准

- [ ] 一条命令或一个 Notebook 可从头重跑；
- [ ] 配置、种子、逐轮日志和指标齐全；
- [ ] 所有策略同初始集、预算和评价；
- [ ] Greedy/UCB/EI 探索消融完成，并正确报告并列或无优势结果；
- [ ] 课程结果没有被称为公开数据基线；
- [ ] 没有声称完成真实粘合剂实验；
- [ ] 能说明下一篇要复现 P01 还是 P02 及原因。

## 自测问题

1. 为什么配置文件和逐轮日志都需要？
2. 课程 Notebook 的预存输出能否作为你的研究结果？
3. 一个算法改进至少需要哪些基线和消融？
4. 接入真实实验时 Oracle 接口需要哪些字段？
5. 何时可以把结果写成论文证据？

## 上一单元 / 下一单元

- 上一单元：[Unit 08](../unit08_physics_closed_loop/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一阶段：[论文复现仓库](https://github.com/kelvint0714-arch/Reproduction-of-Active-Learning-in-Materials-Science)
