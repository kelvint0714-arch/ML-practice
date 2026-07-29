# 课程共享参考

这些文件只解释跨天反复出现的内容。每天先理解算法，再运行当天最小代码；遇到具体语法或报错时才查询这里。

| 文件 | 什么时候查 |
|---|---|
| [完整 Day/Unit 学习包使用方法](day_package_guide.md) | 不清楚教材、教学 Notebook 与个人实验目录有什么区别，或准备开始 Day/Unit 时 |
| [主动学习术语表](../active_learning/shared/glossary.md) | 学习 labeled set、pool、Oracle、采集函数和 regret 时 |
| [主动学习实验协议](../active_learning/shared/experiment_protocol.md) | 比较 Random、UCB、EI 或不同代理模型时 |
| [主动学习泄漏检查表](../active_learning/shared/leakage_checklist.md) | 写 query 代码或审核候选标签权限时 |
| [主动学习 Python 语法速查](../active_learning/shared/python_patterns.md) | Unit 4–9 遇到函数参数、回调、列表推导、`groupby/agg`、`.loc` 或 `zip` 时 |
| [Python 语法字典](../core/day01_beginner/python_basics.md) | 在当天最小代码中遇到不认识的变量、函数、循环、字典、路径或异常时 |
| [机器学习词典](ml_glossary.md) | 忘记 `X/y`、特征、标签、Epoch、OOF 等术语时 |
| [统一实验协议](experiment_protocol.md) | 不确定能否调参、使用测试集或比较模型时 |
| [常见错误排查](error_guide.md) | Notebook、shape、依赖或文件路径报错时 |

## 查询原则

1. 先复制完整报错；
2. 找到报错最先指向的自己代码行；
3. 查当天任务卡的“常见错误”；
4. 再查本目录的通用说明；
5. 仍不能解决时，把完整报错、当天文件和刚做的操作一起发给 Codex。

不要在不理解原因时连续安装多个库或随机改版本，这样通常会把一个问题变成多个问题。
