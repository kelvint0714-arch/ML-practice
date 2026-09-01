# 贡献指南

感谢你帮助改进 ML-Learning。这个仓库优先接受能让学习者“更容易理解、运行和验证”的改动，而不是单纯增加算法数量。

## 开始前

1. 阅读 [课程路线](curriculum/LEARNING_PATHS.md) 和 [教学指南](docs/teaching_guide.md)；
2. 确认改动属于教材、公开数据说明、脚本或维护文档中的哪一类；
3. 不要把真实项目数据、凭据、绝对本机路径或大型缓存提交到仓库；
4. 一个改动解决一个明确问题，避免同时重排无关文件。

## 教学内容的最低结构

Day 2–35 和 Unit 1–9 使用统一学习包：

```text
README.md
01_concepts.md
02_algorithm_walkthrough.md
tutorial.ipynb
03_exercises.md
04_reference_answers.md
```

任务卡至少要说明：为什么学、前置条件、今日产出、核心概念、分步骤任务、常见错误、完成标准、自测问题和前后导航。

练习与参考答案必须分开。练习中可以提供代码骨架，但不要直接填入答案；参考答案需要解释原因，不只给结果。

## Notebook 规范

教学 Notebook 至少包含：

- `Goal`：今天要验证什么；
- `Setup`：输入数据、随机种子和依赖；
- `Steps`：由小到大的实现过程；
- `Checks`：断言、手算或独立基线；
- `Next Steps`：个人练习和证据边界。

提交前：

1. Restart Kernel and Run All；
2. 确认代码单元执行编号连续；
3. 确认没有 error output；
4. 删除本机绝对路径、凭据和临时调试输出；
5. 不把公开数据教程结果描述为真实下游任务结果。

## 新算法或新路线的准入问题

提交前请回答：

- 它补足哪个明确学习目标？
- 学习者需要哪些前置知识？
- 最简单且可比较的基线是什么？
- 如何避免数据泄漏和测试集调参？
- 输出如何验证？
- 结果能说明与不能说明什么？
- 是否真的需要新增依赖或独立环境？

如果这些问题还不能回答，优先完善现有章节，不急着扩展新算法。

## 本地检查

不运行模型训练的快速检查：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

按改动范围执行 Notebook：

```bash
python scripts/run_curriculum_notebooks.py --first-day 2 --last-day 28
python scripts/run_curriculum_notebooks.py --first-day 29 --last-day 35 --kernel-name gnn
python scripts/run_active_learning_notebooks.py --first-unit 1 --last-unit 9
```

只需要运行与你改动相关的范围，但 Pull Request 中要写明运行了哪些检查。

## Pull Request 说明

建议包含：

- 问题：学习者原先会在哪里卡住；
- 改动：哪些文件和学习步骤发生变化；
- 验证：执行过的检查与结果；
- 边界：没有覆盖什么；
- 数据：是否新增数据、来源和授权状态；
- 截图：仅在 Notebook 或文档渲染变化需要视觉核对时提供。

## 不应提交

- 未授权真实配方或运行记录；
- `.env`、API key、账号信息；
- `.cache/`、Notebook 临时检查点和本地虚拟环境；
- 无法追溯来源的论文数据；
- 只为“让分数更好看”而反复查看 test 的结果；
- 把个人练习结果伪装成课程参考输出。

当前项目尚未声明开源许可证。外部贡献和内容复用应在许可证确定后再明确授权范围。
