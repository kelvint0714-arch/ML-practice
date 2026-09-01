# 分支与仓库维护规则

## 分支原则

- `main` 是唯一长期分支，始终保持可阅读、可检查；
- 新工作使用短期分支，完成验证并合并后删除；
- 一个分支只解决一类问题，不混入无关课程或文档改动；
- 不对 `main` 强制推送，不用分支名称表示“永久版本”。

建议命名：

```text
learning/day02-metrics
learning/active-learning-unit03
tooling/notebook-check
docs/paper-review
```

## 文件应该放在哪里

| 内容 | 目录 |
|---|---|
| 学习任务卡、教程与答案 | `curriculum/` |
| 本地学习笔记和运行输出 | `learning_outputs/`（默认不提交） |
| 公开教学数据说明 | `data/public/` |
| 上手、教学与参考资料 | `docs/` |
| 仓库检查工具 | `scripts/` |

不要提交本地课程输出或大型缓存；需要长期保留的教学参考输出应放在对应课程目录，并写清来源。

## 合并前检查

在仓库根目录运行：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

然后确认：

1. 没有把真实项目数据、密码或本机绝对路径提交进去；
2. Notebook 没有错误输出，教学数字能追溯到输入和配置；
3. 新增 Markdown 链接可以打开；
4. 练习与参考答案没有混写；
5. 合并后删除已经完全包含的短期分支。
