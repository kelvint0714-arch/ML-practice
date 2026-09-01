# 分支与仓库维护规则

## 分支原则

- `main` 是唯一长期分支，始终保持可阅读、可检查；
- 新工作使用短期分支，完成验证并合并后删除；
- 一个分支只解决一类问题，不把课程、数据和实验的无关改动混在一起；
- 不对 `main` 强制推送，不用分支名称表示“永久版本”。

建议命名：

```text
learning/day02-metrics
data/adhesive-schema-v1
experiment/esol-multiseed
docs/paper-review
```

## 文件应该放在哪里

| 内容 | 目录 |
|---|---|
| 尚未完成的学习任务卡 | `curriculum/` |
| 实际运行的代码、记录和结果 | `experiments/` |
| 公开数据说明、空白接口 | `data/` |
| 项目路线、参考与归档 | `docs/` |
| 仓库检查工具 | `scripts/` |

不要为了表示“未来会做”而创建空实验目录。开始真实任务时再建立目录，并保存 `README.md`、`notes.md` 和 `results/`。

## 合并前检查

在仓库根目录运行：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

然后确认：

1. 没有把真实配方、合作方数据、密码或本机绝对路径提交进去；
2. Notebook 没有错误输出，实验数字能追溯到配置文件；
3. 新增 Markdown 链接可以打开；
4. 任务卡与实际完成状态没有混写；
5. 合并后删除已经完全包含的短期分支。
