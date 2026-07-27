# 完整 Day 学习包怎样使用

Day 2–28 的每个核心学习单元都使用同一种结构。文件提前准备好，是为了让你可以连续学习；它们不代表你本人已经完成实验。

```text
curriculum/core/dayXX_topic/
├── README.md                    # 当天任务卡与学习顺序
├── 01_concepts.md               # 先用中文理解算法
├── 02_algorithm_walkthrough.md  # 输入、动作、输出与伪代码
├── tutorial.ipynb               # 可从头运行的教学 Notebook
├── 03_exercises.md              # 不看答案完成的练习
└── 04_reference_answers.md      # 做完以后才核对
```

## 推荐顺序

1. 先读 `README.md`，只确认今天要解决的问题；
2. 阅读 `01_concepts.md`，不要急着运行代码；
3. 使用纸笔完成 `02_algorithm_walkthrough.md` 中要求的推演；
4. 从空内核运行 `tutorial.ipynb`；
5. 不看答案完成 `03_exercises.md`；
6. 最后使用 `04_reference_answers.md` 核对；
7. 把自己的 Notebook、笔记和结果保存在 `experiments/`；
8. 能独立解释以后，才在 `curriculum/PROGRESS.md` 勾选完成。

## 课程文件和个人实验的区别

| 位置 | 含义 | 能否当作你的实验成果 |
|---|---|---|
| `curriculum/` | 仓库提前提供的教材、示例与参考输出 | 不能 |
| `experiments/` | 你实际运行、修改并解释过的代码和结果 | 可以 |

教学 Notebook 中出现的数字只说明示例代码运行正常。你只有在自己从空内核运行、核对输出并写出解释后，才能把个人实验标为完成。

## 开始某一天

在仓库根目录运行：

```bash
python scripts/start_day.py 2
```

脚本会把 Day 2 的教学 Notebook 复制到对应的个人实验目录，清除教师预存的执行编号与输出，并建立：

```text
experiments/day02_metrics/
├── README.md
├── day02_metrics.ipynb
├── notes.md
└── results/
    └── README.md
```

脚本默认不会覆盖已经存在的个人文件。想查看将创建什么而不真正写入，可使用：

```bash
python scripts/start_day.py 2 --dry-run
```

个人副本会记录原始教学 Notebook 的相对路径，并标记为
`artifact_role=learner_workspace`。这个标记表示该文件应由学习者亲自运行；
刚复制完成、尚未运行的空输出本身仍不能算实验完成证据。

Day 13 和 Day 14 会根据 Notebook 的当前目录自动选择安全路径：

- 课程源 Notebook 只读写各自的 `curriculum/.../tutorial_outputs/`；
- `experiments/day13_fair_comparison/` 中的个人副本只写自己的 `results/`；
- `experiments/day14_ml_stage_report/` 中的个人副本只读取本人 Day 13 的
  `results/fold_metrics.csv`，并把报告写入自己的 `results/`。

因此应先实际完成并运行个人 Day 13，再运行个人 Day 14。不要把课程
`tutorial_outputs/` 复制到个人 `results/` 代替本人运行。

## 运行 Notebook

先进入仓库验证过的环境，再启动 JupyterLab：

```bash
conda activate esol
jupyter lab
```

也可以在终端执行个人副本：

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  experiments/day02_metrics/day02_metrics.ipynb
```

若你的环境名称不同，只要其中安装了 `requirements-learning.txt` 的依赖即可。

## 不应做的事

- 不要直接把参考答案复制到个人笔记；
- 不要把教学 Notebook 的预存输出称为自己的实验；
- 不要提前创建 27 份“已完成”结果；
- 不要为了赶进度跳过验证集、数据泄漏和测试集边界；
- 不要把 ESOL 或人工数据的教学结果写成粘合剂性能结论。
