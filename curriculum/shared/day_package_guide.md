# Day/Unit 学习包使用方法

Day 2–35 和 Unit 1–9 使用统一结构。文件已经准备好，只表示课程材料完整，不表示学习者已经掌握。

```text
dayXX_topic/ 或 unitXX_topic/
├── README.md                    # 任务卡与学习顺序
├── 01_concepts.md               # 中文概念
├── 02_algorithm_walkthrough.md  # 输入、步骤、公式和伪代码
├── tutorial.ipynb               # 可运行教学 Notebook
├── 03_exercises.md              # 独立练习
└── 04_reference_answers.md      # 完成后核对
```

## 推荐顺序

1. 读 `README.md`，确认今天只解决什么问题；
2. 阅读 `01_concepts.md`，先建立直觉；
3. 完成 `02_algorithm_walkthrough.md` 中的纸笔或流程推演；
4. 打开 `tutorial.ipynb`，Restart Kernel and Run All；
5. 关闭参考答案，独立完成 `03_exercises.md`；
6. 打开 `04_reference_answers.md`，记录差异和错误原因；
7. 在本地 `learning_outputs/` 写下输入、步骤、输出和限制；
8. 通过自测后，再更新 `curriculum/PROGRESS.md`。

## 预存输出怎样理解

教学 Notebook 的预存输出证明课程示例曾经顺序执行。它不是学习者完成课程的证据，也不是可以直接用于新任务的模型结论。

真正完成至少需要：

- 自己从空内核运行；
- 能解释每个关键变量；
- 通过断言或手算检查；
- 独立完成练习；
- 写清结果能说明和不能说明什么。

## 怎样保存个人学习内容

课程仓库不再预建或跟踪个人 Notebook 副本。你可以：

1. 直接运行教程，但不提交产生的输出变化；
2. 在本机复制 `tutorial.ipynb` 后运行；
3. 把笔记、CSV、JSON 和图片放入 `learning_outputs/`。

推荐结构：

```text
learning_outputs/
└── day02_metrics/
    ├── notes.md
    ├── metrics.csv
    └── figure.png
```

`learning_outputs/` 除说明文件外默认被 Git 忽略。

## 运行 Notebook

核心和主动学习课程：

```bash
conda activate esol
jupyter lab
```

也可以只做内存验证：

```bash
python scripts/run_curriculum_notebooks.py --first-day 2 --last-day 28
python scripts/run_active_learning_notebooks.py --first-unit 1 --last-unit 9
```

这些命令默认不改写课程源文件。

## GNN 环境

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
```

验证 Day 29–35：

```bash
python scripts/run_curriculum_notebooks.py \
  --first-day 29 \
  --last-day 35 \
  --kernel-name gnn
```

## 不应做的事

- 不要先看参考答案再填写练习；
- 不要把教学输出称为自己的独立结果；
- 不要用测试集反复调参；
- 不要把公开数据结论直接外推到其他任务；
- 不要提交真实项目数据、凭据或大型缓存。
