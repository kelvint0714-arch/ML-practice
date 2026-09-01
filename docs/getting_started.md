# 新手上手指南

这份指南解决第一次使用 ML-Learning 时最常见的三个问题：环境是否正确、从哪里开始、怎样留下自己的学习记录。

## 开始前需要什么

- Python 3.10；
- Git；
- Conda、Miniforge 或 Miniconda；
- 首次下载依赖和公开数据时可联网；
- 不需要提前掌握 DeepChem、RDKit、PyTorch 或 GNN。

## 第一步：进入仓库

```bash
git clone https://github.com/kelvint0714-arch/ML-practice.git
cd ML-practice
```

所有命令都应从仓库根目录运行。可以用下面的命令确认当前位置和仓库状态：

```bash
pwd
git status --short --branch
```

## 第二步：建立核心课程环境

```bash
conda create -n esol python=3.10.20 -y
conda activate esol
python -m pip install --upgrade pip
python -m pip install -r requirements-learning.txt
```

独立环境可以避免课程依赖与系统 Python 或其他项目互相影响。安装完成后运行：

```bash
python scripts/learn.py doctor --track core
```

理想输出是 Python、课程文件和核心依赖全部显示 `[OK]`。`[MISSING]` 表示当前路线无法完整运行；`[WARN]` 表示版本与验证快照不同，不一定会立即失败。

诊断命令只读取环境，不会自动安装、更新或删除软件。

## 第三步：查看下一项任务

```bash
python scripts/learn.py status
```

进度只来自 [curriculum/PROGRESS.md](../curriculum/PROGRESS.md)。工具会显示总进度、各路线进度、下一项名称和任务卡路径，但不会替你修改复选框。

零基础学习者从 Day 1 开始。已有基础者先阅读 [学习路线选择器](../curriculum/LEARNING_PATHS.md)，尤其不要跳过 Day 8–14 的可信评价。

## 第四步：完成一个学习单元

Day 2–35 和 Unit 1–9 使用统一结构：

```text
README.md                    # 任务、前置条件与完成标准
01_concepts.md               # 中文概念
02_algorithm_walkthrough.md  # 输入、步骤与伪代码
tutorial.ipynb               # 可运行教学 Notebook
03_exercises.md              # 独立练习
04_reference_answers.md      # 完成后核对
```

推荐顺序：

1. 先读任务卡，确认今天只解决什么问题；
2. 阅读概念并完成纸笔推演；
3. 打开 Notebook，选择正确内核；
4. Restart Kernel and Run All；
5. 不看答案完成练习；
6. 打开参考答案，记录错误原因；
7. 用自己的话写出“能说明 / 不能说明”；
8. 通过自测后再更新进度表。

个人笔记、CSV、JSON 或图片可以保存在本地 [`learning_outputs/`](../learning_outputs/README.md)。该目录默认不进入 Git 历史。

## Jupyter 内核选错怎么办

如果 Notebook 内核列表没有课程环境，先激活环境，再注册：

```bash
conda activate esol
python -m ipykernel install --user --name esol --display-name "Python 3 (esol)"
```

在 Notebook 中确认：

```python
import sys

print(sys.executable)
```

路径应指向 `esol` 环境，而不是系统 Python 或另一个项目。

## GNN 为什么使用独立环境

PyTorch 和 PyTorch Geometric 依赖较重，也不是核心课程的前置条件。只有学习 Day 29–35 时才创建：

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
python scripts/learn.py doctor --track gnn
```

## 常见首次失败

### `python: command not found`

Conda 环境可能没有激活。先运行 `conda activate esol`，再检查 `python --version`。

### `ModuleNotFoundError`

通常是安装依赖和运行 Notebook 使用了不同 Python。比较：

```bash
which python
python -m pip --version
```

再与 Notebook 中的 `sys.executable` 对照。

### 找不到文件

先确认当前目录是仓库根目录：

```bash
pwd
python scripts/learn.py status
```

代码中不要写个人电脑的绝对路径。

### ESOL 或 MUTAG 下载失败

确认网络后重试。缓存位于 `.cache/`，不需要提交 Git。网络受限时，可以先学习不依赖下载的人工数据章节。

### Notebook 可以运行，但看不懂结果

先回到 `01_concepts.md` 和 `02_algorithm_walkthrough.md`，用小数组手算。不要通过不断增加模型复杂度来绕过概念问题。

更多问题见 [常见错误排查](../curriculum/shared/error_guide.md)。

## 第一次学习的成功标准

- 环境诊断没有阻断问题；
- 能找到唯一进度表和下一项任务卡；
- 能运行一个最小 Notebook；
- 知道训练、验证和测试的用途不同；
- 知道参考答案只能在独立作答后查看；
- 能保存并解释自己的学习输出。
