# Day 7.2：把 10 个代码单元映射回算法主线

本页不新增模型参数。先用 [tutorial.ipynb](tutorial.ipynb) 审计文件结构，再从空内核运行 [真实 ESOL 基线](../../../curriculum/core/day07_integrated_baseline/reference_baseline/esol_baseline.ipynb)。

## 1. 运行前环境检查

在仓库根目录：

```bash
conda activate esol
which python
python --version
python -c "import deepchem, rdkit, sklearn, numpy, pandas; print('imports ok')"
git status --short
```

若环境名不同，使用安装了 `requirements-learning.txt` 的环境。随后确认 Jupyter kernel 指向同一 Python。覆盖仓库内的参考结果前，`git status --short` 应没有输出；否则 `run_config.json` 会诚实记录工作树为 dirty，这次运行不能替代干净提交上的规范参考产物。课程开发改动尚未提交时，先不要覆盖历史结果。

任何导入失败都先解决，不继续相信 Notebook 旧输出。

## 2. 先做静态地图

当前 Notebook 的 10 个代码单元：

| 代码单元 | 主要职责 | 分类 |
|---:|---|---|
| 1 | 导入、路径、种子、版本、Git 状态 | 工程支持 |
| 2 | 配置 ECFP 并加载 ESOL scaffold splits | 数据准备 |
| 3 | shape、有限值和精确 ID 重叠检查 | 工程检查 |
| 4 | 形成 `X_train/y_train/X_valid/y_valid` | 算法输入 |
| 5 | 定义 MAE、RMSE、R² 函数 | 算法评价 |
| 6 | 创建 5 个固定候选 | 算法设置 |
| 7 | `fit → predict → metric` 循环 | 算法核心 |
| 8 | validation 排序和汇总展示 | 结果解释 |
| 9 | 保存 CSV、JSON 配置 | 工程记录 |
| 10 | 断言与证据边界摘要 | 工程验收 |

一个单元可以同时服务算法和工程；这里按主要职责分类。

## 3. 找出五个核心动作

在真实 Notebook 中亲自搜索并记录代码单元：

```text
X/y 形成：单元 __
模型创建：单元 __
fit：单元 __
predict：单元 __
metric：单元 __
```

不要从上表直接抄，必须在 Notebook 中定位相应代码。

## 4. 重启并顺序运行

Jupyter 菜单执行：

```text
Restart Kernel and Run All Cells
```

或在仓库根目录：

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  curriculum/core/day07_integrated_baseline/reference_baseline/esol_baseline.ipynb
```

执行期间不修改参数、不插入 test 预测。

## 5. 运行中遇到输出怎样处理

决策顺序：

```text
看到输出
├── 普通 print/表格 → 继续并记录
├── warning → 阅读是否影响当前 ECFP＋sklearn 路径
├── traceback → 停止，保存完整异常
└── AssertionError → 停止，找到违反的实验条件
```

出现错误后，不使用错误单元之后的旧保存输出写结论。

## 6. 验收执行状态

运行后检查：

```text
[ ] 10 个代码单元执行号为 1–10
[ ] 没有 error output
[ ] 最后一个单元执行完成
[ ] 结果表包含 5 模型 × 2 split = 10 行
[ ] 只有 train 和 valid，没有 test
[ ] baseline_metrics.csv 存在
[ ] run_config.json 存在
```

可以用课程审计 Notebook 做独立核对，但它不替代真实重跑。

## 7. 读取结果，而不是猜结果

机读指标：

```text
curriculum/core/day07_integrated_baseline/reference_baseline/results/baseline_metrics.csv
```

运行配置：

```text
curriculum/core/day07_integrated_baseline/reference_baseline/results/run_config.json
```

核对：

- `best_model_by_validation_rmse` 与 CSV 排序一致；
- 每个模型有 train/valid 两行；
- `source_commit`、模型参数和版本有记录；
- test 政策明确；
- 本次工作树状态与笔记一致。

## 8. 分析过拟合诊断

找到不限深决策树两行：

```text
train R²
valid R²
train/valid RMSE
```

按结构写：

```text
训练表现：
验证表现：
差距：
诊断：
不能推出：
```

“诊断”可以是明显过拟合信号；“不能推出”至少包括真实下游任务表现。

## 9. 为什么不新建另一个 Day 7 实验目录

Day 7 使用已有的历史参考实验：

```text
curriculum/core/day07_integrated_baseline/reference_baseline/
```

目录名中的 `day01` 是早期实验编号。不要复制一份相同结果到 `learning_outputs/day07...`，否则同一证据会被误认为两次独立实验。

你的学习证据可以追加到个人学习记录，但不能把预存输出标记成自己重跑的日期。只有实际 Run All 成功后，才记录本次重跑状态。

## 10. 完成后的三段话

1. **算法主线：**指出 X/y、模型、fit、predict、metric 的位置；
2. **本次观察：**引用实际重跑生成的 train/valid 指标；
3. **证据边界：**一次 validation、原 test 已暴露、不是下游任务结果。

下一步：[Day 7 练习与验收](03_exercises.md)。
