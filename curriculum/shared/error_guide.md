# 常见错误排查

## `NameError`

示例：

```text
NameError: name 'X_train' is not defined
```

通常原因是定义变量的前一个单元没有运行，或重启内核后直接运行了后面的单元。

处理顺序：

1. 找到第一次定义 `X_train` 的单元；
2. 从最前面按顺序运行；
3. 第一次完整执行使用 Restart Kernel and Run All。

## shape 不匹配

示例：

```text
Found input variables with inconsistent numbers of samples
```

先打印：

```python
print(X_train.shape)
print(y_train.shape)
```

第一维样本数必须相同。回归标签通常整理成 `(样本数,)`，不要凭感觉反复 `reshape`。

## 找不到文件

示例：

```text
FileNotFoundError
```

检查：

```python
from pathlib import Path

print(Path.cwd())
print(target_path)
print(target_path.exists())
```

不要把只在个人电脑上成立的绝对路径写进仓库代码。

## 缺少模块

示例：

```text
ModuleNotFoundError: No module named 'xxx'
```

先确认 Notebook 使用的是哪个 Python：

```python
import sys

print(sys.executable)
```

然后确认自己激活了课程环境。不要在 base、课程环境和 Notebook 内核之间混装依赖。

DeepChem 导入时可能提示缺少 PyTorch、TensorFlow、JAX 或 PyG。Day 1–28 使用的功能不一定需要这些可选依赖；只有出现真正的 traceback 并停止执行时，才按完整错误处理。

## 指标出现负数或很差

R² 小于 0 可能表示模型比预测训练标签均值还差，不等于代码一定坏了。先检查：

- 训练和验证的 `X/y` 是否对应；
- 标签单位和变换是否一致；
- 是否把 MAE/RMSE 的方向理解反了；
- 是否错误地在验证集上重新 `fit()`；
- 模型是否明显欠拟合。

## Notebook 能运行但没有输出文件

检查保存代码是否真的执行，以及目录是否存在：

```python
print(RESULTS_DIR)
print(RESULTS_DIR.exists())
```

保存后立刻检查：

```python
output_path = RESULTS_DIR / "metrics.csv"
table.to_csv(output_path, index=False)
assert output_path.exists()
```

## 训练时间过长

初学阶段不要先增加模型规模。依次检查：

1. 是否意外把参数组合写成了巨大的嵌套循环；
2. 是否把 `n_estimators`、epoch 或隐藏层设得过大；
3. 是否重复运行了同一单元；
4. 是否可以先用小样本完成语法检查；
5. 是否保存了中间结果。

## Git 显示很多陌生文件

先运行：

```bash
git status --short
```

不要使用会删除文件的命令。模型缓存、Notebook 临时文件和大型原始数据应先确认是否已被 `.gitignore` 排除。
