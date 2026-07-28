# Day 34.2：GIN 图分类算法与数据流推演

## 1. 先写清输入、动作、输出

| 阶段 | 输入 | 动作 | 输出 |
|---|---|---|---|
| 数据 | 188 张公开图 | 固定种子打乱索引 | train/validation/test 索引 |
| Batch | 多张大小不同的图 | PyG 拼接节点和边 | `x`、`edge_index`、`batch`、`y` |
| GIN | 节点特征和边 | 邻居求和后经过 MLP | 节点隐藏表示 |
| Pooling | 节点表示和 `batch` | 按图求和 | 每图一个向量 |
| 分类 | 图向量 | 线性层 | 每图两个 logits |
| 评价 | validation logits/标签 | 取最大 logit 并比较 | validation accuracy |

## 2. 自动定位仓库和缓存

课程 Notebook 可能从课程目录或个人实验目录运行，所以不能写死本机绝对路径：

```python
from pathlib import Path

def find_repo_root(start=Path.cwd().resolve()):
    for candidate in (start, *start.parents):
        if (candidate / "curriculum").is_dir():
            return candidate
    raise FileNotFoundError("没有找到仓库根目录")

repo_root = find_repo_root()
cache_root = repo_root / ".cache" / "pyg"

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=str(cache_root), name="MUTAG")
```

`.cache/` 已被 Git 忽略。首次下载需要联网，但原始数据不进入课程提交。

## 3. 固定随机划分，不用 test 选模型

本教程只打乱样本索引，再切出 60%/20%/20%。建立划分时不读取标签，
因此不会为了让 test 类别比例更好看而使用 test 标签：

```python
import random

indices = list(range(len(dataset)))
random.Random(34).shuffle(indices)

n_train = int(0.60 * len(indices))
n_valid = int(0.20 * len(indices))
train_indices = indices[:n_train]
valid_indices = indices[n_train:n_train + n_valid]
test_indices = indices[n_train + n_valid:]

from torch.utils.data import Subset

train_subset = Subset(dataset, train_indices)
valid_subset = Subset(dataset, valid_indices)
```

之后只为 `train_indices` 和 `valid_indices` 创建 subset/loader；test 索引只做
集合互斥与覆盖检查。单次小样本随机划分只是教学协议；正式实验还需预先确定
是否分层、交叉验证、多随机种子、重复次数和结构相似性泄漏控制。

## 4. 看懂 `DataLoader` 输出

```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
one_batch = next(iter(train_loader))

print(one_batch.x.shape)
print(one_batch.edge_index.shape)
print(one_batch.batch.shape)
print(one_batch.y.shape)
print(one_batch.num_graphs)
```

必须满足：

```text
one_batch.x.shape[0] == one_batch.batch.shape[0]
one_batch.y.shape[0] == one_batch.num_graphs
```

## 5. 多数类基线

```python
from collections import Counter

train_labels = [int(dataset[i].y.item()) for i in train_indices]
majority_class = Counter(train_labels).most_common(1)[0][0]

valid_labels = [int(dataset[i].y.item()) for i in valid_indices]
majority_accuracy = sum(
    label == majority_class for label in valid_labels
) / len(valid_labels)
```

这里允许读取 validation 标签做开发评价，但不能读取 `test_indices` 对应标签。

## 6. GIN 前向传播

```python
import torch
from torch_geometric.nn import GINConv, global_add_pool

def make_mlp(in_channels, hidden_channels):
    return torch.nn.Sequential(
        torch.nn.Linear(in_channels, hidden_channels),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_channels, hidden_channels),
    )

class SmallGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(make_mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(make_mlp(hidden_channels, hidden_channels))
        self.output = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        hidden = torch.relu(self.conv1(x, edge_index))
        hidden = torch.relu(self.conv2(hidden, edge_index))
        graph_hidden = global_add_pool(hidden, batch)
        return self.output(graph_hidden)
```

`forward` 的三个输入分别是节点特征、边和节点所属图编号。输出第一维必须等于批次图数。

## 7. 固定轮数训练

中文伪代码：

```text
固定随机种子
建立模型和优化器
重复固定数量 epoch：
    切换 train 模式
    对每个训练 batch：
        前向传播
        只用训练图标签算交叉熵
        清空梯度、反向传播、更新参数
切换 eval 模式
只在 validation loader 上计算一次 accuracy
不创建 test loader，不计算 test accuracy
```

本教程不做 early stopping，也不保存“历史最好”模型，因此输出应写“固定 epoch 后的 validation”，不能写“最终测试性能”。

## 8. 执行顺序

现在从头运行 [教学 Notebook](tutorial.ipynb)，记录每个 shape 和断言。然后独立完成 [练习](03_exercises.md)，最后再查看参考答案。
