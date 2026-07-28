# Day 31.2：两层 GCN 如何完成一次训练

这一节把 Notebook 中的主流程拆开。先读数据，再建立固定掩码，最后才训练。

## 1. 固定随机性

```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- Python、NumPy、PyTorch 各自有随机数生成器；
- 固定种子使同一环境中的初始化和 Dropout 尽量可复现；
- 固定种子不保证所有硬件和所有库版本逐位完全相同，因此仍需记录环境。

## 2. 加载一张图

```python
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
data = dataset[0]
```

`dataset` 是数据集容器，`data` 是其中唯一一张图。不要把 `dataset[0]` 理解成“第一个节点”。

重要 shape：

```python
print(data.x.shape)
print(data.edge_index.shape)
print(data.y.shape)
```

它们分别对应 `[N, F]`、`[2, E]` 和 `[N]`。

## 3. 建立固定 validation/test 掩码

KarateClub 自带少量训练节点。我们保留这个 `train_mask`，把其余节点按固定顺序随机平分：

```python
def make_fixed_masks(data, mask_seed=20260728):
    train_mask = data.train_mask.clone()
    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)

    generator = torch.Generator().manual_seed(mask_seed)
    remaining = remaining[
        torch.randperm(remaining.numel(), generator=generator)
    ]

    valid_count = remaining.numel() // 2
    valid_nodes = remaining[:valid_count]
    test_nodes = remaining[valid_count:]

    valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    valid_mask[valid_nodes] = True
    test_mask[test_nodes] = True
    return train_mask, valid_mask, test_mask
```

这里没有读取剩余节点的标签，因此不会根据 validation/test 答案挑选节点。

必须检查：

```python
assert not torch.any(train_mask & valid_mask)
assert not torch.any(train_mask & test_mask)
assert not torch.any(valid_mask & test_mask)
assert torch.all(train_mask | valid_mask | test_mask)
```

`&` 是逐元素“同时为真”，`|` 是逐元素“至少一个为真”。

## 4. 定义两层模型

```python
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class SmallGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        hidden = F.relu(hidden)
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        logits = self.conv2(hidden, edge_index)
        return logits
```

构造模型：

```python
model = SmallGCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
)
```

`in_channels` 必须等于 `data.x.shape[1]`，`out_channels` 必须等于类别数。

## 5. 只做一次前向传播

```python
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)

assert logits.shape == (data.num_nodes, dataset.num_classes)
```

先检查 shape，可以在进入训练循环前发现维度错误。

## 6. 一次训练 epoch

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

model.train()
optimizer.zero_grad()

logits = model(data.x, data.edge_index)
loss = F.cross_entropy(
    logits[train_mask],
    data.y[train_mask],
)

loss.backward()
optimizer.step()
```

顺序含义：

1. `zero_grad()` 清除上一轮梯度；
2. 前向传播得到所有节点的 logits；
3. 掩码后只用训练标签计算 loss；
4. `backward()` 计算每个参数的梯度；
5. `step()` 根据梯度更新参数。

虽然模型输出所有节点，但 validation/test 标签没有进入 loss。

## 7. validation accuracy

```python
def masked_accuracy(logits, labels, mask):
    predictions = logits.argmax(dim=1)
    correct = predictions[mask] == labels[mask]
    return correct.float().mean().item()

model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    valid_accuracy = masked_accuracy(
        logits,
        data.y,
        valid_mask,
    )
```

validation 只评价，不反向传播。

## 8. 训练循环保存历史

```python
history = []

for epoch in range(1, 121):
    # 训练
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        logits[train_mask],
        data.y[train_mask],
    )
    loss.backward()
    optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        valid_logits = model(data.x, data.edge_index)
        valid_accuracy = masked_accuracy(
            valid_logits,
            data.y,
            valid_mask,
        )

    history.append({
        "epoch": epoch,
        "train_loss": loss.item(),
        "valid_accuracy": valid_accuracy,
    })
```

不要在循环里计算 test accuracy。否则每看一次，就在心理上多使用了一次 test。

## 9. 公平比较的预告

Day 32 替换为 GraphSAGE 时，下列项目不变：

- 数据；
- 三个掩码；
- 隐藏表示宽度；
- 优化器参数；
- Dropout；
- epoch 数；
- 随机种子集合；
- validation 指标。

仅替换图卷积算子，并额外报告参数量。这样比较仍不完美，但比同时修改全部超参数更可解释。

下一步：[进入教学 Notebook](tutorial.ipynb)。
