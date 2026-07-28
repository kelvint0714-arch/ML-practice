# Day 32.2：用配对种子比较 GCN 与 GraphSAGE

## 1. 复用 Day 31 的数据协议

加载相同的一张图，并使用同一 `make_fixed_masks` 函数：

```python
dataset = KarateClub()
data = dataset[0]

train_mask, valid_mask, test_mask = make_fixed_masks(
    data,
    mask_seed=20260728,
)
```

掩码种子与模型种子是不同概念：

- 掩码种子决定哪些剩余节点属于 validation/test；
- 模型种子决定权重初始化和 Dropout；
- 整个比较中掩码只建立一次，不随模型种子变化。

## 2. 定义相同外壳的两个模型

```python
class SmallGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        return self.conv2(hidden, edge_index)
```

GraphSAGE 只替换图算子：

```python
class SmallGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        return self.conv2(hidden, edge_index)
```

两个前向过程的 shape 相同：

```text
[N,F] → [N,16] → [N,C]
```

## 3. 模型工厂

```python
MODEL_BUILDERS = {
    "GCN": SmallGCN,
    "GraphSAGE": SmallGraphSAGE,
}
```

循环中根据名字构造全新模型，避免错误地让第二个算法继承第一个算法训练后的参数。

## 4. 单次训练函数

```python
def train_one_run(model_class, seed):
    set_seed(seed)

    model = model_class(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        dropout=0.5,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    for _ in range(120):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        valid_accuracy = masked_accuracy(
            logits,
            data.y,
            valid_mask,
        )

    return model, valid_accuracy
```

注意：

- `set_seed(seed)` 在构造每个新模型之前调用；
- test mask 没有传给评价函数；
- 函数返回已训练模型，是为了统计参数量和做 shape 检查，不是继续看 test。

## 5. 配对运行

```python
records = []

for seed in [7, 17, 27]:
    for model_name, model_class in MODEL_BUILDERS.items():
        model, valid_accuracy = train_one_run(
            model_class,
            seed,
        )
        records.append({
            "model": model_name,
            "seed": seed,
            "valid_accuracy": valid_accuracy,
            "parameters": count_trainable_parameters(model),
        })
```

循环顺序不会给某个模型更多训练轮数。每一行都保存，避免只选择最好 seed。

## 6. 汇总 mean/std

```python
run_table = pd.DataFrame(records)

summary = (
    run_table
    .groupby("model", as_index=False)
    .agg(
        valid_mean=("valid_accuracy", "mean"),
        valid_std=("valid_accuracy", "std"),
        parameters=("parameters", "first"),
        runs=("seed", "count"),
    )
)
```

参数量在不同种子下不变，所以使用 `first`。`runs` 应当等于 3。

## 7. 结果怎样表述

合适：

> 在固定 KarateClub validation 协议和 3 个配对种子下，两个模型的均值与波动如表所示。由于样本和重复次数很小、参数量不同且 test 未使用，本结果只作为教学比较。

不合适：

> GraphSAGE 已证明优于 GCN，因此应该用于粘合剂。

下一步：[进入教学 Notebook](tutorial.ipynb)。
