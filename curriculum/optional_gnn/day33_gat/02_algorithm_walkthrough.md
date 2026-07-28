# Day 33.2：公平比较并读取有界注意力权重

## 1. 保持 Day 32 的比较框架

继续复用：

```python
SEEDS = [7, 17, 27]
HIDDEN_CHANNELS = 16
EPOCHS = 120
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
```

数据和掩码也不重新随机生成。固定条件写在顶部，避免循环中悄悄为某个模型改变设置。

## 2. 定义 GAT

```python
class SmallGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        heads=2,
    ):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels 必须能被 heads 整除")

        channels_per_head = hidden_channels // heads
        self.conv1 = GATConv(
            in_channels,
            channels_per_head,
            heads=heads,
            concat=True,
            dropout=0.0,
        )
        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.0,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        hidden = F.elu(hidden)
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        return self.conv2(hidden, edge_index)
```

这里把 `GATConv` 内部的注意力 Dropout 固定为 0，只使用与 GCN/SAGE 相同的外部 `0.5` Dropout。这样训练随机正则化的主要位置更容易比较。

## 3. 先检查多头 shape

```python
gat = SmallGAT(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    dropout=0.5,
    heads=2,
)

gat.eval()
with torch.no_grad():
    first_hidden = gat.conv1(data.x, data.edge_index)
    logits = gat(data.x, data.edge_index)

assert first_hidden.shape == (data.num_nodes, 16)
assert logits.shape == (data.num_nodes, dataset.num_classes)
```

如果误把 `out_channels=16, heads=2` 写进第一层并启用拼接，第一层会得到 `[N,32]`。

## 4. 三模型工厂

```python
MODEL_BUILDERS = {
    "GCN": SmallGCN,
    "GraphSAGE": SmallGraphSAGE,
    "GAT": SmallGAT,
}
```

三个类都接收：

```text
in_channels, hidden_channels, out_channels, dropout
```

统一接口使训练函数不需要为某个模型单独增加 epoch 或不同指标。

## 5. 配对运行三种种子

```python
records = []
trained_gat_for_preview = None

for seed in SEEDS:
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

        if model_name == "GAT" and seed == SEEDS[0]:
            trained_gat_for_preview = model
```

保留一个预先指定的 GAT（第一个种子）做结构预览，避免先看所有结果再挑最方便讲故事的模型。

## 6. 汇总结果

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

不按 mean 排名并宣布“获胜者”。表格用于观察，不是统计检验。

## 7. 读取第一层注意力

```python
trained_gat_for_preview.eval()
with torch.no_grad():
    hidden, (attention_edges, alpha) = (
        trained_gat_for_preview.conv1(
            data.x,
            data.edge_index,
            return_attention_weights=True,
        )
    )
```

检查 shape：

```python
assert hidden.shape == (data.num_nodes, 16)
assert attention_edges.shape[0] == 2
assert alpha.shape[0] == attention_edges.shape[1]
assert alpha.shape[1] == 2
```

## 8. 只构造八行预览

```python
preview_rows = min(8, attention_edges.shape[1])

attention_preview = pd.DataFrame({
    "source": attention_edges[0, :preview_rows].cpu().numpy(),
    "target": attention_edges[1, :preview_rows].cpu().numpy(),
    "head_0": alpha[:preview_rows, 0].cpu().numpy(),
    "head_1": alpha[:preview_rows, 1].cpu().numpy(),
})
```

这里只读取前 8 行，不排序，不把它们称为“最重要边”。

## 9. 一个局部数值检查

可以检查某个目标节点、某个头的返回注意力之和：

```python
target_node = int(attention_edges[1, 0])
incoming = attention_edges[1] == target_node
head_zero_sum = alpha[incoming, 0].sum().item()
print(head_zero_sum)
```

由于 softmax 是在同一目标节点的入边之间归一化，这个值通常接近 1。浮点误差可能让它不是精确的十进制 `1.0`。

这个检查只验证实现结构，不证明这些权重有现实因果意义。

下一步：[进入教学 Notebook](tutorial.ipynb)。
