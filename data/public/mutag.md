# MUTAG 数据说明

## 用途

MUTAG 只用于 Day 34 学习“多张图组成一个 batch、GIN 节点更新、全局池化和图分类”。它不是下游任务数据，也不能支持下游任务性能结论。

## 官方来源与基本事实

- 集合：TU Dortmund University 的 TUDataset 图学习基准；
- 任务：二分类图任务；
- 图数量：188；
- 类别数：2；
- 官方统计的平均节点数：17.93；
- 官方统计的平均边数：19.79；
- 数据入口：[TUDataset 数据表](https://chrsmrrs.github.io/datasets/docs/datasets/)；
- 文件格式：[TUDataset 格式说明](https://chrsmrrs.github.io/datasets/docs/format/)；
- PyG 加载器：[PyTorch Geometric `TUDataset`](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.TUDataset.html)。

TUDataset 的无向边文件会同时记录两个方向，因此 PyG 中 `edge_index` 的列数不能直接当作不重复无向边数。

## 本仓库的使用边界

- 首次运行由 PyG 联网下载，缓存写入仓库根目录的 `.cache/pyg/`；
- 原始压缩包和处理后数据不提交到 Git；
- Day 34 使用固定随机种子建立教学划分；
- 只报告 validation 教学结果，不把这个小样本单次划分包装成正式基准；
- 不把 MUTAG 的类别或图表示映射成候选方案结论。

若将来正式使用该数据发表结果，需要按数据站说明同时引用 TUDataset 汇总论文和对应的原始 MUTAG 工作，并重新核对评价协议与许可。
