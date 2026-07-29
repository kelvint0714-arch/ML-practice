# Unit 01 练习

先独立完成，再看答案。

## 练习 1：模块归类

将以下项目放入“表示、代理模型、不确定性、采集函数、Oracle”：

- ECFP
- 随机森林
- 树间标准差
- UCB
- 拉伸强度实验
- GNN embedding

## 练习 2：手算 UCB

候选 D、E、F 的 \((\mu,\sigma)\) 分别为 `(5.0, 0.3)`、`(4.5, 1.1)`、`(5.2, 0.1)`。分别在 \(\beta=0\) 和 \(\beta=1\) 时计算 UCB 并选择候选。

## 练习 3：找泄漏

指出问题：

```python
best_pool_value = y_pool.max()
beta = 2.0 if best_pool_value > 10 else 0.5
query_pos = np.argmax(mean + beta * std)
```

## 练习 4：证据边界

用两句话说明公开数据查表 Oracle 能证明什么、不能证明什么。

## 练习 5：画流程

不用代码画出一轮闭环，并在图中标出 query 固定的位置和标签揭示线。
