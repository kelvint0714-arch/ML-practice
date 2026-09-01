# Day 26 参考答案

## A. 区域题

已标注集用于训练，池用于无标签选择，固定外部对照集用于不参与选择的同口径评价。若混用，策略会看到评价答案或对照集会被训练污染。候选标签返回并加入后，它属于下一轮已标注集。本教程的 ESOL valid 已在课程开发中查看过，因此不是严格未见的最终 test。

## B. 顺序题

拟合集成 → 计算候选分歧 → 固定并保存 query → 揭示被选标签 → 更新已标注集 → 在独立集评价。

## C. 泄漏审计

选择前用 `y_pool` 计算真实误差等于偷看答案。模拟结束后可以用隐藏标签做事后评价，但不能据此回头修改同一轮 query 或只报告有利方案。

## D. 随机对照

相同初始已标注集、候选池、batch budget、总轮数、模型与参数、评估集、指标和重复种子方案；任选五项并说明即可。

## E. 代码题

```python
def select_top_disagreement(prediction_matrix, candidate_ids, batch_size):
    prediction_matrix = np.asarray(prediction_matrix)
    candidate_ids = np.asarray(candidate_ids)
    if prediction_matrix.ndim != 2:
        raise ValueError("prediction_matrix must be 2-D")
    if prediction_matrix.shape[1] != len(candidate_ids):
        raise ValueError("one column is required per candidate")
    if not 1 <= batch_size <= len(candidate_ids):
        raise ValueError("batch_size is out of range")
    score = prediction_matrix.std(axis=0, ddof=0)
    order = np.lexsort((candidate_ids.astype(str), -score))
    return candidate_ids[order[:batch_size]]
```

## F. 交接表示例字段

`candidate_id, formulation_and_process, predicted_property, predicted_unit, disagreement_score, model_version, feasibility_status, prohibition_reason`。真实字段名和单位必须由领域团队确认。

## G. 结论边界

ESOL 的候选和标签都是已有公开练习数据，流程没有真正合成或测试新配方；下游任务体系、可行性约束、测量成本和噪声也不同。因此只能说接口与模拟协议得到练习。
