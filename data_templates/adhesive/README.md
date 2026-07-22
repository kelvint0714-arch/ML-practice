# 粘合剂数据格式讨论稿

[`粘合剂重要化学性质_数据格式_v0.3.xlsx`](./%E7%B2%98%E5%90%88%E5%89%82%E9%87%8D%E8%A6%81%E5%8C%96%E5%AD%A6%E6%80%A7%E8%B4%A8_%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F_v0.3.xlsx) 是一份供导师与化学组评审的空白数据接口，不是已经填好的公开数据集，也不是某一篇论文的原表复制。

## 用途

- 让化学组确认实际研究的粘合剂体系、字段名称、单位和可获得数据；
- 收集配方、核心理化性质、固化条件和一个实测性能标签；
- 在实际数据返回后，缩减为针对单一粘合剂体系和单一预测目标的 v1.0。

工作簿中的浅蓝色单元格由化学组必填，极浅蓝色单元格在条件成立或已有数据时填写。工作簿中没有虚构实验数值。

## 与算法的关系

当前版本是面向表格型监督学习的通用收集格式，可在字段确认后用于 Ridge、Random Forest、XGBoost 或 Gradient Boosting 等基线。它没有绑定某一种模型，也不直接支持 GNN，因为当前不包含分子图或结构文件。

正式建模前还需要固定：

1. 一种粘合剂化学体系；
2. 一个预测性能及其单位；
3. 一致的基材、胶层厚度、表面处理和测试方法；
4. 一行数据所代表的独立配方、样品或重复实验。

## 字段依据

工作簿的 `03_公开依据` 已列出完整链接，主要包括：

- Pruksawan et al. (2019), *Prediction and optimization of epoxy adhesive strength from a small dataset through active learning*: <https://doi.org/10.1080/14686996.2019.1673670>
- 官方补充材料 Table S2/S3: <https://ftp.ebi.ac.uk/pub/databases/biostudies/S-EPMC/118/S-EPMC6818118/Files/TSTA_A_1673670_SM4906.pdf>
- 3M Adhesive Chemistries: <https://www.3m.com/3M/en_US/bonding-and-assembly-us/resources/science-of-adhesion/intro-adhesive-tape-chemistries/>
- ASTM D1084 粘合剂黏度测试: <https://store.astm.org/d1084-16r21.html>
- ASTM D3418 聚合物转变温度测试: <https://store.astm.org/d3418-21.html>
