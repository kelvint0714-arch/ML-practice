# 主动学习课程论文地图

本页说明每篇论文为哪个 Unit 提供方法依据，并给出可核对的 DOI/官方页面与
作者代码。论文精确复现、环境锁定和原文数值核对不在本仓库重复维护。

| 编号 | 完整论文题名（年份） | DOI / 官方论文 | 已确认代码或数据 | Unit 与主要学习点 |
|---|---|---|---|---|
| P01 | *Benchmarking the performance of Bayesian optimization across multiple experimental materials science domains* (2021) | [10.1038/s41524-021-00656-9](https://doi.org/10.1038/s41524-021-00656-9) | [PV-Lab/Benchmarking](https://github.com/PV-Lab/Benchmarking) | Unit 1–5：GP/RF、采集函数、多材料基准 |
| P02 | *Benchmarking active learning strategies for materials optimization and discovery* (2022) | [10.1093/oxfmat/itac006](https://doi.org/10.1093/oxfmat/itac006) | [作者 starter](https://github.com/alex-aa-wang/Benchmarking-Active-Learning-Starter)、[NIST REMI 数据](https://pages.nist.gov/remi/data/) | Unit 1、3–5：探索/利用、UCB、TS、EI、材料知识 |
| P03 | *Bgolearn: a unified Bayesian optimization framework for accelerating materials discovery* (2026) | [10.1038/s41524-026-02226-3](https://doi.org/10.1038/s41524-026-02226-3)、[arXiv:2601.06820](https://arxiv.org/abs/2601.06820) | [Bin-Cao/Bgolearn](https://github.com/Bin-Cao/Bgolearn)、[CodeDemo](https://github.com/Bgolearn/CodeDemo) | Unit 2、3、6：可替换代理、多采集函数、工程框架 |
| P04 | *Active and transfer learning with partially Bayesian neural networks for materials and chemicals* (2025) | [10.1039/D5DD00027K](https://doi.org/10.1039/D5DD00027K) | [NeuroBayes](https://github.com/ziatdinovmax/NeuroBayes)、[论文 AL 脚本](https://github.com/ziatdinovmax/NeuroBayes/tree/paper/active_learning_scripts) | Unit 2、7：PBNN、后验不确定性、主动/迁移学习 |
| P05 | *Uncovering multiscale structure-property correlations via active learning in scanning tunneling microscopy* (2025) | [10.1038/s41524-025-01642-1](https://doi.org/10.1038/s41524-025-01642-1) | [DKL_on_STM](https://github.com/gnganesh99/DKL_on_STM) | Unit 7、8：DNN 表示 + GP + UCB、真实测量坐标 |
| P06 | *Accelerating high-throughput virtual screening through molecular pool-based active learning* (2021) | [10.1039/D0SC06805E](https://doi.org/10.1039/D0SC06805E) | [MolPAL](https://github.com/coleygroup/molpal)、[论文冻结版本](https://github.com/coleygroup/molpal/releases/tag/publication) | Unit 5–7：大分子池、RF/FFN/MPNN、批量选择 |
| P07 | *On-the-fly closed-loop materials discovery via Bayesian active learning* (2020) | [10.1038/s41467-020-19597-w](https://doi.org/10.1038/s41467-020-19597-w) | [CAMEO_NComm](https://github.com/KusneNIST/CAMEO_NComm)、[Zenodo 数据/代码](https://doi.org/10.5281/zenodo.3998287) | Unit 8：材料知识增强的真实实验闭环 |
| P08 | *Autonomous intelligent agents for accelerated materials discovery* (2020) | [10.1039/D0SC01101K](https://doi.org/10.1039/D0SC01101K) | [TRI-AMDD/CAMD](https://github.com/TRI-AMDD/CAMD) | Unit 8：Agent–Experiment–Analyzer–Campaign |
| P09 | *DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models* (2020) | [10.1016/j.cpc.2020.107206](https://doi.org/10.1016/j.cpc.2020.107206)、[arXiv:1910.12690](https://arxiv.org/abs/1910.12690) | [deepmodeling/dpgen](https://github.com/deepmodeling/dpgen) | Unit 7–8：多神经网络分歧、MD 探索、DFT Oracle |
| P10 | *Physics makes the difference: Bayesian optimization and active learning via augmented Gaussian process* (2022) | [10.1088/2632-2153/ac4baa](https://doi.org/10.1088/2632-2153/ac4baa)、[arXiv:2108.10280](https://arxiv.org/abs/2108.10280) | [AugmentedGaussianProcess](https://github.com/ziatdinovmax/AugmentedGaussianProcess)、[gpax](https://github.com/ziatdinovmax/gpax) | Unit 8：物理均值 + GP 残差 + 主动选择 |
| S01 | *Physics-informed neural network for modelling the thermochemical curing process of composite-tool systems during manufacture* (2021) | [10.1016/j.cma.2021.113959](https://doi.org/10.1016/j.cma.2021.113959)、[arXiv:2011.13511](https://arxiv.org/abs/2011.13511) | [第一作者 sequential_PINN](https://github.com/saniaki/sequential_PINN) | Unit 8 支撑：PINN 是物理代理，不自动等于主动学习 |
| S02 | *Probabilistic physics-integrated neural differentiable modeling for isothermal chemical vapor infiltration process* (2024) | [10.1038/s41524-024-01307-5](https://doi.org/10.1038/s41524-024-01307-5)、[arXiv:2311.07798](https://arxiv.org/abs/2311.07798) | [PiNDiff-CVI](https://github.com/jx-wang-s-group/PiNDiff-CVI) | Unit 8 支撑：不完整物理、深度集成、未来采集接口 |
| S03 | *Machine learning models accelerate deep eutectic solvent discovery for the recycling of lithium-ion battery cathodes* (2024) | [10.1039/D4GC01418A](https://doi.org/10.1039/D4GC01418A) | [论文代码](https://github.com/wenbomu/ML-accelerate-deep-eutectic-solvent-discovery) | Unit 7/9 支撑：监督预测、解释、候选生成，不是 AL 闭环 |

## 阅读顺序

课程学习不要求一开始读完论文：

1. Unit 1 后读 P01/P02 的摘要、流程图和实验设置；
2. Unit 3 后比较 P01/P02/P03 的采集函数；
3. Unit 5 后读论文如何定义预算、重复和发现指标；
4. Unit 7 后读 P04/P05/P06，分清神经网络和主动学习；
5. Unit 8 后读 P07–P10，理解物理知识与真实闭环；
6. Unit 9 再选择 P01 或 P02 进入论文复现仓库。

## 不能混淆的边界

- 引用论文不等于复现论文；
- 运行软件示例不等于重现论文结论；
- CGAN 生成候选不等于主动学习；
- PINN 自适应配点不等于选择下一种材料；
- 模型不确定性只是采集依据之一，不能代替实验安全审核。
