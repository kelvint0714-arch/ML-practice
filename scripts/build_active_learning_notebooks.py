"""Build the nine deterministic active-learning tutorial notebooks.

The generated files are instructor-supplied curriculum artifacts. Run
``scripts/run_active_learning_notebooks.py --in-place`` afterwards to execute
them and refresh their saved outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = REPO_ROOT / "curriculum" / "active_learning"

UNIT_DIRECTORIES = {
    1: "unit01_foundations",
    2: "unit02_surrogates_uncertainty",
    3: "unit03_acquisition_functions",
    4: "unit04_multiround_loop",
    5: "unit05_benchmark_protocol",
    6: "unit06_batch_diversity_constraints",
    7: "unit07_neural_graph_surrogates",
    8: "unit08_physics_closed_loop",
    9: "unit09_capstone",
}


def clean(text: str) -> str:
    """Normalize a notebook cell body."""

    return dedent(text).strip() + "\n"


def markdown(text: str) -> nbformat.NotebookNode:
    """Create one Markdown cell."""

    return nbformat.v4.new_markdown_cell(clean(text))


def code(text: str) -> nbformat.NotebookNode:
    """Create one code cell."""

    return nbformat.v4.new_code_cell(clean(text))


def tutorial_notebook(
    unit_number: int,
    title: str,
    goal: str,
    setup_note: str,
    setup_code: str,
    steps: list[tuple[str, str, str]],
    checks_code: str,
    next_steps: str,
) -> nbformat.NotebookNode:
    """Return one consistently structured tutorial notebook."""

    cells = [
        markdown(
            f"""
            # Unit {unit_number:02d}｜{title}

            ## Goal

            {goal}

            本 Notebook 是确定性的人工教学实验，不是学习者已完成的研究，
            也不是粘合剂实验结果。
            """
        ),
        markdown(
            f"""
            ## Setup

            {setup_note}
            """
        ),
        code(setup_code),
        markdown(
            """
            ## Steps

            按顺序执行。每个变量第一次出现时，先确认它的类型、形状和标签权限。
            """
        ),
    ]

    for heading, explanation, source in steps:
        cells.append(
            markdown(
                f"""
                ### {heading}

                {explanation}
                """
            )
        )
        cells.append(code(source))

    cells.extend(
        [
            markdown(
                """
                ## Checks

                这些断言检查形状、预算和无重复等机械条件；通过断言不代表研究结论已经成立。
                """
            ),
            code(checks_code),
            markdown(
                f"""
                ## Next Steps

                {next_steps}
                """
            ),
        ]
    )

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "course_artifact": {
                "kind": "supplied_tutorial",
                "learner_evidence": False,
                "route": "active_learning",
                "unit": unit_number,
            },
            "kernelspec": {
                "display_name": "Python 3 (esol)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
    )


def unit01() -> nbformat.NotebookNode:
    """Build Unit 01: one leakage-safe query."""

    return tutorial_notebook(
        1,
        "主动学习闭环与标签揭示线",
        "用固定预测表完成一次 UCB query 和同预算 Random query，区分选样与标签返回。",
        "候选特征和模型预测在 query 前可见；`oracle_values` 只在候选编号固定后读取。",
        """
        import numpy as np
        import pandas as pd

        candidate_ids = np.array([f"C{i:02d}" for i in range(12)])
        X_pool = np.column_stack([
            np.linspace(0.10, 0.90, 12),
            np.linspace(80.0, 140.0, 12),
        ])
        prediction_mean = np.array(
            [5.1, 5.4, 5.8, 6.0, 6.1, 6.2, 6.3, 6.5, 6.6, 6.7, 6.8, 6.9]
        )
        prediction_std = np.array(
            [0.3, 0.5, 0.2, 0.9, 0.4, 0.2, 1.1, 0.3, 0.7, 0.2, 0.4, 0.6]
        )

        # 教学 Oracle：query 前不能用它调整策略。
        oracle_values = np.array(
            [4.9, 5.7, 5.6, 6.8, 6.0, 6.1, 7.5, 6.4, 7.0, 6.6, 6.9, 7.2]
        )

        print("X_pool shape:", X_pool.shape)
        print("候选数:", len(candidate_ids))
        """,
        [
            (
                "1. 只根据预测计算 UCB",
                "`beta` 在查看任何候选真实标签前固定。",
                """
                beta = 1.0
                ucb_score = prediction_mean + beta * prediction_std
                query_position = int(np.argmax(ucb_score))
                query_id = candidate_ids[query_position]

                ranking = pd.DataFrame({
                    "candidate_id": candidate_ids,
                    "prediction_mean": prediction_mean,
                    "prediction_std": prediction_std,
                    "ucb_score": ucb_score,
                }).sort_values(
                    ["ucb_score", "candidate_id"],
                    ascending=[False, True],
                )
                print(ranking.head(5).to_string(index=False))
                print("UCB query:", query_id)
                """,
            ),
            (
                "2. 固定同预算 Random query",
                "Random 也在标签揭示前确定，且只选择一个候选。",
                """
                random_rng = np.random.default_rng(2026)
                random_position = int(random_rng.integers(len(candidate_ids)))
                random_query_id = candidate_ids[random_position]
                print("Random query:", random_query_id)
                """,
            ),
            (
                "3. 越过标签揭示线",
                "现在 query 已经固定，才模拟实验返回真实性能。",
                """
                selected_y = float(oracle_values[query_position])
                random_selected_y = float(oracle_values[random_position])

                query_summary = pd.DataFrame([
                    {
                        "strategy": "ucb",
                        "candidate_id": query_id,
                        "observed_y": selected_y,
                    },
                    {
                        "strategy": "random",
                        "candidate_id": random_query_id,
                        "observed_y": random_selected_y,
                    },
                ])
                print(query_summary.to_string(index=False))
                """,
            ),
        ],
        """
        assert X_pool.shape == (12, 2)
        assert query_id in candidate_ids
        assert random_query_id in candidate_ids
        assert len(query_summary) == 2
        assert query_summary["observed_y"].notna().all()
        print("Unit 01 checks passed.")
        """,
        "完成练习并用自己的话画出五模块闭环，然后进入 Unit 02。不要根据这一轮胜负评价策略优劣。",
    )


def unit02() -> nbformat.NotebookNode:
    """Build Unit 02: GP posterior versus RF ensemble disagreement."""

    return tutorial_notebook(
        2,
        "代理模型与不确定性",
        "在同一一维候选池上比较 GP 后验标准差与 RF 集成分歧，并在 query 固定后诊断误差。",
        "人工目标函数只用于离线 Oracle。GP 使用固定核参数，避免教程输出随优化器变化。",
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

        X_all = np.linspace(0.0, 10.0, 31).reshape(-1, 1)
        oracle_values = np.sin(X_all[:, 0]) + 0.08 * X_all[:, 0]
        initial_indices = np.array([0, 6, 12, 18, 24, 30])
        pool_indices = np.setdiff1d(np.arange(len(X_all)), initial_indices)

        X_labeled = X_all[initial_indices]
        y_labeled = oracle_values[initial_indices]
        X_pool = X_all[pool_indices]
        print("labeled/pool:", X_labeled.shape, X_pool.shape)
        """,
        [
            (
                "1. GP 输出均值与后验标准差",
                "核参数固定；标准差的含义依赖该核与噪声假设。",
                """
                kernel = (
                    ConstantKernel(1.0, constant_value_bounds="fixed")
                    * RBF(1.2, length_scale_bounds="fixed")
                    + WhiteKernel(0.03, noise_level_bounds="fixed")
                )
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    optimizer=None,
                    normalize_y=True,
                )
                gp.fit(X_labeled, y_labeled)
                gp_mean, gp_std = gp.predict(X_pool, return_std=True)
                print("GP output shapes:", gp_mean.shape, gp_std.shape)
                """,
            ),
            (
                "2. RF 多成员产生启发式分歧",
                "每个成员使用不同种子；输出矩阵的行是成员、列是候选。",
                """
                seeds = [11, 22, 33, 44, 55]
                rf_predictions = []
                for seed in seeds:
                    member = RandomForestRegressor(
                        n_estimators=80,
                        max_depth=4,
                        random_state=seed,
                        n_jobs=1,
                    )
                    member.fit(X_labeled, y_labeled)
                    rf_predictions.append(member.predict(X_pool))

                rf_matrix = np.vstack(rf_predictions)
                rf_mean = rf_matrix.mean(axis=0)
                rf_std = rf_matrix.std(axis=0)
                print("RF prediction matrix:", rf_matrix.shape)
                """,
            ),
            (
                "3. 先固定 query，再做离线误差诊断",
                "两个一次性 query 都固定后，campaign 结束；evaluator 才查看整池误差，且结果不再返回策略。",
                """
                pool_ids = np.array([
                    f"U{index:02d}" for index in pool_indices
                ])

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])

                gp_query_local = select_with_tie_break(gp_std, pool_ids)
                rf_query_local = select_with_tie_break(rf_std, pool_ids)
                gp_query_global = int(pool_indices[gp_query_local])
                rf_query_global = int(pool_indices[rf_query_local])

                def offline_absolute_error(hidden_truth, prediction):
                    return np.abs(hidden_truth - prediction)

                # 事后 evaluator：两个 query 已固定，且本单元不再继续选点。
                offline_pool_truth = oracle_values[pool_indices]
                gp_error = offline_absolute_error(
                    offline_pool_truth,
                    gp_mean,
                )
                rf_error = offline_absolute_error(
                    offline_pool_truth,
                    rf_mean,
                )
                diagnosis = pd.DataFrame({
                    "candidate_x": X_pool[:, 0],
                    "gp_std": gp_std,
                    "gp_abs_error": gp_error,
                    "rf_disagreement": rf_std,
                    "rf_abs_error": rf_error,
                })
                print("GP query x:", float(X_all[gp_query_global, 0]))
                print("RF query x:", float(X_all[rf_query_global, 0]))
                print(diagnosis.sort_values("gp_std", ascending=False).head(5).to_string(index=False))
                """,
            ),
            (
                "4. 可视化预测与分歧",
                "阴影用于比较模型行为，不宣称是经过验证的 95% 预测区间。",
                """
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
                axes[0].plot(X_pool[:, 0], gp_mean, label="GP mean")
                axes[0].fill_between(
                    X_pool[:, 0],
                    gp_mean - gp_std,
                    gp_mean + gp_std,
                    alpha=0.25,
                    label="±1 GP std",
                )
                axes[0].scatter(X_labeled[:, 0], y_labeled, color="black", label="labeled")
                axes[0].set_title("GP posterior signal")
                axes[0].set_xlabel("candidate x")
                axes[0].set_ylabel("predicted/observed value")
                axes[0].legend()

                axes[1].plot(X_pool[:, 0], rf_mean, label="RF mean")
                axes[1].fill_between(
                    X_pool[:, 0],
                    rf_mean - rf_std,
                    rf_mean + rf_std,
                    alpha=0.25,
                    label="±1 ensemble std",
                )
                axes[1].scatter(X_labeled[:, 0], y_labeled, color="black", label="labeled")
                axes[1].set_title("RF ensemble disagreement")
                axes[1].set_xlabel("candidate x")
                axes[1].legend()
                plt.tight_layout()
                plt.show()
                """,
            ),
        ],
        """
        assert gp_mean.shape == (len(pool_indices),)
        assert rf_matrix.shape == (5, len(pool_indices))
        assert np.all(gp_std >= 0)
        assert np.all(rf_std >= 0)
        assert gp_query_global not in initial_indices
        assert rf_query_global not in initial_indices
        print("Unit 02 checks passed.")
        """,
        "完成不确定性来源练习。Unit 03 将保持预测表不变，只更换采集函数。",
    )


def unit03() -> nbformat.NotebookNode:
    """Build Unit 03: acquisition functions."""

    return tutorial_notebook(
        3,
        "采集函数",
        "在同一候选预测表上实现 Greedy、Uncertainty、UCB、PI、EI 和 Thompson Sampling。",
        "本单元不重新训练模型，避免把代理模型变化与采集函数变化混在一起。",
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        pool_ids = np.array([f"M{i:02d}" for i in range(8)])
        mean = np.array([6.9, 6.5, 6.2, 7.0, 6.7, 6.1, 6.8, 6.4])
        std = np.array([0.2, 0.9, 1.3, 0.1, 0.5, 1.6, 0.4, 1.0])
        best_observed = 6.6  # 只能来自已标注集
        beta = 1.0
        xi = 0.05
        print("候选数:", len(pool_ids), "已观测最好值:", best_observed)
        """,
        [
            (
                "1. 计算确定性采集分数",
                "Greedy 只看均值，Uncertainty 只看标准差，UCB 同时使用。",
                """
                greedy = mean
                uncertainty = std
                ucb = mean + beta * std
                """,
            ),
            (
                "2. 计算 PI 与 EI",
                "`safe_std` 防止除以零，`best_observed` 不得来自候选真实标签。",
                """
                improvement = mean - best_observed - xi
                safe_std = np.maximum(std, 1e-12)
                z = improvement / safe_std
                pi = norm.cdf(z)
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                ei = np.where(
                    std > 0,
                    ei,
                    np.maximum(improvement, 0.0),
                )
                """,
            ),
            (
                "3. 固定种子进行边际近似 TS",
                "这里只给出均值/标准差，无法恢复候选间协方差；因此是独立边际近似，不是 GP 联合后验的标准 TS。",
                """
                ts_rng = np.random.default_rng(2026)
                thompson_marginal_approx = ts_rng.normal(mean, std)
                """,
            ),
            (
                "4. 汇总策略选择",
                "所有策略使用相同并列规则和相同候选 ID。",
                """
                scores = {
                    "greedy": greedy,
                    "uncertainty": uncertainty,
                    "ucb": ucb,
                    "pi": pi,
                    "ei": ei,
                    "thompson_marginal_approx": thompson_marginal_approx,
                }

                score_table = pd.DataFrame({"candidate_id": pool_ids, **scores})
                selected = {}
                for name, score in scores.items():
                    order = np.lexsort((pool_ids.astype(str), -score))
                    selected[name] = pool_ids[order[0]]

                print(score_table.round(4).to_string(index=False))
                print("各策略选择:", selected)
                """,
            ),
            (
                "5. 查看 beta 如何改变 UCB",
                "beta 在实验前声明；本表只用于理解敏感性，不根据隐藏标签选 beta。",
                """
                beta_table = pd.DataFrame({
                    "candidate_id": pool_ids,
                    "beta_0": mean,
                    "beta_0.5": mean + 0.5 * std,
                    "beta_2": mean + 2.0 * std,
                })
                print(beta_table.round(3).to_string(index=False))
                """,
            ),
        ],
        """
        assert set(selected) == set(scores)
        assert all(candidate in pool_ids for candidate in selected.values())
        assert np.all(ei >= 0)
        assert np.all((pi >= 0) & (pi <= 1))
        assert selected["greedy"] == "M03"
        print("Unit 03 checks passed.")
        """,
        "手算一个 UCB/EI 例子。Unit 04 会把一种采集函数放进多轮循环。",
    )


def unit04() -> nbformat.NotebookNode:
    """Build Unit 04: multi-round loop."""

    return tutorial_notebook(
        4,
        "多轮主动学习循环",
        "用 GP + UCB 与 Random 在同一人工候选池运行八轮，并保存可审计 query log。",
        "策略函数只接收已标注 X/y、候选 X/ID 和 Oracle 回调；全局最优只在两条 campaign 完成后由 evaluator 读取。",
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

        X_all = np.linspace(0.0, 12.0, 41).reshape(-1, 1)
        oracle_values = (
            np.sin(0.9 * X_all[:, 0])
            + 0.35 * np.cos(2.2 * X_all[:, 0])
            + 0.06 * X_all[:, 0]
        )
        candidate_ids = np.array([f"C{i:02d}" for i in range(len(X_all))])
        initial_indices = np.array([0, 13, 27, 40])

        def offline_oracle(global_indices):
            indices = np.atleast_1d(global_indices).astype(int)
            values = oracle_values[indices].copy()
            return float(values[0]) if np.isscalar(global_indices) else values

        initial_values = offline_oracle(initial_indices)
        print("候选/初始:", len(X_all), len(initial_indices))
        """,
        [
            (
                "1. 定义固定 GP",
                "每轮建立同规格模型，避免上一轮对象中的隐藏状态。",
                """
                def make_gp():
                    kernel = (
                        ConstantKernel(1.0, constant_value_bounds="fixed")
                        * RBF(1.0, length_scale_bounds="fixed")
                        + WhiteKernel(0.02, noise_level_bounds="fixed")
                    )
                    return GaussianProcessRegressor(
                        kernel=kernel,
                        optimizer=None,
                        normalize_y=True,
                    )

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])
                """,
            ),
            (
                "2. 实现策略循环",
                "函数没有完整候选标签参数；局部位置映射到全局位置后，才调用 Oracle。",
                """
                def run_strategy(
                    strategy,
                    run_seed,
                    initial_indices,
                    initial_values,
                    oracle,
                    n_rounds=8,
                    beta=1.2,
                ):
                    labeled = list(np.asarray(initial_indices, dtype=int))
                    labeled_y = list(np.asarray(initial_values, dtype=float))
                    pool = [
                        index for index in range(len(X_all))
                        if index not in labeled
                    ]
                    rng = np.random.default_rng(run_seed)
                    records = []

                    for round_number in range(1, n_rounds + 1):
                        model = make_gp()
                        model.fit(X_all[labeled], np.asarray(labeled_y))
                        mean, std = model.predict(X_all[pool], return_std=True)

                        if strategy == "ucb":
                            score = mean + beta * std
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        elif strategy == "random":
                            score = np.full(len(pool), np.nan)
                            query_local = int(rng.integers(len(pool)))
                        else:
                            raise ValueError(f"Unknown strategy: {strategy}")

                        query_global = int(pool[query_local])
                        query_id = candidate_ids[query_global]
                        prediction = float(mean[query_local])
                        uncertainty = float(std[query_local])
                        acquisition = (
                            float(score[query_local])
                            if strategy == "ucb"
                            else np.nan
                        )

                        # 标签揭示线
                        observed_y = float(oracle(query_global))
                        labeled.append(query_global)
                        labeled_y.append(observed_y)
                        pool.remove(query_global)
                        best_so_far = float(np.max(labeled_y))

                        records.append({
                            "strategy": strategy,
                            "run_seed": run_seed,
                            "round": round_number,
                            "candidate_id": query_id,
                            "prediction_mean": prediction,
                            "prediction_std": uncertainty,
                            "acquisition_score": acquisition,
                            "observed_y": observed_y,
                            "best_so_far": best_so_far,
                            "n_labeled": len(labeled),
                            "model_version": "fixed_gp_rbf_v1",
                        })

                    return pd.DataFrame(records)
                """,
            ),
            (
                "3. 运行同初始集的两种策略",
                "两条轨迹初始点和新增标签数相同；两条 campaign 完成后 evaluator 才读取全局最优。",
                """
                shared = {
                    "initial_indices": initial_indices.copy(),
                    "initial_values": initial_values.copy(),
                    "oracle": offline_oracle,
                }
                ucb_log = run_strategy("ucb", run_seed=2026, **shared)
                random_log = run_strategy("random", run_seed=2026, **shared)
                query_log = pd.concat([ucb_log, random_log], ignore_index=True)

                # 事后 evaluator：不再返回策略循环修改选点。
                offline_global_best = float(oracle_values.max())
                query_log["simple_regret"] = (
                    offline_global_best - query_log["best_so_far"]
                )
                print(query_log.head(6).round(4).to_string(index=False))
                """,
            ),
            (
                "4. 画 best-so-far 轨迹",
                "单种子图只验证机制，不能证明 UCB 稳定优于 Random。",
                """
                for strategy, frame in query_log.groupby("strategy"):
                    plt.plot(
                        frame["round"],
                        frame["best_so_far"],
                        marker="o",
                        label=strategy,
                    )
                plt.axhline(
                    offline_global_best,
                    color="black",
                    linestyle="--",
                    label="offline global best",
                )
                plt.xlabel("query round")
                plt.ylabel("best observed value")
                plt.title("Single-seed teaching trajectory")
                plt.legend()
                plt.show()
                """,
            ),
        ],
        """
        assert len(query_log) == 16
        assert query_log.groupby("strategy")["candidate_id"].nunique().eq(8).all()
        assert query_log.groupby("strategy")["n_labeled"].max().eq(12).all()
        assert (query_log["simple_regret"] >= -1e-12).all()
        assert query_log["model_version"].eq("fixed_gp_rbf_v1").all()
        for _, frame in query_log.groupby("strategy"):
            assert frame["best_so_far"].is_monotonic_increasing
        print("Unit 04 checks passed.")
        """,
        "检查 query log 字段并完成索引练习。Unit 05 将把单种子轨迹扩展为公平多种子基准。",
    )


def unit05() -> nbformat.NotebookNode:
    """Build Unit 05: repeated benchmark."""

    return tutorial_notebook(
        5,
        "公平基准协议",
        "先多种子比较材料发现的 Random/Greedy/UCB/EI，再比较全局模型学习的 Random/Uncertainty。",
        "本教程使用固定核 GP 和 8 个预声明种子，分别报告 simple regret 与固定测试集 RMSE。",
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

        def synthetic_objective(x):
            x = np.asarray(x, dtype=float)
            return (
                np.sin(0.8 * x)
                + 0.25 * np.cos(2.4 * x)
                + 0.04 * x
            )

        X_all = np.linspace(0.0, 14.0, 51).reshape(-1, 1)
        oracle_values = synthetic_objective(X_all[:, 0])
        candidate_ids = np.array([f"D{i:02d}" for i in range(len(X_all))])

        def discovery_oracle(global_indices):
            indices = np.atleast_1d(global_indices).astype(int)
            values = oracle_values[indices].copy()
            return float(values[0]) if np.isscalar(global_indices) else values

        run_seeds = [11, 22, 33, 44, 55, 66, 77, 88]
        print("候选数:", len(X_all), "种子数:", len(run_seeds))
        """,
        [
            (
                "1. 定义模型与 EI",
                "固定 GP 规格；EI 的当前最好值只来自已标注集。",
                """
                def make_gp():
                    kernel = (
                        ConstantKernel(1.0, constant_value_bounds="fixed")
                        * RBF(0.9, length_scale_bounds="fixed")
                        + WhiteKernel(0.02, noise_level_bounds="fixed")
                    )
                    return GaussianProcessRegressor(
                        kernel=kernel,
                        optimizer=None,
                        normalize_y=True,
                    )

                def expected_improvement(mean, std, best_observed, xi=0.01):
                    safe_std = np.maximum(std, 1e-12)
                    improvement = mean - best_observed - xi
                    z = improvement / safe_std
                    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                    return np.where(
                        std > 0,
                        ei,
                        np.maximum(improvement, 0.0),
                    )

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])
                """,
            ),
            (
                "2. 运行一个策略",
                "策略只接收初始标签和 Oracle 回调；记录第 0 轮及协议要求的全部 query 字段。",
                """
                def run_strategy(
                    strategy,
                    initial_indices,
                    initial_values,
                    run_seed,
                    oracle,
                    n_rounds=8,
                ):
                    labeled = list(np.array(initial_indices, dtype=int))
                    labeled_y = list(np.asarray(initial_values, dtype=float))
                    pool = [
                        index for index in range(len(X_all))
                        if index not in labeled
                    ]
                    rng = np.random.default_rng(run_seed + 1000)
                    records = [{
                        "event": "initial_state",
                        "strategy": strategy,
                        "run_seed": run_seed,
                        "round": 0,
                        "candidate_id": None,
                        "prediction_mean": np.nan,
                        "prediction_std": np.nan,
                        "acquisition_score": np.nan,
                        "observed_y": np.nan,
                        "n_labeled": len(labeled),
                        "best_so_far": float(np.max(labeled_y)),
                        "model_version": "fixed_gp_rbf_v1",
                    }]

                    for round_number in range(1, n_rounds + 1):
                        model = make_gp()
                        model.fit(X_all[labeled], np.asarray(labeled_y))
                        mean, std = model.predict(X_all[pool], return_std=True)

                        if strategy == "random":
                            score = np.full(len(pool), np.nan)
                            query_local = int(rng.integers(len(pool)))
                        elif strategy == "greedy":
                            score = mean
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        elif strategy == "ucb":
                            score = mean + 1.2 * std
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        elif strategy == "ei":
                            best_observed = float(np.max(labeled_y))
                            score = expected_improvement(mean, std, best_observed)
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        else:
                            raise ValueError(strategy)

                        query_global = int(pool[query_local])
                        record = {
                            "event": "query",
                            "strategy": strategy,
                            "run_seed": run_seed,
                            "round": round_number,
                            "candidate_id": candidate_ids[query_global],
                            "prediction_mean": float(mean[query_local]),
                            "prediction_std": float(std[query_local]),
                            "acquisition_score": (
                                float(score[query_local])
                                if strategy != "random"
                                else np.nan
                            ),
                            "model_version": "fixed_gp_rbf_v1",
                        }

                        # 标签揭示线
                        observed_y = float(oracle(query_global))
                        labeled.append(query_global)
                        labeled_y.append(observed_y)
                        pool.remove(query_global)
                        best_so_far = float(np.max(labeled_y))
                        record.update({
                            "observed_y": observed_y,
                            "n_labeled": len(labeled),
                            "best_so_far": best_so_far,
                        })
                        records.append(record)
                    return records
                """,
            ),
            (
                "3. 建立共享初始集并运行实验矩阵",
                "每个种子的四种策略共享同一初始集；全部 campaign 完成后 evaluator 才计算全局最优与 regret。",
                """
                records = []
                initial_sets = {}
                for run_seed in run_seeds:
                    initial_rng = np.random.default_rng(run_seed)
                    initial = initial_rng.choice(
                        len(X_all), size=4, replace=False
                    )
                    initial_sets[run_seed] = tuple(sorted(initial.tolist()))
                    initial_values = discovery_oracle(initial)
                    for strategy in ("random", "greedy", "ucb", "ei"):
                        records.extend(
                            run_strategy(
                                strategy,
                                initial_indices=initial.copy(),
                                initial_values=initial_values.copy(),
                                run_seed=run_seed,
                                oracle=discovery_oracle,
                            )
                        )

                runs = pd.DataFrame(records)
                offline_global_best = float(oracle_values.max())
                runs["simple_regret"] = (
                    offline_global_best - runs["best_so_far"]
                )
                print(runs.head(9).round(4).to_string(index=False))
                """,
            ),
            (
                "4. 汇总均值和标准差",
                "长表保留原始轨迹，汇总表用于阅读。",
                """
                summary = runs.groupby(
                    ["strategy", "round"], as_index=False
                ).agg(
                    regret_mean=("simple_regret", "mean"),
                    regret_std=("simple_regret", "std"),
                    n_runs=("run_seed", "nunique"),
                )
                final_summary = summary[summary["round"] == 8]
                print(final_summary.round(4).to_string(index=False))
                """,
            ),
            (
                "5. 增加全局模型学习基准",
                "Uncertainty 以降低固定测试集 RMSE 为目标；测试标签只由 evaluator 读取，不参与选样。",
                """
                X_model_pool = np.linspace(0.0, 14.0, 51).reshape(-1, 1)
                model_pool_values = synthetic_objective(X_model_pool[:, 0])
                model_candidate_ids = np.array([
                    f"L{i:02d}" for i in range(len(X_model_pool))
                ])
                X_fixed_test = np.linspace(0.07, 13.93, 80).reshape(-1, 1)
                y_fixed_test = synthetic_objective(X_fixed_test[:, 0])

                def model_learning_oracle(global_indices):
                    indices = np.atleast_1d(global_indices).astype(int)
                    values = model_pool_values[indices].copy()
                    return (
                        float(values[0])
                        if np.isscalar(global_indices)
                        else values
                    )

                def evaluate_fixed_test(model):
                    prediction = model.predict(X_fixed_test)
                    return float(np.sqrt(np.mean(
                        (y_fixed_test - prediction) ** 2
                    )))

                def run_model_learning(
                    strategy,
                    initial_indices,
                    initial_values,
                    run_seed,
                    oracle,
                    evaluator,
                    n_rounds=8,
                ):
                    labeled = list(np.asarray(initial_indices, dtype=int))
                    labeled_y = list(np.asarray(initial_values, dtype=float))
                    pool = [
                        index for index in range(len(X_model_pool))
                        if index not in labeled
                    ]
                    rng = np.random.default_rng(run_seed + 2000)
                    metric_records = []
                    query_records = []

                    for round_number in range(n_rounds + 1):
                        model = make_gp()
                        model.fit(
                            X_model_pool[labeled],
                            np.asarray(labeled_y),
                        )
                        metric_records.append({
                            "strategy": strategy,
                            "run_seed": run_seed,
                            "round": round_number,
                            "n_labeled": len(labeled),
                            "test_rmse": evaluator(model),
                            "model_version": "fixed_gp_rbf_v1",
                        })
                        if round_number == n_rounds:
                            break

                        mean, std = model.predict(
                            X_model_pool[pool],
                            return_std=True,
                        )
                        if strategy == "random":
                            score = np.full(len(pool), np.nan)
                            query_local = int(rng.integers(len(pool)))
                        elif strategy == "uncertainty":
                            score = std
                            query_local = select_with_tie_break(
                                score,
                                model_candidate_ids[pool],
                            )
                        else:
                            raise ValueError(strategy)

                        query_global = int(pool[query_local])
                        query_record = {
                            "strategy": strategy,
                            "run_seed": run_seed,
                            "round": round_number + 1,
                            "candidate_id": model_candidate_ids[query_global],
                            "prediction_mean": float(mean[query_local]),
                            "prediction_std": float(std[query_local]),
                            "acquisition_score": (
                                float(score[query_local])
                                if strategy != "random"
                                else np.nan
                            ),
                            "model_version": "fixed_gp_rbf_v1",
                        }
                        observed_y = float(oracle(query_global))
                        query_record["observed_y"] = observed_y
                        query_records.append(query_record)
                        labeled.append(query_global)
                        labeled_y.append(observed_y)
                        pool.remove(query_global)

                    return metric_records, query_records

                model_metric_records = []
                model_query_records = []
                for run_seed in run_seeds:
                    initial_rng = np.random.default_rng(run_seed)
                    initial = initial_rng.choice(
                        len(X_model_pool), size=4, replace=False
                    )
                    initial_values = model_learning_oracle(initial)
                    for strategy in ("random", "uncertainty"):
                        metric_rows, query_rows = run_model_learning(
                            strategy,
                            initial.copy(),
                            initial_values.copy(),
                            run_seed,
                            model_learning_oracle,
                            evaluate_fixed_test,
                        )
                        model_metric_records.extend(metric_rows)
                        model_query_records.extend(query_rows)

                model_learning_metrics = pd.DataFrame(model_metric_records)
                model_learning_queries = pd.DataFrame(model_query_records)
                model_learning_summary = model_learning_metrics.groupby(
                    ["strategy", "round"], as_index=False
                ).agg(
                    rmse_mean=("test_rmse", "mean"),
                    rmse_std=("test_rmse", "std"),
                    n_runs=("run_seed", "nunique"),
                )
                print(
                    model_learning_summary[
                        model_learning_summary["round"] == 8
                    ].round(4).to_string(index=False)
                )
                """,
            ),
            (
                "6. 画两类多种子学习曲线",
                "左图回答找最优材料，右图回答提升全局预测；误差带都是跨运行标准差，不是显著性检验。",
                """
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
                for strategy, frame in summary.groupby("strategy"):
                    x = frame["round"].to_numpy()
                    y = frame["regret_mean"].to_numpy()
                    spread = frame["regret_std"].fillna(0).to_numpy()
                    axes[0].plot(x, y, marker="o", label=strategy)
                    axes[0].fill_between(
                        x, y - spread, y + spread, alpha=0.12
                    )
                axes[0].set_xlabel("query round")
                axes[0].set_ylabel("simple regret")
                axes[0].set_title("Discovery objective")
                axes[0].legend()

                for strategy, frame in model_learning_summary.groupby("strategy"):
                    x = frame["round"].to_numpy()
                    y = frame["rmse_mean"].to_numpy()
                    spread = frame["rmse_std"].fillna(0).to_numpy()
                    axes[1].plot(x, y, marker="o", label=strategy)
                    axes[1].fill_between(
                        x, y - spread, y + spread, alpha=0.12
                    )
                axes[1].set_xlabel("query round")
                axes[1].set_ylabel("fixed-test RMSE")
                axes[1].set_title("Global model-learning objective")
                axes[1].legend()
                plt.tight_layout()
                plt.show()
                """,
            ),
        ],
        """
        assert len(runs) == 8 * 4 * 9
        assert summary["n_runs"].eq(8).all()
        assert runs.groupby(["strategy", "run_seed"])["n_labeled"].max().eq(12).all()
        query_rows = runs[runs["event"] == "query"]
        required_query_fields = [
            "candidate_id",
            "prediction_mean",
            "prediction_std",
            "observed_y",
            "model_version",
        ]
        assert query_rows[required_query_fields].notna().all().all()
        assert query_rows.groupby(
            ["strategy", "run_seed"]
        )["candidate_id"].nunique().eq(8).all()
        for run_seed in run_seeds:
            round_zero = runs[(runs["run_seed"] == run_seed) & (runs["round"] == 0)]
            assert round_zero["best_so_far"].nunique() == 1
        assert (runs["simple_regret"] >= -1e-12).all()
        assert len(model_learning_metrics) == 8 * 2 * 9
        assert len(model_learning_queries) == 8 * 2 * 8
        assert model_learning_summary["n_runs"].eq(8).all()
        assert model_learning_queries[
            required_query_fields
        ].notna().all().all()
        for run_seed in run_seeds:
            round_zero = model_learning_metrics[
                (model_learning_metrics["run_seed"] == run_seed)
                & (model_learning_metrics["round"] == 0)
            ]
            assert round_zero["test_rmse"].nunique() == 1
        print("Unit 05 checks passed.")
        """,
        "保存个人长表并写谨慎结论。Unit 06 会在同一选样思想上加入批量、多样性和可行性。",
    )


def unit06() -> nbformat.NotebookNode:
    """Build Unit 06: batch diversity and constraints."""

    return tutorial_notebook(
        6,
        "批量、多样性与约束",
        "在二维候选池中先过滤不可行候选，再比较纯 top-score 与兼顾多样性的批次。",
        "二维特征已处于 0–1 范围。可行性只使用候选输入，不读取隐藏目标。",
        """
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import pairwise_distances

        axis = np.linspace(0.0, 1.0, 9)
        grid_a, grid_b = np.meshgrid(axis, axis)
        X_all = np.column_stack([grid_a.ravel(), grid_b.ravel()])
        candidate_ids = np.array([f"B{i:02d}" for i in range(len(X_all))])
        oracle_values = (
            2.0 * np.sin(2.5 * X_all[:, 0])
            + 1.5 * np.cos(2.0 * X_all[:, 1])
            + X_all[:, 0] * X_all[:, 1]
        )

        def offline_oracle(global_index):
            return float(oracle_values[int(global_index)])

        feasible_mask = (
            (X_all[:, 0] >= 0.125)
            & (X_all[:, 0] + X_all[:, 1] <= 1.45)
        )
        feasible_indices = np.flatnonzero(feasible_mask)
        initial_indices = feasible_indices[
            np.linspace(0, len(feasible_indices) - 1, 7, dtype=int)
        ]
        initial_values = np.array([
            offline_oracle(index) for index in initial_indices
        ])
        pool_indices = np.setdiff1d(feasible_indices, initial_indices)
        print("全部/可行/初始/候选:", len(X_all), len(feasible_indices), len(initial_indices), len(pool_indices))
        """,
        [
            (
                "1. 训练 RF 集成并计算分歧",
                "集成只读取已标注标签，候选标签保持隐藏。",
                """
                predictions = []
                for seed in [11, 22, 33, 44, 55]:
                    member = RandomForestRegressor(
                        n_estimators=100,
                        max_features="sqrt",
                        random_state=seed,
                        n_jobs=1,
                    )
                    member.fit(X_all[initial_indices], initial_values)
                    predictions.append(member.predict(X_all[pool_indices]))
                matrix = np.vstack(predictions)
                mean = matrix.mean(axis=0)
                disagreement = matrix.std(axis=0)
                """,
            ),
            (
                "2. 生成纯 top-score 批次",
                "分数相同时按候选 ID 固定顺序。",
                """
                batch_size = 6
                top_order = np.lexsort((
                    candidate_ids[pool_indices].astype(str),
                    -disagreement,
                ))
                top_local = top_order[:batch_size]
                top_global = pool_indices[top_local]

                random_rng = np.random.default_rng(2026)
                random_local = random_rng.choice(
                    len(pool_indices),
                    size=batch_size,
                    replace=False,
                )
                random_global = pool_indices[random_local]
                """,
            ),
            (
                "3. 生成 score + diversity 批次",
                "每次选择与已选批次距离较远、同时分数较高的候选。",
                """
                score_range = np.ptp(disagreement)
                normalized_score = (
                    (disagreement - disagreement.min())
                    / max(score_range, 1e-12)
                )
                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])

                selected_local = [
                    select_with_tie_break(
                        normalized_score,
                        candidate_ids[pool_indices],
                    )
                ]
                diversity_weight = 0.8

                while len(selected_local) < batch_size:
                    distances = pairwise_distances(
                        X_all[pool_indices],
                        X_all[pool_indices[selected_local]],
                    ).min(axis=1)
                    distance_range = np.ptp(distances)
                    normalized_distance = (
                        (distances - distances.min())
                        / max(distance_range, 1e-12)
                    )
                    combined = (
                        normalized_score
                        + diversity_weight * normalized_distance
                    )
                    combined[selected_local] = -np.inf
                    selected_local.append(
                        select_with_tie_break(
                            combined,
                            candidate_ids[pool_indices],
                        )
                    )

                diverse_local = np.array(selected_local)
                diverse_global = pool_indices[diverse_local]
                """,
            ),
            (
                "4. 比较批内距离并建立审核表",
                "只比较候选特征距离；proposed 表不包含 Oracle 标签。",
                """
                def batch_distance_summary(global_indices):
                    distances = pairwise_distances(X_all[global_indices])
                    upper = distances[
                        np.triu_indices(len(global_indices), k=1)
                    ]
                    return {
                        "minimum": float(upper.min()),
                        "mean": float(upper.mean()),
                    }

                distance_table = pd.DataFrame([
                    {
                        "strategy": "random",
                        **batch_distance_summary(random_global),
                    },
                    {
                        "strategy": "top_score",
                        **batch_distance_summary(top_global),
                    },
                    {
                        "strategy": "score_plus_diversity",
                        **batch_distance_summary(diverse_global),
                    },
                ])

                proposed = pd.DataFrame({
                    "global_index": diverse_global,
                    "candidate_id": candidate_ids[diverse_global],
                    "feature_a": X_all[diverse_global, 0],
                    "feature_b": X_all[diverse_global, 1],
                    "prediction_mean": mean[diverse_local],
                    "disagreement": disagreement[diverse_local],
                    "feasible": feasible_mask[diverse_global],
                    "review_status": "proposed",
                    "rejection_reason": "",
                })
                print(distance_table.round(3).to_string(index=False))
                print(proposed.round(3).to_string(index=False))
                """,
            ),
            (
                "5. 审核后才调用 Oracle，并保留失败/拒绝",
                "这里用确定性规则模拟人工审核与一次测量失败；真实项目必须由授权人员操作。",
                """
                reviewed = proposed.copy()
                reviewed["review_status"] = "approved"
                reviewed.loc[
                    reviewed.index[-1],
                    ["review_status", "rejection_reason"],
                ] = ["rejected", "教学模拟：设备档期不允许"]

                failed_index = reviewed.index[0]
                reviewed["observed_y"] = np.nan
                reviewed["failure_reason"] = ""
                for row_index, row in reviewed.iterrows():
                    if row["review_status"] != "approved":
                        continue
                    if row_index == failed_index:
                        reviewed.loc[
                            row_index,
                            ["review_status", "failure_reason"],
                        ] = ["failed", "教学模拟：测量失败"]
                        continue

                    # 标签揭示线：只有 approved 且实际完成的候选才查询。
                    reviewed.loc[row_index, "observed_y"] = offline_oracle(
                        int(row["global_index"])
                    )
                    reviewed.loc[row_index, "review_status"] = "completed"

                print(reviewed.round(3).to_string(index=False))
                """,
            ),
        ],
        """
        assert len(np.unique(top_global)) == batch_size
        assert len(np.unique(diverse_global)) == batch_size
        assert len(np.unique(random_global)) == batch_size
        assert feasible_mask[top_global].all()
        assert feasible_mask[diverse_global].all()
        assert feasible_mask[random_global].all()
        assert proposed["review_status"].eq("proposed").all()
        assert "observed_y" not in proposed.columns
        means = distance_table.set_index("strategy")["mean"]
        assert means["score_plus_diversity"] >= means["top_score"]
        assert set(reviewed["review_status"]) == {
            "completed", "failed", "rejected"
        }
        assert reviewed.loc[
            reviewed["review_status"] == "completed", "observed_y"
        ].notna().all()
        assert reviewed.loc[
            reviewed["review_status"] != "completed", "observed_y"
        ].isna().all()
        print("Unit 06 checks passed.")
        """,
        "改变多样性权重并记录批次变化。Unit 07 将在相同采集接口下替换 RF/MLP 等代理模型。",
    )


def unit07() -> nbformat.NotebookNode:
    """Build Unit 07: RF versus MLP ensemble roles."""

    return tutorial_notebook(
        7,
        "神经网络与图代理模型的位置",
        "在同一表格候选池上比较 RF ensemble 与 MLP ensemble 的均值、分歧和 UCB 选择。",
        "本 Notebook 不训练 GNN、PBNN 或 DKL；它演示所有代理模型都应满足的统一输出接口。",
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X_all, oracle_values = make_regression(
            n_samples=120,
            n_features=8,
            n_informative=6,
            noise=8.0,
            random_state=2026,
        )
        candidate_ids = np.array([f"N{i:03d}" for i in range(len(X_all))])
        split_rng = np.random.default_rng(2026)
        labeled_indices = np.sort(
            split_rng.choice(len(X_all), size=32, replace=False)
        )
        pool_indices = np.setdiff1d(np.arange(len(X_all)), labeled_indices)
        X_labeled = X_all[labeled_indices]
        y_labeled = oracle_values[labeled_indices]
        X_pool = X_all[pool_indices]
        print("labeled/pool/features:", X_labeled.shape, X_pool.shape)
        """,
        [
            (
                "1. 建立 RF 和 MLP 成员",
                "两者使用相同样本和目标；MLP 只在已标注集拟合 X/y 标准化，并显式检查是否达到迭代上限。",
                """
                seeds = [11, 22, 33, 44, 55]
                rf_members = [
                    RandomForestRegressor(
                        n_estimators=100,
                        max_features="sqrt",
                        random_state=seed,
                        n_jobs=1,
                    )
                    for seed in seeds
                ]
                mlp_members = [
                    make_pipeline(
                        StandardScaler(),
                        TransformedTargetRegressor(
                            regressor=MLPRegressor(
                                hidden_layer_sizes=(24,),
                                solver="lbfgs",
                                alpha=0.01,
                                max_iter=2000,
                                tol=1e-6,
                                random_state=seed,
                            ),
                            transformer=StandardScaler(),
                        ),
                    )
                    for seed in seeds
                ]
                """,
            ),
            (
                "2. 使用统一 fit/predict 接口",
                "函数不接收候选标签，输出形状为成员数 × 候选数。",
                """
                def ensemble_predictions(members):
                    outputs = []
                    for member in members:
                        member.fit(X_labeled, y_labeled)
                        outputs.append(member.predict(X_pool))
                    return np.vstack(outputs)

                rf_matrix = ensemble_predictions(rf_members)
                mlp_matrix = ensemble_predictions(mlp_members)
                mlp_iterations = [
                    member.named_steps[
                        "transformedtargetregressor"
                    ].regressor_.n_iter_
                    for member in mlp_members
                ]
                print("RF/MLP matrices:", rf_matrix.shape, mlp_matrix.shape)
                print("MLP 迭代次数:", mlp_iterations)
                """,
            ),
            (
                "3. 使用同一 UCB 公式",
                "两个集成的分歧未经过概率校准，只作为教学采集信号。",
                """
                beta = 1.0
                rf_mean = rf_matrix.mean(axis=0)
                rf_std = rf_matrix.std(axis=0)
                mlp_mean = mlp_matrix.mean(axis=0)
                mlp_std = mlp_matrix.std(axis=0)

                rf_ucb = rf_mean + beta * rf_std
                mlp_ucb = mlp_mean + beta * mlp_std

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])

                rf_query_local = select_with_tie_break(
                    rf_ucb,
                    candidate_ids[pool_indices],
                )
                mlp_query_local = select_with_tie_break(
                    mlp_ucb,
                    candidate_ids[pool_indices],
                )
                rf_query_global = int(pool_indices[rf_query_local])
                mlp_query_global = int(pool_indices[mlp_query_local])
                """,
            ),
            (
                "4. query 后查看离线结果并映射模块",
                "一次 query 不能证明哪类模型更优。",
                """
                # 标签揭示线
                comparison = pd.DataFrame([
                    {
                        "surrogate": "RF ensemble",
                        "candidate_id": candidate_ids[rf_query_global],
                        "prediction_mean": rf_mean[rf_query_local],
                        "disagreement": rf_std[rf_query_local],
                        "offline_observed_y": oracle_values[rf_query_global],
                    },
                    {
                        "surrogate": "MLP ensemble",
                        "candidate_id": candidate_ids[mlp_query_global],
                        "prediction_mean": mlp_mean[mlp_query_local],
                        "disagreement": mlp_std[mlp_query_local],
                        "offline_observed_y": oracle_values[mlp_query_global],
                    },
                ])
                roles = pd.DataFrame([
                    ["RF/MLP/GNN", "表示或确定性代理"],
                    ["GP/PBNN", "概率代理模型"],
                    ["ensemble disagreement / posterior std", "不确定性信号"],
                    ["UCB/EI", "采集函数"],
                    ["查表/DFT/实验", "Oracle"],
                ], columns=["method", "active_learning_role"])
                result_note = (
                    "两类代理选中同一候选；本轮没有模型选择证据。"
                    if rf_query_global == mlp_query_global
                    else "两类代理选中不同候选；单轮差异不能证明谁更优。"
                )
                print(comparison.round(3).to_string(index=False))
                print(roles.to_string(index=False))
                print(result_note)
                """,
            ),
        ],
        """
        assert rf_matrix.shape == (5, len(pool_indices))
        assert mlp_matrix.shape == (5, len(pool_indices))
        assert max(mlp_iterations) < 2000
        assert rf_query_global in pool_indices
        assert mlp_query_global in pool_indices
        assert set(roles["active_learning_role"]) == {
            "表示或确定性代理",
            "概率代理模型",
            "不确定性信号",
            "采集函数",
            "Oracle",
        }
        assert "不能证明" in result_note or "没有模型选择证据" in result_note
        print("Unit 07 checks passed.")
        """,
        "完成模型角色练习。若要运行 GNN 扩展，先完成 Day 29–35；Unit 08 将加入物理均值和人工审核。",
    )


def unit08() -> nbformat.NotebookNode:
    """Build Unit 08: structured GP and approval state."""

    return tutorial_notebook(
        8,
        "物理先验与真实闭环状态",
        "比较普通 GP、合理物理均值和错误物理均值，并将成本、可行性和人工批准加入 query。",
        "两种物理趋势都在生成隐藏标签前定义；教程只模拟审批状态，不连接真实设备。",
        """
        import numpy as np
        import pandas as pd
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        temperature_axis = np.linspace(30.0, 120.0, 13)
        ratio_axis = np.linspace(0.1, 0.9, 11)
        temperature, ratio = np.meshgrid(temperature_axis, ratio_axis)
        X_all = np.column_stack([temperature.ravel(), ratio.ravel()])
        candidate_ids = np.array([f"P{i:03d}" for i in range(len(X_all))])

        def physics_mean(X):
            return (
                0.035 * X[:, 0]
                - 3.0 * (X[:, 1] - 0.60) ** 2
            )

        def wrong_physics_mean(X):
            return (
                -0.020 * X[:, 0]
                + 2.5 * (X[:, 1] - 0.25) ** 2
            )

        residual_truth = (
            0.55 * np.sin(X_all[:, 0] / 16.0)
            + 0.30 * np.cos(7.0 * X_all[:, 1])
        )
        oracle_values = physics_mean(X_all) + residual_truth

        def offline_oracle(global_index):
            return float(oracle_values[int(global_index)])

        feasible_mask = (
            (X_all[:, 0] <= 110.0)
            & (X_all[:, 1] >= 0.2)
            & (X_all[:, 1] <= 0.8)
        )
        experiment_cost = 1.0 + X_all[:, 0] / 120.0
        feasible_indices = np.flatnonzero(feasible_mask)
        initial_indices = feasible_indices[
            np.linspace(0, len(feasible_indices) - 1, 9, dtype=int)
        ]
        initial_values = np.array([
            offline_oracle(index) for index in initial_indices
        ])
        pool_indices = np.setdiff1d(np.arange(len(X_all)), initial_indices)
        print("all/feasible/initial/pool:", len(X_all), len(feasible_indices), len(initial_indices), len(pool_indices))
        """,
        [
            (
                "1. 定义同规格 GP Pipeline",
                "StandardScaler 只在已标注输入上 fit；固定核避免教程漂移。",
                """
                def make_gp():
                    kernel = (
                        ConstantKernel(1.0, constant_value_bounds="fixed")
                        * RBF(1.0, length_scale_bounds="fixed")
                        + WhiteKernel(0.02, noise_level_bounds="fixed")
                    )
                    return make_pipeline(
                        StandardScaler(),
                        GaussianProcessRegressor(
                            kernel=kernel,
                            optimizer=None,
                            normalize_y=True,
                        ),
                    )
                """,
            ),
            (
                "2. 普通、合理先验与错误先验 GP",
                "两个结构化模型都先减去预先定义的趋势，再让同规格 GP 学残差。",
                """
                ordinary_gp = make_gp()
                ordinary_gp.fit(
                    X_all[initial_indices],
                    initial_values,
                )
                ordinary_mean, ordinary_std = ordinary_gp.predict(
                    X_all[pool_indices],
                    return_std=True,
                )

                residual_gp = make_gp()
                residual_y = (
                    initial_values
                    - physics_mean(X_all[initial_indices])
                )
                residual_gp.fit(X_all[initial_indices], residual_y)
                residual_mean, structured_std = residual_gp.predict(
                    X_all[pool_indices],
                    return_std=True,
                )
                structured_mean = (
                    physics_mean(X_all[pool_indices]) + residual_mean
                )

                wrong_residual_gp = make_gp()
                wrong_residual_y = (
                    initial_values
                    - wrong_physics_mean(X_all[initial_indices])
                )
                wrong_residual_gp.fit(
                    X_all[initial_indices],
                    wrong_residual_y,
                )
                wrong_residual_mean, wrong_structured_std = (
                    wrong_residual_gp.predict(
                        X_all[pool_indices],
                        return_std=True,
                    )
                )
                wrong_structured_mean = (
                    wrong_physics_mean(X_all[pool_indices])
                    + wrong_residual_mean
                )
                """,
            ),
            (
                "3. 加入成本与可行性",
                "不可行候选分数设为负无穷；成本必须为正。",
                """
                beta = 1.2
                pool_cost = experiment_cost[pool_indices]
                pool_feasible = feasible_mask[pool_indices]
                structured_score = (
                    structured_mean + beta * structured_std
                ) / pool_cost
                structured_score = np.where(
                    pool_feasible, structured_score, -np.inf
                )
                ordinary_score = (
                    ordinary_mean + beta * ordinary_std
                ) / pool_cost
                ordinary_score = np.where(
                    pool_feasible, ordinary_score, -np.inf
                )
                wrong_structured_score = (
                    wrong_structured_mean
                    + beta * wrong_structured_std
                ) / pool_cost
                wrong_structured_score = np.where(
                    pool_feasible,
                    wrong_structured_score,
                    -np.inf,
                )

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])

                structured_local = select_with_tie_break(
                    structured_score,
                    candidate_ids[pool_indices],
                )
                ordinary_local = select_with_tie_break(
                    ordinary_score,
                    candidate_ids[pool_indices],
                )
                wrong_structured_local = select_with_tie_break(
                    wrong_structured_score,
                    candidate_ids[pool_indices],
                )
                structured_global = int(pool_indices[structured_local])
                ordinary_global = int(pool_indices[ordinary_local])
                wrong_structured_global = int(
                    pool_indices[wrong_structured_local]
                )
                """,
            ),
            (
                "4. 建立 proposed → approved → completed 状态",
                "教学中模拟人工批准；真实系统必须由授权人员完成。",
                """
                proposal = pd.DataFrame([
                    {
                        "model": "ordinary_gp",
                        "candidate_id": candidate_ids[ordinary_global],
                        "temperature": X_all[ordinary_global, 0],
                        "ratio": X_all[ordinary_global, 1],
                        "cost": experiment_cost[ordinary_global],
                        "feasible": feasible_mask[ordinary_global],
                        "review_status": "proposed",
                    },
                    {
                        "model": "physics_mean_plus_residual_gp",
                        "candidate_id": candidate_ids[structured_global],
                        "temperature": X_all[structured_global, 0],
                        "ratio": X_all[structured_global, 1],
                        "cost": experiment_cost[structured_global],
                        "feasible": feasible_mask[structured_global],
                        "review_status": "proposed",
                    },
                    {
                        "model": "wrong_physics_mean_plus_residual_gp",
                        "candidate_id": candidate_ids[
                            wrong_structured_global
                        ],
                        "temperature": X_all[
                            wrong_structured_global, 0
                        ],
                        "ratio": X_all[wrong_structured_global, 1],
                        "cost": experiment_cost[wrong_structured_global],
                        "feasible": feasible_mask[wrong_structured_global],
                        "review_status": "proposed",
                    },
                ])
                print("算法只生成 proposed:")
                print(proposal.round(3).to_string(index=False))

                # 教学模拟：人工只批准满足可行性规则的候选。
                reviewed = proposal.copy()
                reviewed.loc[
                    reviewed["feasible"], "review_status"
                ] = "approved"

                # 标签揭示线：只有 approved 才模拟 Oracle 返回。
                id_to_index = {
                    candidate_id: index
                    for index, candidate_id in enumerate(candidate_ids)
                }
                reviewed["observed_y"] = [
                    offline_oracle(id_to_index[candidate_id])
                    if status == "approved"
                    else np.nan
                    for candidate_id, status in zip(
                        reviewed["candidate_id"],
                        reviewed["review_status"],
                    )
                ]
                reviewed.loc[
                    reviewed["observed_y"].notna(),
                    "review_status",
                ] = "completed"
                print("审核与离线 Oracle 返回后:")
                print(reviewed.round(3).to_string(index=False))
                """,
            ),
            (
                "5. 离线检查模型误差",
                "该误差只用于教程诊断，不回头修改同一轮 query。",
                """
                offline_pool_truth = oracle_values[pool_indices]
                ordinary_mae = float(np.mean(np.abs(
                    offline_pool_truth - ordinary_mean
                )))
                structured_mae = float(np.mean(np.abs(
                    offline_pool_truth - structured_mean
                )))
                wrong_structured_mae = float(np.mean(np.abs(
                    offline_pool_truth - wrong_structured_mean
                )))
                ablation = pd.DataFrame([
                    {"model": "ordinary_gp", "offline_pool_mae": ordinary_mae},
                    {
                        "model": "reasonable_prior_plus_residual_gp",
                        "offline_pool_mae": structured_mae,
                    },
                    {
                        "model": "wrong_prior_plus_residual_gp",
                        "offline_pool_mae": wrong_structured_mae,
                    },
                ]).sort_values("offline_pool_mae")
                print(ablation.round(4).to_string(index=False))
                """,
            ),
        ],
        """
        assert np.all(experiment_cost > 0)
        selected_globals = [
            ordinary_global,
            structured_global,
            wrong_structured_global,
        ]
        assert feasible_mask[selected_globals].all()
        assert proposal["review_status"].eq("proposed").all()
        assert "observed_y" not in proposal.columns
        assert reviewed["review_status"].eq("completed").all()
        assert reviewed["observed_y"].notna().all()
        assert structured_mean.shape == ordinary_mean.shape
        assert wrong_structured_mean.shape == ordinary_mean.shape
        assert set(ablation["model"]) == {
            "ordinary_gp",
            "reasonable_prior_plus_residual_gp",
            "wrong_prior_plus_residual_gp",
        }
        print("Unit 08 checks passed.")
        """,
        "解释三组先验消融，并说明 PINN 还缺哪些模块才能进入主动学习。Unit 09 将组合完整研究包。",
    )


def unit09() -> nbformat.NotebookNode:
    """Build Unit 09: capstone research package in memory."""

    return tutorial_notebook(
        9,
        "主动学习综合演练",
        "在人工候选池用冻结配置运行 GP + Random/Greedy/UCB/EI，生成可审计研究包并准备公开论文复现。",
        "全部结果仅在内存中生成，不是公开数据基线。个人副本可将表保存到自己的 results/。",
        """
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

        config = {
            "artifact_kind": "deterministic_synthetic_tutorial",
            "objective": "maximize",
            "n_initial": 4,
            "n_rounds": 8,
            "batch_size": 1,
            "run_seeds": [11, 22, 33, 44, 55],
            "strategies": ["random", "greedy", "ucb", "ei"],
            "beta": 1.2,
            "xi": 0.01,
        }

        x1 = np.linspace(0.0, 1.0, 8)
        x2 = np.linspace(0.0, 1.0, 8)
        grid_1, grid_2 = np.meshgrid(x1, x2)
        X_all = np.column_stack([grid_1.ravel(), grid_2.ravel()])
        oracle_values = (
            1.8 * np.sin(3.2 * X_all[:, 0])
            + 1.2 * np.cos(4.0 * X_all[:, 1])
            + 0.7 * X_all[:, 0] * X_all[:, 1]
        )
        candidate_ids = np.array([f"CAP{i:03d}" for i in range(len(X_all))])

        def capstone_oracle(global_indices):
            indices = np.atleast_1d(global_indices).astype(int)
            values = oracle_values[indices].copy()
            return float(values[0]) if np.isscalar(global_indices) else values

        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("候选数:", len(X_all), "数据类型: 人工教学候选池")
        """,
        [
            (
                "1. 定义冻结模型和采集函数",
                "同一 GP 用于所有策略；Random 也训练模型以记录可比预测，但不使用分数选点。",
                """
                def make_gp():
                    kernel = (
                        ConstantKernel(1.0, constant_value_bounds="fixed")
                        * RBF(0.35, length_scale_bounds="fixed")
                        + WhiteKernel(0.02, noise_level_bounds="fixed")
                    )
                    return GaussianProcessRegressor(
                        kernel=kernel,
                        optimizer=None,
                        normalize_y=True,
                    )

                def expected_improvement(mean, std, best_observed, xi):
                    safe_std = np.maximum(std, 1e-12)
                    improvement = mean - best_observed - xi
                    z = improvement / safe_std
                    result = (
                        improvement * norm.cdf(z)
                        + std * norm.pdf(z)
                    )
                    return np.where(
                        std > 0,
                        result,
                        np.maximum(improvement, 0.0),
                    )

                def select_with_tie_break(score, ids):
                    order = np.lexsort((ids.astype(str), -np.asarray(score)))
                    return int(order[0])
                """,
            ),
            (
                "2. 运行一条完整轨迹",
                "逐轮日志保存 query 前字段；`observed_y` 在 ID 固定后加入。",
                """
                def run_strategy(
                    strategy,
                    initial_indices,
                    initial_values,
                    run_seed,
                    oracle,
                ):
                    labeled = list(np.array(initial_indices, dtype=int))
                    labeled_y = list(np.asarray(initial_values, dtype=float))
                    pool = [
                        index for index in range(len(X_all))
                        if index not in labeled
                    ]
                    rng = np.random.default_rng(run_seed + 5000)
                    records = []

                    for round_number in range(1, config["n_rounds"] + 1):
                        model = make_gp()
                        model.fit(X_all[labeled], np.asarray(labeled_y))
                        mean, std = model.predict(X_all[pool], return_std=True)

                        if strategy == "random":
                            score = np.full(len(pool), np.nan)
                            query_local = int(rng.integers(len(pool)))
                        elif strategy == "greedy":
                            score = mean
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        elif strategy == "ucb":
                            score = mean + config["beta"] * std
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        elif strategy == "ei":
                            score = expected_improvement(
                                mean,
                                std,
                                float(np.max(labeled_y)),
                                config["xi"],
                            )
                            query_local = select_with_tie_break(
                                score,
                                candidate_ids[pool],
                            )
                        else:
                            raise ValueError(strategy)

                        query_global = int(pool[query_local])
                        record = {
                            "strategy": strategy,
                            "run_seed": run_seed,
                            "round": round_number,
                            "candidate_id": candidate_ids[query_global],
                            "prediction_mean": float(mean[query_local]),
                            "prediction_std": float(std[query_local]),
                            "acquisition_score": (
                                float(score[query_local])
                                if strategy != "random"
                                else np.nan
                            ),
                            "model_version": "fixed_gp_rbf_v1",
                        }

                        # 标签揭示线
                        observed_y = float(oracle(query_global))
                        labeled.append(query_global)
                        labeled_y.append(observed_y)
                        pool.remove(query_global)
                        best_so_far = float(np.max(labeled_y))
                        record.update({
                            "observed_y": observed_y,
                            "n_labeled": len(labeled),
                            "best_so_far": best_so_far,
                        })
                        records.append(record)
                    return records
                """,
            ),
            (
                "3. 运行冻结实验矩阵",
                "每个种子的所有策略共享初始集，`.copy()` 避免相互修改。",
                """
                records = []
                for run_seed in config["run_seeds"]:
                    initial_rng = np.random.default_rng(run_seed)
                    initial_indices = initial_rng.choice(
                        len(X_all),
                        size=config["n_initial"],
                        replace=False,
                    )
                    initial_values = capstone_oracle(initial_indices)
                    for strategy in config["strategies"]:
                        records.extend(
                            run_strategy(
                                strategy,
                                initial_indices.copy(),
                                initial_values.copy(),
                                run_seed,
                                capstone_oracle,
                            )
                        )
                query_log = pd.DataFrame(records)
                offline_global_best = float(oracle_values.max())
                query_log["simple_regret"] = (
                    offline_global_best - query_log["best_so_far"]
                )
                print(query_log.head(8).round(4).to_string(index=False))
                """,
            ),
            (
                "4. 从长表生成指标",
                "指标表可以重新生成；不要手工修改以获得更好结论。",
                """
                metrics = query_log.groupby(
                    ["strategy", "round"], as_index=False
                ).agg(
                    regret_mean=("simple_regret", "mean"),
                    regret_std=("simple_regret", "std"),
                    best_mean=("best_so_far", "mean"),
                    n_runs=("run_seed", "nunique"),
                )
                final_metrics = metrics[
                    metrics["round"] == config["n_rounds"]
                ].sort_values("regret_mean")
                print(final_metrics.round(4).to_string(index=False))
                """,
            ),
            (
                "5. 生成学习曲线和探索消融",
                "Greedy 与 UCB/EI 的差异用于检查探索模块；曲线比最终单点更完整。",
                """
                for strategy, frame in metrics.groupby("strategy"):
                    plt.plot(
                        frame["round"],
                        frame["regret_mean"],
                        marker="o",
                        label=strategy,
                    )
                plt.xlabel("query round")
                plt.ylabel("mean simple regret")
                plt.title("Synthetic capstone rehearsal")
                plt.legend()
                plt.show()

                ablation_table = final_metrics[
                    final_metrics["strategy"].isin(
                        ["greedy", "ucb", "ei"]
                    )
                ][["strategy", "regret_mean", "regret_std"]]
                print(ablation_table.round(4).to_string(index=False))
                """,
            ),
            (
                "6. 形成处理并列且边界正确的报告",
                "最低值可能并列；报告必须引用人工候选池、固定协议、多种子和下一阶段。",
                """
                best_regret = float(final_metrics["regret_mean"].min())
                tied = final_metrics[
                    np.isclose(
                        final_metrics["regret_mean"],
                        best_regret,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ]["strategy"].astype(str).tolist()
                result_sentence = (
                    f"最终平均 regret 最低的策略为 {tied[0]} "
                    f"({best_regret:.3f})。"
                    if len(tied) == 1
                    else
                    f"最终平均 regret 最低值为 {best_regret:.3f}，"
                    f"{'、'.join(tied)} 并列；当前预算不能区分它们。"
                )
                report = "\\n".join([
                    "# 主动学习人工综合演练报告",
                    (
                        f"- 协议：人工候选池，{config['n_rounds']} 轮预算，"
                        f"{len(config['run_seeds'])} 个预声明种子。"
                    ),
                    f"- 结果：{result_sentence}",
                    "- 消融：比较 Greedy、UCB 与 EI，检查探索项是否带来差异。",
                    (
                        "- 负结果：若策略并列或复杂策略无优势，原样保留，"
                        "不事后改种子或参数。"
                    ),
                    "- 限制：结果不代表公开材料基准或真实粘合剂性能。",
                    "- 下一步：在 P01/P02 公开数据上按作者协议复现。",
                ])
                print(report)
                """,
            ),
        ],
        """
        expected_rows = (
            len(config["run_seeds"])
            * len(config["strategies"])
            * config["n_rounds"]
        )
        assert len(query_log) == expected_rows
        assert metrics["n_runs"].eq(len(config["run_seeds"])).all()
        assert query_log.groupby(
            ["strategy", "run_seed"]
        )["candidate_id"].nunique().eq(config["n_rounds"]).all()
        required_query_fields = [
            "candidate_id",
            "prediction_mean",
            "prediction_std",
            "observed_y",
            "model_version",
        ]
        assert query_log[required_query_fields].notna().all().all()
        assert query_log.groupby(
            ["strategy", "run_seed"]
        )["n_labeled"].max().eq(
            config["n_initial"] + config["n_rounds"]
        ).all()
        assert (query_log["simple_regret"] >= -1e-12).all()
        assert set(ablation_table["strategy"]) == {"greedy", "ucb", "ei"}
        assert len(tied) >= 1
        if len(tied) > 1:
            assert "并列" in report
        assert "真实粘合剂性能" in report
        print("Unit 09 checks passed.")
        """,
        "在个人副本保存 config、query_log、metrics、图和报告，再选择 P01 或 P02 进入公开论文复现。",
    )


BUILDERS = {
    1: unit01,
    2: unit02,
    3: unit03,
    4: unit04,
    5: unit05,
    6: unit06,
    7: unit07,
    8: unit08,
    9: unit09,
}


def comparable_source(notebook: nbformat.NotebookNode) -> dict[str, object]:
    """Return source-of-truth fields while ignoring execution artifacts."""

    kernelspec = notebook.metadata.get("kernelspec", {})
    language_info = notebook.metadata.get("language_info", {})
    return {
        "metadata": {
            "course_artifact": notebook.metadata.get("course_artifact"),
            "kernelspec": {
                key: kernelspec.get(key)
                for key in ("display_name", "language", "name")
            },
            "language_info": {
                key: language_info.get(key)
                for key in ("name",)
            },
        },
        "cells": [
            {
                "cell_type": cell.cell_type,
                "source": cell.source,
            }
            for cell in notebook.cells
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse maintainer options."""

    parser = argparse.ArgumentParser(
        description=(
            "Build active-learning tutorials, or verify that tracked "
            "Notebook sources match the builders."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated source cells without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    """Write or verify all active-learning tutorial notebooks."""

    args = parse_args()
    mismatches: list[str] = []
    for unit_number, directory_name in UNIT_DIRECTORIES.items():
        destination = ACTIVE_ROOT / directory_name / "tutorial.ipynb"
        notebook = BUILDERS[unit_number]()
        if args.check:
            if not destination.is_file():
                mismatches.append(
                    f"{destination.relative_to(REPO_ROOT)} is missing"
                )
                continue
            tracked = nbformat.read(destination, as_version=4)
            if comparable_source(tracked) != comparable_source(notebook):
                mismatches.append(
                    f"{destination.relative_to(REPO_ROOT)} source differs"
                )
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        print(f"built {destination.relative_to(REPO_ROOT)}")

    if mismatches:
        print("Active-learning Notebook source check failed:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    if args.check:
        print("All 9 tracked Notebook sources match the builders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
