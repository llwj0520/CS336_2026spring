"""拟合并绘制第二章要求的 IsoFLOPs 缩放定律。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "isoflops_curves.json"
MODEL_PLOT_PATH = ROOT / "isoflops_model_scaling.png"
DATA_PLOT_PATH = ROOT / "isoflops_data_scaling.png"
TARGET_COMPUTE = np.array([1e23, 1e24], dtype=float)


def load_runs(path: Path) -> list[dict[str, float]]:
    """读取作业提供的合成训练运行数据。"""
    with path.open(encoding="utf-8") as file:
        runs = json.load(file)

    required_fields = {"parameters", "compute_budget", "final_loss"}
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    if any(not required_fields.issubset(run) for run in runs):
        raise ValueError(f"Every run must contain {sorted(required_fields)}")
    return runs


def select_isoflops_minima(
    runs: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在每个计算预算下选择最终训练损失最低的运行。

    返回按计算预算升序排列的计算预算、最优参数量、最优训练 token 数，
    以及对应的最终训练损失。
    """
    runs_by_budget: dict[float, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        runs_by_budget[float(run["compute_budget"])].append(run)

    compute_budgets: list[float] = []
    optimal_parameters: list[float] = []
    optimal_tokens: list[float] = []
    optimal_losses: list[float] = []

    for compute_budget in sorted(runs_by_budget):
        best_run = min(
            runs_by_budget[compute_budget],
            key=lambda run: float(run["final_loss"]),
        )
        parameters = float(best_run["parameters"])
        tokens = compute_budget / (6.0 * parameters)

        compute_budgets.append(compute_budget)
        optimal_parameters.append(parameters)
        optimal_tokens.append(tokens)
        optimal_losses.append(float(best_run["final_loss"]))

    return (
        np.asarray(compute_budgets),
        np.asarray(optimal_parameters),
        np.asarray(optimal_tokens),
        np.asarray(optimal_losses),
    )


def fit_power_law(
    compute_budgets: np.ndarray,
    optimal_sizes: np.ndarray,
) -> tuple[float, float, float]:
    """在以 10 为底的对数空间中用普通最小二乘法拟合幂律。"""
    if np.any(compute_budgets <= 0) or np.any(optimal_sizes <= 0):
        raise ValueError("Power-law fitting requires strictly positive values")

    log_compute = np.log10(compute_budgets)
    log_size = np.log10(optimal_sizes)
    exponent, log10_coefficient = np.polyfit(log_compute, log_size, 1)

    fitted_log_size = exponent * log_compute + log10_coefficient
    residual_sum_squares = np.sum((log_size - fitted_log_size) ** 2)
    total_sum_squares = np.sum((log_size - log_size.mean()) ** 2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    coefficient = 10.0**log10_coefficient
    return float(coefficient), float(exponent), float(r_squared)


def predict_power_law(
    compute_budgets: np.ndarray,
    coefficient: float,
    exponent: float,
) -> np.ndarray:
    return coefficient * compute_budgets**exponent


def format_billions(value: float) -> str:
    return f"{value / 1e9:.1f}B"


def plot_power_law(
    *,
    compute_budgets: np.ndarray,
    observed_sizes: np.ndarray,
    coefficient: float,
    exponent: float,
    r_squared: float,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """绘制 IsoFLOPs 最优点、拟合曲线和目标预算下的预测点。"""
    curve_compute = np.logspace(
        np.log10(compute_budgets.min()),
        np.log10(TARGET_COMPUTE.max()),
        400,
    )
    curve_sizes = predict_power_law(curve_compute, coefficient, exponent)
    target_sizes = predict_power_law(TARGET_COMPUTE, coefficient, exponent)

    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.scatter(
        compute_budgets,
        observed_sizes,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        s=72,
        zorder=3,
        label="IsoFLOPs minima used for fitting",
    )
    ax.plot(
        curve_compute,
        curve_sizes,
        color="#1f77b4",
        linewidth=2.2,
        label="Fitted power law",
    )
    ax.scatter(
        TARGET_COMPUTE,
        target_sizes,
        color="#d62728",
        edgecolor="white",
        linewidth=0.8,
        marker="D",
        s=76,
        zorder=4,
        label="Predictions at target budgets",
    )

    for compute, size in zip(TARGET_COMPUTE, target_sizes, strict=True):
        exponent_label = int(np.log10(compute))
        is_last_prediction = compute == TARGET_COMPUTE.max()
        ax.annotate(
            rf"$10^{{{exponent_label}}}$: {format_billions(size)}",
            xy=(compute, size),
            xytext=(-8, -15 if is_last_prediction else 13),
            textcoords="offset points",
            ha="right",
            va="top" if is_last_prediction else "bottom",
            fontsize=9.5,
            color="#a61b1b",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C (FLOPs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.98,
        0.04,
        rf"$y={coefficient:.4g}C^{{{exponent:.4f}}}$"
        + "\n"
        + rf"$R^2={r_squared:.4f}$ (log space)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results(
    compute_budgets: np.ndarray,
    optimal_parameters: np.ndarray,
    optimal_tokens: np.ndarray,
    optimal_losses: np.ndarray,
    model_fit: tuple[float, float, float],
    data_fit: tuple[float, float, float],
) -> None:
    """输出复现第二章报告所需的全部数值。"""
    print("IsoFLOPs minima")
    print("C (FLOPs)        N_opt             D_opt             final loss")
    for compute, parameters, tokens, loss in zip(
        compute_budgets,
        optimal_parameters,
        optimal_tokens,
        optimal_losses,
        strict=True,
    ):
        print(f"{compute:10.3e}    {parameters:13.6e}    {tokens:13.6e}    {loss:.6f}")

    model_coefficient, model_exponent, model_r_squared = model_fit
    data_coefficient, data_exponent, data_r_squared = data_fit
    model_predictions = predict_power_law(
        TARGET_COMPUTE, model_coefficient, model_exponent
    )
    data_predictions = predict_power_law(
        TARGET_COMPUTE, data_coefficient, data_exponent
    )

    print("\nFitted scaling laws")
    print(
        f"N_opt(C) = {model_coefficient:.9g} * C^{model_exponent:.9f} "
        f"(log-space R^2 = {model_r_squared:.6f})"
    )
    print(
        f"D_opt(C) = {data_coefficient:.9g} * C^{data_exponent:.9f} "
        f"(log-space R^2 = {data_r_squared:.6f})"
    )

    print("\nExtrapolated predictions")
    for compute, parameters, tokens in zip(
        TARGET_COMPUTE, model_predictions, data_predictions, strict=True
    ):
        print(
            f"C = {compute:.0e}: N_opt = {parameters:.6e} parameters, "
            f"D_opt = {tokens:.6e} tokens"
        )


def main() -> None:
    runs = load_runs(DATA_PATH)
    compute_budgets, optimal_parameters, optimal_tokens, optimal_losses = (
        select_isoflops_minima(runs)
    )

    model_fit = fit_power_law(compute_budgets, optimal_parameters)
    data_fit = fit_power_law(compute_budgets, optimal_tokens)

    plot_power_law(
        compute_budgets=compute_budgets,
        observed_sizes=optimal_parameters,
        coefficient=model_fit[0],
        exponent=model_fit[1],
        r_squared=model_fit[2],
        ylabel="Compute-optimal parameters N",
        title="IsoFLOPs Scaling Law for Model Size",
        output_path=MODEL_PLOT_PATH,
    )
    plot_power_law(
        compute_budgets=compute_budgets,
        observed_sizes=optimal_tokens,
        coefficient=data_fit[0],
        exponent=data_fit[1],
        r_squared=data_fit[2],
        ylabel="Compute-optimal training tokens D",
        title="IsoFLOPs Scaling Law for Dataset Size",
        output_path=DATA_PLOT_PATH,
    )

    print_results(
        compute_budgets,
        optimal_parameters,
        optimal_tokens,
        optimal_losses,
        model_fit,
        data_fit,
    )
    print(f"\nSaved {MODEL_PLOT_PATH.name} and {DATA_PLOT_PATH.name}")


if __name__ == "__main__":
    main()
