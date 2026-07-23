"""第三章离线练习：用合成实验拟合缩放定律并外推到 48 小时。"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


EFFECTIVE_FLOPS_PER_SECOND = 4e13
TARGET_HOURS = 48.0
TRAINING_BUDGETS = [0.25, 0.5, 0.75, 1.0]

# 每个预算运行 4 个模型，总模拟实验时间为 10 小时，低于 12 小时上限。
ARCHITECTURES = [
    {"num_layers": 6, "hidden_size": 256},
    {"num_layers": 8, "hidden_size": 384},
    {"num_layers": 10, "hidden_size": 512},
    {"num_layers": 12, "hidden_size": 640},
]


def compute_for_hours(hours):
    """将单卡训练小时数转换为合成计算量。"""
    return hours * 3600 * EFFECTIVE_FLOPS_PER_SECOND


def estimate_parameters(num_layers, hidden_size):
    """估算 Transformer 的非 embedding 参数量。"""
    return 12 * num_layers * hidden_size**2


def tokens_for_budget(hours, parameters):
    """根据 C=6ND 计算固定预算下的训练 token 数。"""
    return compute_for_hours(hours) / (6 * parameters)


def synthetic_validation_loss(parameters, tokens):
    """生成仅用于离线教学的合成验证损失。"""
    return 2.1 + 250 * parameters ** (-0.34) + 250 * tokens ** (-0.28)


def fit_isoflops_profile(hours):
    """用二次函数估计一个计算预算下的 IsoFLOPs 最低点。"""
    parameters = np.array(
        [
            estimate_parameters(item["num_layers"], item["hidden_size"])
            for item in ARCHITECTURES
        ]
    )
    tokens = np.array([tokens_for_budget(hours, value) for value in parameters])
    losses = np.array(
        [
            synthetic_validation_loss(model_size, data_size)
            for model_size, data_size in zip(parameters, tokens, strict=True)
        ]
    )

    log_parameters = np.log10(parameters)
    quadratic, linear, intercept = np.polyfit(log_parameters, losses, 2)
    if quadratic <= 0:
        raise ValueError(f"{hours} 小时的 profile 没有形成 U 形")

    log_n_opt = np.clip(
        -linear / (2 * quadratic),
        log_parameters.min(),
        log_parameters.max(),
    )
    n_opt = 10**log_n_opt
    d_opt = tokens_for_budget(hours, n_opt)
    loss_opt = quadratic * log_n_opt**2 + linear * log_n_opt + intercept

    return {
        "hours": hours,
        "compute": compute_for_hours(hours),
        "n_opt": n_opt,
        "d_opt": d_opt,
        "loss_opt": loss_opt,
    }


def fit_power_law(x, y):
    """在双对数空间拟合 y=A*x**a，并返回 A、a 和 R²。"""
    log_x = np.log10(x)
    log_y = np.log10(y)
    exponent, log_coefficient = np.polyfit(log_x, log_y, 1)
    fitted_log_y = exponent * log_x + log_coefficient
    r_squared = 1 - np.sum((log_y - fitted_log_y) ** 2) / np.sum(
        (log_y - log_y.mean()) ** 2
    )
    return 10**log_coefficient, exponent, r_squared


def loss_scaling_law(normalized_compute, irreducible_loss, coefficient, exponent):
    """描述计算最优验证损失随计算量下降的幂律。"""
    return irreducible_loss + coefficient * normalized_compute ** (-exponent)


def choose_architecture(target_parameters):
    """寻找与连续参数量预测最接近的合法层数和隐藏维度。"""
    candidates = []
    for hidden_size in range(256, 1281, 64):
        for num_layers in range(4, 33):
            parameters = estimate_parameters(num_layers, hidden_size)
            candidates.append(
                (
                    abs(math.log(parameters / target_parameters)),
                    num_layers,
                    hidden_size,
                    parameters,
                )
            )

    _, num_layers, hidden_size, parameters = min(candidates)
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "head_dim": 64,
        "num_attention_heads": hidden_size // 64,
        "intermediate_size": math.ceil((8 * hidden_size / 3) / 128) * 128,
        "parameters": parameters,
    }


def main():
    optimal_points = [fit_isoflops_profile(hours) for hours in TRAINING_BUDGETS]
    compute_values = np.array([point["compute"] for point in optimal_points])
    n_values = np.array([point["n_opt"] for point in optimal_points])
    d_values = np.array([point["d_opt"] for point in optimal_points])
    loss_values = np.array([point["loss_opt"] for point in optimal_points])

    model_coefficient, model_exponent, model_r_squared = fit_power_law(
        compute_values, n_values
    )
    data_coefficient, data_exponent, data_r_squared = fit_power_law(
        compute_values, d_values
    )

    minimum_compute = compute_values.min()
    normalized_compute = compute_values / minimum_compute
    loss_parameters, _ = curve_fit(
        loss_scaling_law,
        normalized_compute,
        loss_values,
        p0=[2.1, loss_values[0] - 2.1, 0.2],
        bounds=([0, 0, 0.01], [loss_values.min(), 10, 2]),
        maxfev=100_000,
    )
    fitted_losses = loss_scaling_law(normalized_compute, *loss_parameters)
    loss_r_squared = 1 - np.sum((loss_values - fitted_losses) ** 2) / np.sum(
        (loss_values - loss_values.mean()) ** 2
    )

    target_compute = compute_for_hours(TARGET_HOURS)
    predicted_n = model_coefficient * target_compute**model_exponent
    predicted_d = data_coefficient * target_compute**data_exponent
    predicted_loss = loss_scaling_law(
        target_compute / minimum_compute, *loss_parameters
    )

    architecture = choose_architecture(predicted_n)
    token_quantum = 512 * 128 * 16
    architecture_tokens = (
        round(target_compute / (6 * architecture["parameters"]) / token_quantum)
        * token_quantum
    )

    print("警告：以下结果来自离线合成数据，不能用于官方排行榜。")
    print(f"模拟实验预算：{4 * sum(TRAINING_BUDGETS):.2f} / 12.00 小时")
    print(
        f"N_opt(C) = {model_coefficient:.6g} * C^{model_exponent:.6f}，"
        f"R² = {model_r_squared:.6f}"
    )
    print(
        f"D_opt(C) = {data_coefficient:.6g} * C^{data_exponent:.6f}，"
        f"R² = {data_r_squared:.6f}"
    )
    print(f"损失拟合 R² = {loss_r_squared:.6f}")
    print(
        f"48 小时连续预测：N={predicted_n:.6e}，D={predicted_d:.6e}，loss={predicted_loss:.6f}"
    )
    print(
        "离散架构："
        f"{architecture['num_layers']} 层，hidden_size={architecture['hidden_size']}，"
        f"heads={architecture['num_attention_heads']}，N={architecture['parameters']:,}，"
        f"D={architecture_tokens:,}"
    )

    compute_curve = np.logspace(
        np.log10(compute_values.min()), np.log10(target_compute), 300
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, observed, coefficient, exponent, ylabel, title in (
        (
            axes[0],
            n_values,
            model_coefficient,
            model_exponent,
            "Parameters N",
            "Model scaling",
        ),
        (
            axes[1],
            d_values,
            data_coefficient,
            data_exponent,
            "Training tokens D",
            "Data scaling",
        ),
    ):
        curve = coefficient * compute_curve**exponent
        target = coefficient * target_compute**exponent
        ax.scatter(compute_values, observed, label="Profile optima")
        ax.plot(compute_curve, curve, label="Power-law fit")
        ax.scatter(target_compute, target, color="red", marker="D", label="48 h")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Synthetic compute C")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    loss_curve = loss_scaling_law(compute_curve / minimum_compute, *loss_parameters)
    axes[2].scatter(compute_values, loss_values, label="Profile optima")
    axes[2].plot(compute_curve, loss_curve, label="Loss fit")
    axes[2].scatter(
        target_compute, predicted_loss, color="red", marker="D", label="48 h"
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Synthetic compute C")
    axes[2].set_ylabel("Validation loss")
    axes[2].set_title("Loss scaling")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    fig.suptitle("Offline Scaling-Law Summary")
    fig.tight_layout()
    fig.savefig("chapter3_offline_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
