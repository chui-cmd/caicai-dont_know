"""
Linear programming full workflow template.

Standard form handled by scipy.optimize.linprog:
    minimize     c^T x
    subject to   A_ub x <= b_ub
                 A_eq x == b_eq
                 lower_i <= x_i <= upper_i

For maximization, set OBJECTIVE_SENSE = "max"; the script will transform it.

Usage:
    1. Edit VARIABLE_NAMES, OBJECTIVE_COEFFICIENTS, constraints, and BOUNDS.
    2. Run:
       python scripts/四/线性规划全过程.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linprog


OUTPUT_DIR = Path("outputs/四/线性规划")

# Decision variables.
VARIABLE_NAMES = ["x1", "x2", "x3"]

# Objective function coefficients.
# Example: maximize 3*x1 + 2*x2 + 5*x3
OBJECTIVE_SENSE = "max"
OBJECTIVE_COEFFICIENTS = np.array([3, 2, 5], dtype=float)

# Inequality constraints: A_ub @ x <= b_ub.
A_UB = np.array([
    [1, 2, 1],
    [3, 1, 2],
], dtype=float)
B_UB = np.array([10, 18], dtype=float)
INEQUALITY_NAMES = ["资源约束1", "资源约束2"]

# Equality constraints: A_eq @ x == b_eq.
# Set both to None if there is no equality constraint.
A_EQ = None
B_EQ = None
EQUALITY_NAMES = []

# Bounds for each variable: (lower_bound, upper_bound).
# Use None for unbounded upper/lower side.
BOUNDS = [
    (0, None),
    (0, None),
    (0, None),
]


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="SimHei")


def check_model():
    n_vars = len(VARIABLE_NAMES)
    if len(OBJECTIVE_COEFFICIENTS) != n_vars:
        raise ValueError("OBJECTIVE_COEFFICIENTS length must match VARIABLE_NAMES.")
    if len(BOUNDS) != n_vars:
        raise ValueError("BOUNDS length must match VARIABLE_NAMES.")
    if OBJECTIVE_SENSE not in ["min", "max"]:
        raise ValueError("OBJECTIVE_SENSE must be 'min' or 'max'.")
    if A_UB is not None and A_UB.shape[1] != n_vars:
        raise ValueError("A_UB column count must match variable count.")
    if A_EQ is not None and A_EQ.shape[1] != n_vars:
        raise ValueError("A_EQ column count must match variable count.")


def solve_linear_programming():
    check_model()
    c = OBJECTIVE_COEFFICIENTS.copy()
    if OBJECTIVE_SENSE == "max":
        c = -c

    result = linprog(
        c=c,
        A_ub=A_UB,
        b_ub=B_UB,
        A_eq=A_EQ,
        b_eq=B_EQ,
        bounds=BOUNDS,
        method="highs",
    )
    return result


def build_solution_table(result):
    if not result.success:
        return pd.DataFrame(columns=["变量", "最优取值"])
    return pd.DataFrame({
        "变量": VARIABLE_NAMES,
        "最优取值": result.x,
    })


def build_objective_table(result):
    if result.success:
        objective_value = OBJECTIVE_COEFFICIENTS @ result.x
    else:
        objective_value = np.nan

    return pd.DataFrame([{
        "优化方向": "最大化" if OBJECTIVE_SENSE == "max" else "最小化",
        "是否求解成功": result.success,
        "求解状态码": result.status,
        "求解信息": result.message,
        "目标函数最优值": objective_value,
        "迭代次数": getattr(result, "nit", np.nan),
    }])


def build_constraint_table(result):
    records = []
    if not result.success:
        return pd.DataFrame(records)

    if A_UB is not None:
        lhs_values = A_UB @ result.x
        for i, lhs in enumerate(lhs_values):
            name = INEQUALITY_NAMES[i] if i < len(INEQUALITY_NAMES) else f"不等式约束{i + 1}"
            records.append({
                "约束名称": name,
                "约束类型": "<=",
                "左端值": lhs,
                "右端值": B_UB[i],
                "松弛量": B_UB[i] - lhs,
            })

    if A_EQ is not None:
        lhs_values = A_EQ @ result.x
        for i, lhs in enumerate(lhs_values):
            name = EQUALITY_NAMES[i] if i < len(EQUALITY_NAMES) else f"等式约束{i + 1}"
            records.append({
                "约束名称": name,
                "约束类型": "==",
                "左端值": lhs,
                "右端值": B_EQ[i],
                "松弛量": lhs - B_EQ[i],
            })

    return pd.DataFrame(records)


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_solution_bar(solution_df):
    if solution_df.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(data=solution_df, x="变量", y="最优取值", color="#3A6EA5")
    plt.title("线性规划最优决策变量")
    plt.xlabel("变量")
    plt.ylabel("最优取值")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_programming_solution_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_constraint_slack(constraint_df):
    if constraint_df.empty or "松弛量" not in constraint_df.columns:
        return

    plt.figure(figsize=(9, max(4, len(constraint_df) * 0.45)))
    sns.barplot(data=constraint_df, x="松弛量", y="约束名称", color="#1B9E77")
    plt.axvline(0, color="#D95F02", linestyle="--", linewidth=1)
    plt.title("线性规划约束松弛量")
    plt.xlabel("松弛量")
    plt.ylabel("约束")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_programming_constraint_slack.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = solve_linear_programming()
    solution_df = build_solution_table(result)
    objective_df = build_objective_table(result)
    constraint_df = build_constraint_table(result)

    save_table(solution_df, OUTPUT_DIR / "01_solution_variables.csv")
    save_table(objective_df, OUTPUT_DIR / "02_objective_result.csv")
    save_table(constraint_df, OUTPUT_DIR / "03_constraint_slack.csv")

    plot_solution_bar(solution_df)
    plot_constraint_slack(constraint_df)

    print("Linear programming objective result:")
    print(objective_df)
    print("\nSolution variables:")
    print(solution_df)
    print(f"\nLinear programming workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
