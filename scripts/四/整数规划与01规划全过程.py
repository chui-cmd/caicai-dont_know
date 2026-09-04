"""
Integer programming and 0-1 programming full workflow template.

Solved by scipy.optimize.milp:
    minimize     c^T x
    subject to   b_l <= A x <= b_u
                 lower_i <= x_i <= upper_i
                 x_i can be continuous, integer, or binary

For maximization, set OBJECTIVE_SENSE = "max"; the script will transform it.

Usage:
    1. Edit VARIABLE_NAMES, OBJECTIVE_COEFFICIENTS, constraints, BOUNDS, and INTEGRALITY.
    2. Run:
       python scripts/四/整数规划与01规划全过程.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import Bounds, LinearConstraint, milp


OUTPUT_DIR = Path("outputs/四/整数规划与01规划")

# Example: mixed integer 0-1 selection problem.
VARIABLE_NAMES = ["选择项目1", "选择项目2", "选择项目3", "选择项目4"]

# Example: maximize total benefit.
OBJECTIVE_SENSE = "max"
OBJECTIVE_COEFFICIENTS = np.array([8, 6, 7, 9], dtype=float)

# Constraint format for scipy.milp:
#   lower_bounds <= A @ x <= upper_bounds
# For <= constraints, set lower bound to -np.inf.
# For >= constraints, set upper bound to np.inf.
# For == constraints, set lower bound equal to upper bound.
CONSTRAINT_MATRIX = np.array([
    [4, 2, 3, 5],
    [1, 1, 1, 1],
], dtype=float)
CONSTRAINT_LOWER_BOUNDS = np.array([-np.inf, -np.inf], dtype=float)
CONSTRAINT_UPPER_BOUNDS = np.array([9, 2], dtype=float)
CONSTRAINT_NAMES = ["资源约束", "最多选择数量"]

# Bounds for variables.
# For 0-1 programming, bounds should be 0 <= x <= 1 and integrality should be 1.
LOWER_BOUNDS = np.array([0, 0, 0, 0], dtype=float)
UPPER_BOUNDS = np.array([1, 1, 1, 1], dtype=float)

# Integrality:
#   0: continuous variable
#   1: integer variable
#   2: semi-continuous variable
#   3: semi-integer variable
# For 0-1 variables, use integrality=1 with bounds [0, 1].
INTEGRALITY = np.array([1, 1, 1, 1], dtype=int)


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="SimHei")


def check_model():
    n_vars = len(VARIABLE_NAMES)
    if len(OBJECTIVE_COEFFICIENTS) != n_vars:
        raise ValueError("OBJECTIVE_COEFFICIENTS length must match VARIABLE_NAMES.")
    if len(LOWER_BOUNDS) != n_vars or len(UPPER_BOUNDS) != n_vars:
        raise ValueError("Bounds length must match VARIABLE_NAMES.")
    if len(INTEGRALITY) != n_vars:
        raise ValueError("INTEGRALITY length must match VARIABLE_NAMES.")
    if CONSTRAINT_MATRIX.shape[1] != n_vars:
        raise ValueError("CONSTRAINT_MATRIX column count must match variable count.")
    if len(CONSTRAINT_LOWER_BOUNDS) != CONSTRAINT_MATRIX.shape[0]:
        raise ValueError("CONSTRAINT_LOWER_BOUNDS length must match constraint count.")
    if len(CONSTRAINT_UPPER_BOUNDS) != CONSTRAINT_MATRIX.shape[0]:
        raise ValueError("CONSTRAINT_UPPER_BOUNDS length must match constraint count.")
    if OBJECTIVE_SENSE not in ["min", "max"]:
        raise ValueError("OBJECTIVE_SENSE must be 'min' or 'max'.")


def solve_integer_programming():
    check_model()
    c = OBJECTIVE_COEFFICIENTS.copy()
    if OBJECTIVE_SENSE == "max":
        c = -c

    constraints = LinearConstraint(
        CONSTRAINT_MATRIX,
        lb=CONSTRAINT_LOWER_BOUNDS,
        ub=CONSTRAINT_UPPER_BOUNDS,
    )
    bounds = Bounds(lb=LOWER_BOUNDS, ub=UPPER_BOUNDS)

    result = milp(
        c=c,
        integrality=INTEGRALITY,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 300},
    )
    return result


def build_solution_table(result):
    if not result.success:
        return pd.DataFrame(columns=["变量", "最优取值", "变量类型"])

    variable_types = []
    for integrality, lower, upper in zip(INTEGRALITY, LOWER_BOUNDS, UPPER_BOUNDS):
        if integrality == 1 and lower == 0 and upper == 1:
            variable_types.append("0-1变量")
        elif integrality == 1:
            variable_types.append("整数变量")
        else:
            variable_types.append("连续变量")

    return pd.DataFrame({
        "变量": VARIABLE_NAMES,
        "最优取值": result.x,
        "变量类型": variable_types,
        "目标函数系数": OBJECTIVE_COEFFICIENTS,
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
        "节点数": getattr(result, "mip_node_count", np.nan),
        "最优性间隙": getattr(result, "mip_gap", np.nan),
    }])


def build_constraint_table(result):
    records = []
    if not result.success:
        return pd.DataFrame(records)

    lhs_values = CONSTRAINT_MATRIX @ result.x
    for i, lhs in enumerate(lhs_values):
        name = CONSTRAINT_NAMES[i] if i < len(CONSTRAINT_NAMES) else f"约束{i + 1}"
        lower = CONSTRAINT_LOWER_BOUNDS[i]
        upper = CONSTRAINT_UPPER_BOUNDS[i]

        if np.isneginf(lower) and not np.isposinf(upper):
            relation = "<="
            slack = upper - lhs
            rhs_text = upper
        elif not np.isneginf(lower) and np.isposinf(upper):
            relation = ">="
            slack = lhs - lower
            rhs_text = lower
        elif lower == upper:
            relation = "=="
            slack = lhs - lower
            rhs_text = lower
        else:
            relation = "区间约束"
            slack = min(lhs - lower, upper - lhs)
            rhs_text = f"[{lower}, {upper}]"

        records.append({
            "约束名称": name,
            "约束类型": relation,
            "左端值": lhs,
            "右端值": rhs_text,
            "松弛量": slack,
        })

    return pd.DataFrame(records)


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_solution_bar(solution_df):
    if solution_df.empty:
        return

    plt.figure(figsize=(9, 5))
    sns.barplot(data=solution_df, x="变量", y="最优取值", hue="变量类型")
    plt.title("整数规划/0-1规划最优决策变量")
    plt.xlabel("变量")
    plt.ylabel("最优取值")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "integer_programming_solution_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_constraint_slack(constraint_df):
    if constraint_df.empty or "松弛量" not in constraint_df.columns:
        return

    plt.figure(figsize=(9, max(4, len(constraint_df) * 0.45)))
    sns.barplot(data=constraint_df, x="松弛量", y="约束名称", color="#1B9E77")
    plt.axvline(0, color="#D95F02", linestyle="--", linewidth=1)
    plt.title("整数规划/0-1规划约束松弛量")
    plt.xlabel("松弛量")
    plt.ylabel("约束")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "integer_programming_constraint_slack.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = solve_integer_programming()
    solution_df = build_solution_table(result)
    objective_df = build_objective_table(result)
    constraint_df = build_constraint_table(result)

    save_table(solution_df, OUTPUT_DIR / "01_solution_variables.csv")
    save_table(objective_df, OUTPUT_DIR / "02_objective_result.csv")
    save_table(constraint_df, OUTPUT_DIR / "03_constraint_slack.csv")

    plot_solution_bar(solution_df)
    plot_constraint_slack(constraint_df)

    print("Integer programming objective result:")
    print(objective_df)
    print("\nSolution variables:")
    print(solution_df)
    print(f"\nInteger/0-1 programming workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
