"""
Analytic Hierarchy Process (AHP) full workflow template.

Usage:
    1. Edit JUDGMENT_MATRIX and CRITERIA_NAMES.
    2. Optionally put alternative scores in data/ and edit INPUT_FILE.
    3. Run:
       python scripts/02综合评价与权重方法/层次分析法全过程.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from 科研绘图配色 import (
    科研调色板,
    科研配色,
    科研连续色带,
    科研顺序色带,
    设置科研绘图风格,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("outputs/02综合评价与权重方法/层次分析法")

# Pairwise judgment matrix. a_ij means criterion i is a_ij times as important as criterion j.
# Common scale: 1, 3, 5, 7, 9 and reciprocals 1/3, 1/5, 1/7, 1/9.
CRITERIA_NAMES = ["指标1", "指标2", "指标3"]
JUDGMENT_MATRIX = np.array([
    [1, 3, 5],
    [1 / 3, 1, 2],
    [1 / 5, 1 / 2, 1],
], dtype=float)

# Optional comprehensive scoring for alternatives.
# If INPUT_FILE does not exist, only criterion weights and consistency results are generated.
INPUT_FILE = Path("data/your_alternative_scores.csv")
ALTERNATIVE_NAME_COLUMN = "方案"
SCORE_COLUMNS = CRITERIA_NAMES

# If scores are raw indicators, set NEED_MINMAX_NORMALIZE = True.
# The template assumes larger normalized score means better.
NEED_MINMAX_NORMALIZE = True

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def setup_plot_style():
    设置科研绘图风格(plt, sns)


def check_judgment_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Judgment matrix must be a square matrix.")
    if np.any(matrix <= 0):
        raise ValueError("All elements in judgment matrix must be positive.")
    if not np.allclose(matrix * matrix.T, np.ones_like(matrix), atol=1e-6):
        raise ValueError("Judgment matrix must satisfy reciprocal property: a_ij * a_ji = 1.")
    return matrix


def calculate_ahp_weights(matrix):
    matrix = check_judgment_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_index].real
    weight_vector = eigenvectors[:, max_index].real
    weight_vector = np.abs(weight_vector)
    weights = weight_vector / weight_vector.sum()
    return lambda_max, weights


def consistency_test(matrix, lambda_max):
    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = RI_TABLE.get(n)
    if ri is None:
        raise ValueError("RI table supports matrix size 1 to 10.")
    cr = 0 if ri == 0 else ci / ri
    passed = cr < 0.1
    return {
        "矩阵阶数": n,
        "最大特征值": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "是否通过一致性检验": "通过" if passed else "未通过",
    }


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def minmax_normalize(df, cols):
    result = df.copy()
    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        value_range = x.max() - x.min()
        result[col] = (x - x.min()) / value_range if value_range != 0 else 0
    return result


def score_alternatives(df, score_columns, weights):
    missing = [col for col in score_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    score_df = df.copy()
    if NEED_MINMAX_NORMALIZE:
        score_df = minmax_normalize(score_df, score_columns)

    score_matrix = score_df[score_columns].apply(pd.to_numeric, errors="coerce")
    score_df["AHP综合得分"] = score_matrix.values @ weights
    score_df["排名"] = score_df["AHP综合得分"].rank(ascending=False, method="min").astype(int)
    return score_df.sort_values("排名")


def save_outputs(weights, criteria_names, consistency_result, judgment_matrix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weight_df = pd.DataFrame({
        "指标": criteria_names,
        "权重": weights,
    }).sort_values("权重", ascending=False)
    weight_df.to_csv(OUTPUT_DIR / "ahp_weight_result.csv", index=False, encoding="utf-8-sig")

    consistency_df = pd.DataFrame([consistency_result])
    consistency_df.to_csv(OUTPUT_DIR / "ahp_consistency_test.csv", index=False, encoding="utf-8-sig")

    matrix_df = pd.DataFrame(judgment_matrix, index=criteria_names, columns=criteria_names)
    matrix_df.to_csv(OUTPUT_DIR / "ahp_judgment_matrix.csv", encoding="utf-8-sig")

    return weight_df


def plot_weight_bar(weight_df):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=weight_df, x="权重", y="指标", color=科研配色["深蓝"])
    plt.title("层次分析法指标权重")
    plt.xlabel("权重")
    plt.ylabel("指标")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ahp_weight_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_judgment_heatmap(matrix, criteria_names):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        pd.DataFrame(matrix, index=criteria_names, columns=criteria_names),
        annot=True,
        fmt=".3g",
        cmap=科研顺序色带,
        linewidths=0.5,
        cbar_kws={"label": "相对重要性"},
    )
    plt.title("AHP判断矩阵热力图")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ahp_judgment_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_alternative_scores(score_df):
    if "AHP综合得分" not in score_df.columns:
        return
    name_col = ALTERNATIVE_NAME_COLUMN if ALTERNATIVE_NAME_COLUMN in score_df.columns else None
    plot_df = score_df.copy()
    plot_df["方案名称"] = plot_df[name_col] if name_col else plot_df.index.astype(str)

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="AHP综合得分", y="方案名称", color=科研配色["浅蓝"])
    plt.title("方案AHP综合得分排序")
    plt.xlabel("综合得分")
    plt.ylabel("方案")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ahp_alternative_score_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    setup_plot_style()
    if len(CRITERIA_NAMES) != JUDGMENT_MATRIX.shape[0]:
        raise ValueError("CRITERIA_NAMES length must match judgment matrix size.")

    lambda_max, weights = calculate_ahp_weights(JUDGMENT_MATRIX)
    consistency_result = consistency_test(JUDGMENT_MATRIX, lambda_max)
    weight_df = save_outputs(weights, CRITERIA_NAMES, consistency_result, JUDGMENT_MATRIX)

    plot_weight_bar(weight_df)
    plot_judgment_heatmap(JUDGMENT_MATRIX, CRITERIA_NAMES)

    print("AHP weight result:")
    print(weight_df)
    print("\nConsistency test:")
    print(pd.DataFrame([consistency_result]))

    if INPUT_FILE.exists():
        raw_df = read_table(INPUT_FILE)
        score_df = score_alternatives(raw_df, SCORE_COLUMNS, weights)
        score_df.to_csv(OUTPUT_DIR / "ahp_alternative_score_result.csv", index=False, encoding="utf-8-sig")
        plot_alternative_scores(score_df)
        print(f"\nAlternative scoring completed. Results saved to: {OUTPUT_DIR}")
    else:
        print(f"\nAlternative score file not found: {INPUT_FILE}")
        print(f"Only AHP weights and consistency results were saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
