"""
Entropy Weight Method + TOPSIS full workflow template.

Workflow:
1. Read CSV or Excel data.
2. Fill numeric missing values with median.
3. Transform all indicators into positive indicators.
4. Apply min-max normalization for entropy weight calculation.
5. Calculate entropy weights.
6. Apply vector normalization for TOPSIS.
7. Calculate positive/negative ideal solutions, closeness, and ranking.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, ID_COLUMN, and INDICATORS.
    3. Run:
       python scripts/02综合评价与权重方法/熵权法加TOPSIS全过程.py
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


INPUT_FILE = Path("data/your_data.csv")
OUTPUT_DIR = Path("outputs/02综合评价与权重方法/熵权法加TOPSIS")

# Optional ID/name column. Set to None if there is no ID column.
ID_COLUMN = "方案"

# direction:
#   "positive": larger is better
#   "negative": smaller is better
#   "moderate": closer to target is better, requires "target"
INDICATORS = {
    "指标1": {"direction": "positive"},
    "指标2": {"direction": "negative"},
    "指标3": {"direction": "moderate", "target": 60},
}

EPS = 1e-12


def setup_plot_style():
    设置科研绘图风格(plt, sns)


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def check_columns(df, indicators):
    missing = [col for col in indicators if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")


def fill_numeric_missing(df, cols):
    result = df.copy()
    fill_log = []

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        fill_value = x.median()
        before_missing = int(x.isna().sum())
        result[col] = x.fillna(fill_value)
        fill_log.append({
            "指标": col,
            "填补方法": "median",
            "填补值": fill_value,
            "填补前缺失数量": before_missing,
            "填补后缺失数量": int(result[col].isna().sum()),
        })

    return result, pd.DataFrame(fill_log)


def positive_transform(df, indicators):
    result = df.copy()

    for col, config in indicators.items():
        x = pd.to_numeric(result[col], errors="coerce")
        direction = config.get("direction", "positive")

        if direction == "positive":
            result[col] = x
        elif direction == "negative":
            result[col] = x.max() - x
        elif direction == "moderate":
            if "target" not in config:
                raise ValueError(f"Moderate indicator {col} needs a target value.")
            target = config["target"]
            max_distance = (x - target).abs().max()
            result[col] = max_distance - (x - target).abs()
        else:
            raise ValueError(f"Unsupported direction for {col}: {direction}")

    return result


def minmax_normalize(df, cols):
    result = df.copy()

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        value_range = x.max() - x.min()
        result[col] = (x - x.min()) / value_range if value_range != 0 else 0

    return result


def vector_normalize(df, cols):
    result = df.copy()

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        denominator = np.sqrt(np.sum(x ** 2))
        result[col] = x / denominator if denominator != 0 else 0

    return result


def calculate_entropy_weights(normalized_df, cols):
    x = normalized_df[cols].to_numpy(dtype=float)
    x = np.where(np.isnan(x), 0, x)

    column_sums = x.sum(axis=0)
    p = x / (column_sums + EPS)

    n_samples = x.shape[0]
    if n_samples <= 1:
        raise ValueError("Entropy weight method requires at least two samples.")

    k = 1 / np.log(n_samples)
    entropy = -k * np.sum(p * np.log(p + EPS), axis=0)
    difference = 1 - entropy

    if np.allclose(difference.sum(), 0):
        weights = np.ones_like(difference) / len(difference)
    else:
        weights = difference / difference.sum()

    weight_df = pd.DataFrame({
        "指标": cols,
        "信息熵": entropy,
        "差异系数": difference,
        "权重": weights,
    }).sort_values("权重", ascending=False)

    return weight_df


def calculate_topsis(vector_normalized_df, cols, weight_df):
    weight_vector = weight_df.set_index("指标").loc[cols, "权重"].to_numpy(dtype=float)
    normalized_matrix = vector_normalized_df[cols].to_numpy(dtype=float)
    weighted_matrix = normalized_matrix * weight_vector

    positive_ideal = weighted_matrix.max(axis=0)
    negative_ideal = weighted_matrix.min(axis=0)

    distance_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
    closeness = distance_negative / (distance_positive + distance_negative + EPS)

    score_df = vector_normalized_df.copy()
    score_df["正理想解距离"] = distance_positive
    score_df["负理想解距离"] = distance_negative
    score_df["TOPSIS贴近度"] = closeness
    score_df["排名"] = score_df["TOPSIS贴近度"].rank(ascending=False, method="min").astype(int)

    weighted_df = pd.DataFrame(weighted_matrix, columns=cols, index=vector_normalized_df.index)
    ideal_df = pd.DataFrame({
        "指标": cols,
        "熵权法权重": weight_vector,
        "正理想解": positive_ideal,
        "负理想解": negative_ideal,
    })

    return score_df.sort_values("排名"), weighted_df, ideal_df


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_entropy_weight_bar(weight_df):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=weight_df, x="权重", y="指标", color=科研配色["深蓝"])
    plt.title("熵权法指标权重")
    plt.xlabel("权重")
    plt.ylabel("指标")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_weight_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_topsis_ranking(score_df):
    plot_df = score_df.copy()
    if ID_COLUMN and ID_COLUMN in plot_df.columns:
        plot_df["方案名称"] = plot_df[ID_COLUMN]
    else:
        plot_df["方案名称"] = plot_df.index.astype(str)

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="TOPSIS贴近度", y="方案名称", color=科研配色["浅蓝"])
    plt.title("熵权法-TOPSIS综合评价排序")
    plt.xlabel("贴近度")
    plt.ylabel("方案")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_topsis_ranking_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_distance_scatter(score_df):
    plot_df = score_df.copy()
    if ID_COLUMN and ID_COLUMN in plot_df.columns:
        labels = plot_df[ID_COLUMN].astype(str).tolist()
    else:
        labels = plot_df.index.astype(str).tolist()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="正理想解距离",
        y="负理想解距离",
        size="TOPSIS贴近度",
        sizes=(60, 260),
        color=科研配色["深蓝"],
        legend=False,
    )
    for i, label in enumerate(labels):
        plt.text(
            plot_df["正理想解距离"].iloc[i],
            plot_df["负理想解距离"].iloc[i],
            label,
            fontsize=9,
        )
    plt.title("熵权法-TOPSIS正负理想解距离分布")
    plt.xlabel("到正理想解的距离")
    plt.ylabel("到负理想解的距离")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_topsis_distance_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_normalized_heatmap(normalized_df, cols):
    plt.figure(figsize=(max(8, len(cols) * 0.85), max(5, len(normalized_df) * 0.28)))
    sns.heatmap(
        normalized_df[cols],
        cmap=科研顺序色带,
        linewidths=0.2,
        cbar_kws={"label": "归一化值"},
    )
    plt.title("熵权法-TOPSIS归一化指标矩阵")
    plt.xlabel("指标")
    plt.ylabel("样本序号")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_topsis_normalized_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def entropy_topsis_workflow(df, indicators):
    cols = list(indicators.keys())
    check_columns(df, indicators)

    filled_df, fill_log = fill_numeric_missing(df, cols)
    positive_df = positive_transform(filled_df, indicators)

    entropy_normalized_df = minmax_normalize(positive_df, cols)
    weight_df = calculate_entropy_weights(entropy_normalized_df, cols)

    topsis_normalized_df = vector_normalize(positive_df, cols)
    score_df, weighted_df, ideal_df = calculate_topsis(topsis_normalized_df, cols, weight_df)

    return {
        "filled_df": filled_df,
        "fill_log": fill_log,
        "positive_df": positive_df,
        "entropy_normalized_df": entropy_normalized_df,
        "topsis_normalized_df": topsis_normalized_df,
        "weight_df": weight_df,
        "score_df": score_df,
        "weighted_df": weighted_df,
        "ideal_df": ideal_df,
    }


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, ID_COLUMN, and INDICATORS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    results = entropy_topsis_workflow(raw_df, INDICATORS)

    save_table(results["filled_df"], OUTPUT_DIR / "01_missing_filled_data.csv")
    save_table(results["fill_log"], OUTPUT_DIR / "02_missing_fill_log.csv")
    save_table(results["positive_df"], OUTPUT_DIR / "03_positive_transformed_data.csv")
    save_table(results["entropy_normalized_df"], OUTPUT_DIR / "04_entropy_minmax_normalized_data.csv")
    save_table(results["topsis_normalized_df"], OUTPUT_DIR / "05_topsis_vector_normalized_data.csv")
    save_table(results["weight_df"], OUTPUT_DIR / "06_entropy_weight_result.csv")
    save_table(results["weighted_df"].reset_index(drop=True), OUTPUT_DIR / "07_topsis_weighted_matrix.csv")
    save_table(results["ideal_df"], OUTPUT_DIR / "08_topsis_ideal_solution.csv")
    save_table(results["score_df"], OUTPUT_DIR / "09_entropy_topsis_score_ranking.csv")

    plot_entropy_weight_bar(results["weight_df"])
    plot_topsis_ranking(results["score_df"])
    plot_distance_scatter(results["score_df"])
    plot_normalized_heatmap(results["entropy_normalized_df"], list(INDICATORS.keys()))

    print("Entropy weight result:")
    print(results["weight_df"])
    print("\nEntropy-TOPSIS ranking result:")
    print(results["score_df"])
    print(f"\nEntropy-TOPSIS workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
