"""
Entropy Weight Method full workflow template.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, INDICATORS, and ID_COLUMN.
    3. Run:
       python scripts/二/熵权法全过程.py
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
OUTPUT_DIR = Path("outputs/二/熵权法")

# Optional ID/name column. Set to None if there is no ID column.
ID_COLUMN = "样本"

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


def fill_numeric_missing(df, cols):
    result = df.copy()
    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        result[col] = x.fillna(x.median())
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

    result = pd.DataFrame({
        "指标": cols,
        "信息熵": entropy,
        "差异系数": difference,
        "权重": weights,
    }).sort_values("权重", ascending=False)

    return result


def calculate_comprehensive_score(normalized_df, cols, weight_df):
    result = normalized_df.copy()
    weights = weight_df.set_index("指标").loc[cols, "权重"].to_numpy()
    score_matrix = result[cols].to_numpy(dtype=float)
    result["熵权法综合得分"] = score_matrix @ weights
    result["排名"] = result["熵权法综合得分"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("排名")


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_weight_bar(weight_df):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=weight_df, x="权重", y="指标", color=科研配色["深蓝"])
    plt.title("熵权法指标权重")
    plt.xlabel("权重")
    plt.ylabel("指标")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_weight_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_score_ranking(score_df):
    if "熵权法综合得分" not in score_df.columns:
        return

    plot_df = score_df.copy()
    if ID_COLUMN and ID_COLUMN in plot_df.columns:
        plot_df["样本名称"] = plot_df[ID_COLUMN]
    else:
        plot_df["样本名称"] = plot_df.index.astype(str)

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="熵权法综合得分", y="样本名称", color=科研配色["浅蓝"])
    plt.title("熵权法综合得分排序")
    plt.xlabel("综合得分")
    plt.ylabel("样本")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_score_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_normalized_heatmap(normalized_df, cols):
    plt.figure(figsize=(max(8, len(cols) * 0.85), max(5, len(normalized_df) * 0.28)))
    sns.heatmap(
        normalized_df[cols],
        cmap=科研顺序色带,
        linewidths=0.2,
        cbar_kws={"label": "归一化值"},
    )
    plt.title("熵权法归一化指标矩阵")
    plt.xlabel("指标")
    plt.ylabel("样本序号")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_normalized_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def entropy_weight_workflow(df, indicators):
    cols = list(indicators.keys())
    check_columns(df, indicators)

    numeric_df = fill_numeric_missing(df, cols)
    positive_df = positive_transform(numeric_df, indicators)
    normalized_df = minmax_normalize(positive_df, cols)
    weight_df = calculate_entropy_weights(normalized_df, cols)
    score_df = calculate_comprehensive_score(normalized_df, cols, weight_df)

    return positive_df, normalized_df, weight_df, score_df


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE and INDICATORS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    positive_df, normalized_df, weight_df, score_df = entropy_weight_workflow(raw_df, INDICATORS)

    save_table(positive_df, OUTPUT_DIR / "entropy_positive_transformed.csv")
    save_table(normalized_df, OUTPUT_DIR / "entropy_minmax_normalized.csv")
    save_table(weight_df, OUTPUT_DIR / "entropy_weight_result.csv")
    save_table(score_df, OUTPUT_DIR / "entropy_score_result.csv")

    plot_weight_bar(weight_df)
    plot_score_ranking(score_df)
    plot_normalized_heatmap(normalized_df, list(INDICATORS.keys()))

    print("Entropy weight result:")
    print(weight_df)
    print(f"\nEntropy weight workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
