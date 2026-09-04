"""
Indicator preprocessing template for mathematical modeling contests.

Features:
1. Positive transformation for negative and moderate indicators.
2. Z-score standardization.
3. Min-max normalization.
4. Heatmap and distribution comparison plots.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE and INDICATORS below.
    3. Run:
       python scripts/数据预处理/指标预处理模板.py
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
OUTPUT_DIR = Path("outputs/indicator_preprocess")

# Edit this dictionary after you get the real data.
# direction:
#   "positive" means larger is better.
#   "negative" means smaller is better.
#   "moderate" means closer to target is better.
INDICATORS = {
    "指标1": {"direction": "positive"},
    "指标2": {"direction": "negative"},
    "指标3": {"direction": "moderate", "target": 60},
}


def setup_plot_style():
    设置科研绘图风格(plt, sns)


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def check_indicators(df, indicators):
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


def zscore_standardize(df, cols):
    result = df.copy()

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        std = x.std(ddof=0)
        result[col] = (x - x.mean()) / std if std != 0 else 0

    return result


def minmax_normalize(df, cols):
    result = df.copy()

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        value_range = x.max() - x.min()
        result[col] = (x - x.min()) / value_range if value_range != 0 else 0

    return result


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_heatmap(df, cols, title, output_path):
    plt.figure(figsize=(9, 6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=科研连续色带, center=0, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution_compare(original_df, positive_df, zscore_df, minmax_df, cols, output_path):
    plot_data = []

    for col in cols:
        for label, source in [
            ("原始值", original_df),
            ("正向化", positive_df),
            ("Z-score", zscore_df),
            ("极差标准化", minmax_df),
        ]:
            temp = pd.DataFrame({
                "指标": col,
                "处理方式": label,
                "数值": pd.to_numeric(source[col], errors="coerce"),
            })
            plot_data.append(temp)

    plot_df = pd.concat(plot_data, ignore_index=True)

    g = sns.FacetGrid(
        plot_df,
        row="指标",
        col="处理方式",
        sharex=False,
        sharey=False,
        height=2.6,
        aspect=1.25,
    )
    g.map_dataframe(sns.histplot, x="数值", kde=True, color=科研配色["深蓝"])
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.figure.suptitle("指标预处理前后分布对比", y=1.02, fontsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_indicators(df, indicators):
    cols = list(indicators.keys())
    check_indicators(df, indicators)

    positive_df = positive_transform(df, indicators)
    zscore_df = zscore_standardize(positive_df, cols)
    minmax_df = minmax_normalize(positive_df, cols)

    return positive_df, zscore_df, minmax_df


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE and INDICATORS before running this script.")
        return

    df = read_table(INPUT_FILE)
    cols = list(INDICATORS.keys())

    positive_df, zscore_df, minmax_df = preprocess_indicators(df, INDICATORS)

    save_table(positive_df, OUTPUT_DIR / "positive_transformed.csv")
    save_table(zscore_df, OUTPUT_DIR / "zscore_standardized.csv")
    save_table(minmax_df, OUTPUT_DIR / "minmax_normalized.csv")

    plot_heatmap(
        minmax_df,
        cols,
        "极差标准化后指标相关性热力图",
        OUTPUT_DIR / "minmax_correlation_heatmap.png",
    )
    plot_distribution_compare(
        df,
        positive_df,
        zscore_df,
        minmax_df,
        cols,
        OUTPUT_DIR / "preprocess_distribution_compare.png",
    )

    print(f"Preprocessing completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
