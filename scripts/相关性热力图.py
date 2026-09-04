"""
Pearson and Spearman correlation heatmap template for mathematical modeling.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE and INDICATOR_COLUMNS.
    3. Run:
       python scripts/相关性热力图.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("data/your_data.csv")
OUTPUT_DIR = Path("outputs/correlation_heatmap")

# Edit these column names after you get the real data.
# If this list is empty, all numeric columns will be used automatically.
INDICATOR_COLUMNS = [
    "指标1",
    "指标2",
    "指标3",
]

# Optional: set an ID/name column to display row labels in exported cleaned data.
ID_COLUMN = None


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="SimHei")


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def get_analysis_data(df, indicator_columns):
    if indicator_columns:
        missing = [col for col in indicator_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in data file: {missing}")
        analysis_df = df[indicator_columns].copy()
    else:
        analysis_df = df.select_dtypes(include="number").copy()

    for col in analysis_df.columns:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors="coerce")

    analysis_df = analysis_df.dropna(axis=1, how="all")
    if analysis_df.shape[1] < 2:
        raise ValueError("At least two numeric indicator columns are required.")

    return analysis_df


def calculate_correlation(df, method):
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    return df.corr(method=method)


def plot_correlation_heatmap(corr, title, output_path):
    n_cols = len(corr.columns)
    fig_size = max(7, n_cols * 0.8)

    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.82, "label": "相关系数"},
    )
    plt.title(title, fontsize=15, pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_correlation_table(corr, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(output_path, encoding="utf-8-sig")


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE and INDICATOR_COLUMNS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    analysis_df = get_analysis_data(raw_df, INDICATOR_COLUMNS)

    if ID_COLUMN and ID_COLUMN in raw_df.columns:
        cleaned_df = pd.concat([raw_df[[ID_COLUMN]], analysis_df], axis=1)
    else:
        cleaned_df = analysis_df
    cleaned_df.to_csv(OUTPUT_DIR / "correlation_input_data.csv", index=False, encoding="utf-8-sig")

    pearson_corr = calculate_correlation(analysis_df, method="pearson")
    spearman_corr = calculate_correlation(analysis_df, method="spearman")

    save_correlation_table(pearson_corr, OUTPUT_DIR / "pearson_correlation_matrix.csv")
    save_correlation_table(spearman_corr, OUTPUT_DIR / "spearman_correlation_matrix.csv")

    plot_correlation_heatmap(
        pearson_corr,
        "Pearson 相关性热力图",
        OUTPUT_DIR / "pearson_correlation_heatmap.png",
    )
    plot_correlation_heatmap(
        spearman_corr,
        "Spearman 相关性热力图",
        OUTPUT_DIR / "spearman_correlation_heatmap.png",
    )

    print(f"Correlation analysis completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
