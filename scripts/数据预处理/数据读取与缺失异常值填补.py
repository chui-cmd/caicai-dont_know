"""
Data reading, missing value imputation, and outlier imputation template.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, OUTPUT_DIR, and COLUMN_RULES.
    3. Run:
       python scripts/数据预处理/数据读取与缺失异常值填补.py
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
OUTPUT_DIR = Path("outputs/data_cleaning")

# Fill strategy:
#   "mean": mean value, suitable for near-normal numeric data
#   "median": median value, suitable for skewed numeric data
#   "mode": most frequent value, suitable for categorical/discrete data
#   "constant": fixed value, requires "fill_value"
#
# Outlier method:
#   "iqr": Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
#   "zscore": values with abs(z-score) > threshold
#   None: do not process outliers
#
# Outlier action:
#   "median": replace outliers with median
#   "mean": replace outliers with mean
#   "clip": clip outliers to lower/upper bound
#   "nan": replace outliers with NaN, then fill by missing strategy
COLUMN_RULES = {
    "指标1": {
        "fill_strategy": "mean",
        "outlier_method": "iqr",
        "outlier_action": "median",
    },
    "指标2": {
        "fill_strategy": "median",
        "outlier_method": "zscore",
        "zscore_threshold": 3,
        "outlier_action": "clip",
    },
    "类别": {
        "fill_strategy": "mode",
        "outlier_method": None,
    },
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


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def summarize_missing_values(df):
    summary = pd.DataFrame({
        "字段名": df.columns,
        "缺失数量": df.isna().sum().values,
        "缺失比例": df.isna().mean().values,
        "数据类型": [str(dtype) for dtype in df.dtypes],
    })
    return summary.sort_values("缺失比例", ascending=False)


def get_fill_value(series, strategy, fill_value=None):
    if strategy == "mean":
        return pd.to_numeric(series, errors="coerce").mean()
    if strategy == "median":
        return pd.to_numeric(series, errors="coerce").median()
    if strategy == "mode":
        mode_value = series.mode(dropna=True)
        return mode_value.iloc[0] if not mode_value.empty else fill_value
    if strategy == "constant":
        return fill_value
    raise ValueError(f"Unsupported fill strategy: {strategy}")


def fill_missing_values(df, column_rules):
    result = df.copy()
    records = []

    for col, rule in column_rules.items():
        if col not in result.columns:
            print(f"Warning: column not found and skipped: {col}")
            continue

        before_missing = result[col].isna().sum()
        strategy = rule.get("fill_strategy", "median")
        fill_value = get_fill_value(result[col], strategy, rule.get("fill_value"))
        result[col] = result[col].fillna(fill_value)
        after_missing = result[col].isna().sum()

        records.append({
            "字段名": col,
            "填补方法": strategy,
            "填补值": fill_value,
            "填补前缺失数量": before_missing,
            "填补后缺失数量": after_missing,
        })

    return result, pd.DataFrame(records)


def calculate_iqr_bounds(series):
    x = pd.to_numeric(series, errors="coerce")
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def calculate_zscore_bounds(series, threshold):
    x = pd.to_numeric(series, errors="coerce")
    mean_value = x.mean()
    std_value = x.std(ddof=0)
    if std_value == 0 or pd.isna(std_value):
        return mean_value, mean_value
    return mean_value - threshold * std_value, mean_value + threshold * std_value


def replace_outliers(series, mask, lower, upper, action):
    x = pd.to_numeric(series, errors="coerce")
    if action == "median":
        replacement = x[~mask].median()
        x.loc[mask] = replacement
    elif action == "mean":
        replacement = x[~mask].mean()
        x.loc[mask] = replacement
    elif action == "clip":
        x = x.clip(lower=lower, upper=upper)
    elif action == "nan":
        x.loc[mask] = np.nan
    else:
        raise ValueError(f"Unsupported outlier action: {action}")
    return x


def process_outliers(df, column_rules):
    result = df.copy()
    records = []

    for col, rule in column_rules.items():
        if col not in result.columns:
            continue

        method = rule.get("outlier_method")
        if method is None:
            continue

        x = pd.to_numeric(result[col], errors="coerce")
        if x.notna().sum() == 0:
            continue

        if method == "iqr":
            lower, upper = calculate_iqr_bounds(x)
        elif method == "zscore":
            threshold = rule.get("zscore_threshold", 3)
            lower, upper = calculate_zscore_bounds(x, threshold)
        else:
            raise ValueError(f"Unsupported outlier method: {method}")

        mask = (x < lower) | (x > upper)
        outlier_count = int(mask.sum())
        action = rule.get("outlier_action", "median")
        result[col] = replace_outliers(x, mask, lower, upper, action)

        if action == "nan":
            fill_strategy = rule.get("fill_strategy", "median")
            fill_value = get_fill_value(result[col], fill_strategy, rule.get("fill_value"))
            result[col] = result[col].fillna(fill_value)

        records.append({
            "字段名": col,
            "异常值方法": method,
            "处理方式": action,
            "下界": lower,
            "上界": upper,
            "异常值数量": outlier_count,
            "异常值比例": outlier_count / len(result) if len(result) else 0,
        })

    return result, pd.DataFrame(records)


def plot_missing_summary(missing_summary, output_path):
    plot_df = missing_summary[missing_summary["缺失数量"] > 0].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="缺失比例", y="字段名", color=科研配色["深蓝"])
    plt.title("各字段缺失值比例")
    plt.xlabel("缺失比例")
    plt.ylabel("字段名")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_numeric_boxplots(before_df, after_df, column_rules, output_path):
    numeric_cols = [
        col for col in column_rules
        if col in before_df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(before_df[col], errors="coerce"))
    ]
    if not numeric_cols:
        return

    plot_data = []
    for col in numeric_cols:
        plot_data.append(pd.DataFrame({
            "字段名": col,
            "阶段": "处理前",
            "数值": pd.to_numeric(before_df[col], errors="coerce"),
        }))
        plot_data.append(pd.DataFrame({
            "字段名": col,
            "阶段": "处理后",
            "数值": pd.to_numeric(after_df[col], errors="coerce"),
        }))

    plot_df = pd.concat(plot_data, ignore_index=True)

    plt.figure(figsize=(max(9, len(numeric_cols) * 1.2), 6))
    sns.boxplot(data=plot_df, x="字段名", y="数值", hue="阶段", palette=[科研配色["红"], 科研配色["浅蓝"]])
    plt.title("异常值处理前后箱线图对比")
    plt.xlabel("字段名")
    plt.ylabel("数值")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def clean_data(df, column_rules):
    missing_summary = summarize_missing_values(df)
    missing_filled_df, fill_log = fill_missing_values(df, column_rules)
    cleaned_df, outlier_log = process_outliers(missing_filled_df, column_rules)
    return cleaned_df, missing_summary, fill_log, outlier_log


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE and COLUMN_RULES before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    cleaned_df, missing_summary, fill_log, outlier_log = clean_data(raw_df, COLUMN_RULES)

    save_table(cleaned_df, OUTPUT_DIR / "cleaned_data.csv")
    save_table(missing_summary, OUTPUT_DIR / "missing_value_summary.csv")
    save_table(fill_log, OUTPUT_DIR / "missing_value_fill_log.csv")
    save_table(outlier_log, OUTPUT_DIR / "outlier_process_log.csv")

    plot_missing_summary(missing_summary, OUTPUT_DIR / "missing_value_ratio.png")
    plot_numeric_boxplots(raw_df, cleaned_df, COLUMN_RULES, OUTPUT_DIR / "outlier_boxplot_compare.png")

    print(f"Data cleaning completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
