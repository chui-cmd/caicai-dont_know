"""
TOPSIS ranking full workflow template.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, ID_COLUMN, INDICATORS, and WEIGHTS.
    3. Run:
       python scripts/二/TOPSIS排序全过程.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("data/your_data.csv")
OUTPUT_DIR = Path("outputs/二/TOPSIS排序")

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

# If all weights are None, equal weights will be used.
# You can also paste weights from AHP or entropy weight method here.
WEIGHTS = {
    "指标1": None,
    "指标2": None,
    "指标3": None,
}


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="SimHei")


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
    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        result[col] = x.fillna(x.median())
    return result


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


def vector_normalize(df, cols):
    result = df.copy()
    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        denominator = np.sqrt(np.sum(x ** 2))
        result[col] = x / denominator if denominator != 0 else 0
    return result


def minmax_normalize(df, cols):
    result = df.copy()
    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        value_range = x.max() - x.min()
        result[col] = (x - x.min()) / value_range if value_range != 0 else 0
    return result


def get_weight_vector(cols, weights):
    raw_weights = []
    use_equal_weight = all(weights.get(col) is None for col in cols)

    if use_equal_weight:
        return np.ones(len(cols)) / len(cols)

    for col in cols:
        value = weights.get(col)
        if value is None:
            raise ValueError(f"Weight for {col} is None. Fill all weights or set all to None.")
        raw_weights.append(float(value))

    weight_vector = np.array(raw_weights, dtype=float)
    if np.any(weight_vector < 0):
        raise ValueError("Weights must be non-negative.")
    if weight_vector.sum() == 0:
        raise ValueError("Weight sum must be greater than 0.")
    return weight_vector / weight_vector.sum()


def calculate_topsis(normalized_df, cols, weight_vector):
    score_matrix = normalized_df[cols].to_numpy(dtype=float)
    weighted_matrix = score_matrix * weight_vector

    positive_ideal = weighted_matrix.max(axis=0)
    negative_ideal = weighted_matrix.min(axis=0)

    distance_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
    closeness = distance_negative / (distance_positive + distance_negative)

    result = normalized_df.copy()
    result["正理想解距离"] = distance_positive
    result["负理想解距离"] = distance_negative
    result["TOPSIS贴近度"] = closeness
    result["排名"] = result["TOPSIS贴近度"].rank(ascending=False, method="min").astype(int)

    weighted_df = pd.DataFrame(weighted_matrix, columns=cols, index=normalized_df.index)
    ideal_df = pd.DataFrame({
        "指标": cols,
        "权重": weight_vector,
        "正理想解": positive_ideal,
        "负理想解": negative_ideal,
    })

    return result.sort_values("排名"), weighted_df, ideal_df


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_ranking(score_df):
    plot_df = score_df.copy()
    if ID_COLUMN and ID_COLUMN in plot_df.columns:
        plot_df["方案名称"] = plot_df[ID_COLUMN]
    else:
        plot_df["方案名称"] = plot_df.index.astype(str)

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="TOPSIS贴近度", y="方案名称", color="#1B9E77")
    plt.title("TOPSIS综合评价排序")
    plt.xlabel("贴近度")
    plt.ylabel("方案")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "topsis_ranking_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_distance_scatter(score_df):
    plot_df = score_df.copy()
    if ID_COLUMN and ID_COLUMN in plot_df.columns:
        labels = plot_df[ID_COLUMN].astype(str).tolist()
    else:
        labels = plot_df.index.astype(str).tolist()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="正理想解距离", y="负理想解距离", size="TOPSIS贴近度", sizes=(60, 260), color="#3A6EA5")
    for i, label in enumerate(labels):
        plt.text(plot_df["正理想解距离"].iloc[i], plot_df["负理想解距离"].iloc[i], label, fontsize=9)
    plt.title("TOPSIS正负理想解距离分布")
    plt.xlabel("到正理想解的距离")
    plt.ylabel("到负理想解的距离")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "topsis_distance_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_weight_bar(cols, weight_vector):
    weight_df = pd.DataFrame({"指标": cols, "权重": weight_vector}).sort_values("权重", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=weight_df, x="权重", y="指标", color="#3A6EA5")
    plt.title("TOPSIS指标权重")
    plt.xlabel("权重")
    plt.ylabel("指标")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "topsis_weight_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def topsis_workflow(df, indicators, weights):
    cols = list(indicators.keys())
    check_columns(df, indicators)

    filled_df = fill_numeric_missing(df, cols)
    positive_df = positive_transform(filled_df, indicators)
    normalized_df = vector_normalize(positive_df, cols)
    weight_vector = get_weight_vector(cols, weights)
    score_df, weighted_df, ideal_df = calculate_topsis(normalized_df, cols, weight_vector)

    return positive_df, normalized_df, score_df, weighted_df, ideal_df, weight_vector


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, INDICATORS, and WEIGHTS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    positive_df, normalized_df, score_df, weighted_df, ideal_df, weight_vector = topsis_workflow(
        raw_df,
        INDICATORS,
        WEIGHTS,
    )

    save_table(positive_df, OUTPUT_DIR / "topsis_positive_transformed.csv")
    save_table(normalized_df, OUTPUT_DIR / "topsis_vector_normalized.csv")
    save_table(score_df, OUTPUT_DIR / "topsis_score_result.csv")
    save_table(weighted_df.reset_index(drop=True), OUTPUT_DIR / "topsis_weighted_matrix.csv")
    save_table(ideal_df, OUTPUT_DIR / "topsis_ideal_solution.csv")

    plot_ranking(score_df)
    plot_distance_scatter(score_df)
    plot_weight_bar(list(INDICATORS.keys()), weight_vector)

    print("TOPSIS ranking result:")
    print(score_df)
    print(f"\nTOPSIS workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
