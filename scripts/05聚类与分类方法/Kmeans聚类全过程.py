"""
K-means clustering full workflow template.

Outputs:
1. Cleaned and standardized feature data.
2. Elbow curve and silhouette score curve for choosing K.
3. K-means clustering labels and cluster centers.
4. PCA 2D visualization of clustering result.
5. Cluster feature profile heatmap.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, ID_COLUMN, FEATURE_COLUMNS, and K_VALUE.
    3. Run:
       python scripts/05聚类与分类方法/Kmeans聚类全过程.py
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
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


INPUT_FILE = Path("data/your_cluster_data.csv")
OUTPUT_DIR = Path("outputs/05聚类与分类方法/Kmeans聚类")

# Optional ID/name column. Set to None if there is no ID column.
ID_COLUMN = "样本"

# If this list is empty, all numeric columns will be used automatically.
FEATURE_COLUMNS = [
    "特征1",
    "特征2",
    "特征3",
]

K_VALUE = 3
K_RANGE = range(2, 11)
RANDOM_STATE = 42


def setup_plot_style():
    设置科研绘图风格(plt, sns)


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def get_feature_data(df):
    if FEATURE_COLUMNS:
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        feature_df = df[FEATURE_COLUMNS].copy()
    else:
        feature_df = df.select_dtypes(include="number").copy()

    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    feature_df = feature_df.dropna(axis=1, how="all")
    if feature_df.shape[1] < 2:
        raise ValueError("At least two numeric feature columns are required.")

    return feature_df


def standardize_features(feature_df):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled_values, columns=feature_df.columns, index=feature_df.index)
    return scaled_df, scaler


def evaluate_k_values(scaled_df):
    records = []
    n_samples = len(scaled_df)

    for k in K_RANGE:
        if k >= n_samples:
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_df)
        records.append({
            "K值": k,
            "SSE": model.inertia_,
            "轮廓系数": silhouette_score(scaled_df, labels),
        })

    return pd.DataFrame(records)


def fit_kmeans(scaled_df):
    if K_VALUE < 2:
        raise ValueError("K_VALUE must be at least 2.")
    if K_VALUE >= len(scaled_df):
        raise ValueError("K_VALUE must be smaller than sample size.")

    model = KMeans(n_clusters=K_VALUE, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(scaled_df)
    return model, labels


def build_result_table(raw_df, feature_df, scaled_df, labels):
    result_df = pd.DataFrame()
    if ID_COLUMN and ID_COLUMN in raw_df.columns:
        result_df[ID_COLUMN] = raw_df[ID_COLUMN]

    for col in feature_df.columns:
        result_df[col] = feature_df[col]
    for col in scaled_df.columns:
        result_df[f"标准化_{col}"] = scaled_df[col]

    result_df["聚类类别"] = labels + 1
    return result_df


def build_center_tables(model, feature_df, scaler):
    scaled_centers = pd.DataFrame(
        model.cluster_centers_,
        columns=feature_df.columns,
    )
    scaled_centers.insert(0, "聚类类别", range(1, len(scaled_centers) + 1))

    original_centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=feature_df.columns,
    )
    original_centers.insert(0, "聚类类别", range(1, len(original_centers) + 1))

    return scaled_centers, original_centers


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_k_selection(k_eval_df):
    if k_eval_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(data=k_eval_df, x="K值", y="SSE", marker="o", ax=axes[0], color=科研配色["深蓝"])
    axes[0].set_title("肘部法选择K值")
    axes[0].set_xlabel("K值")
    axes[0].set_ylabel("SSE")

    sns.lineplot(data=k_eval_df, x="K值", y="轮廓系数", marker="o", ax=axes[1], color=科研配色["浅蓝"])
    axes[1].set_title("轮廓系数选择K值")
    axes[1].set_xlabel("K值")
    axes[1].set_ylabel("轮廓系数")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "k_selection_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_cluster(scaled_df, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(scaled_df)
    plot_df = pd.DataFrame({
        "主成分1": pca_values[:, 0],
        "主成分2": pca_values[:, 1],
        "聚类类别": labels + 1,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="主成分1", y="主成分2", hue="聚类类别", palette=科研调色板, s=80)
    plt.title("K-means聚类结果PCA二维展示")
    plt.xlabel(f"主成分1 解释方差: {pca.explained_variance_ratio_[0]:.2%}")
    plt.ylabel(f"主成分2 解释方差: {pca.explained_variance_ratio_[1]:.2%}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_pca_cluster.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_profile(scaled_centers):
    profile_df = scaled_centers.set_index("聚类类别")

    plt.figure(figsize=(max(8, profile_df.shape[1] * 0.8), max(4, profile_df.shape[0] * 0.7)))
    sns.heatmap(profile_df, annot=True, fmt=".2f", cmap=科研连续色带, center=0, linewidths=0.5)
    plt.title("各聚类中心标准化特征画像")
    plt.xlabel("特征")
    plt.ylabel("聚类类别")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_profile_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def kmeans_workflow(df):
    feature_df = get_feature_data(df)
    scaled_df, scaler = standardize_features(feature_df)
    k_eval_df = evaluate_k_values(scaled_df)
    model, labels = fit_kmeans(scaled_df)
    result_df = build_result_table(df, feature_df, scaled_df, labels)
    scaled_centers, original_centers = build_center_tables(model, feature_df, scaler)

    return {
        "feature_df": feature_df,
        "scaled_df": scaled_df,
        "k_eval_df": k_eval_df,
        "model": model,
        "labels": labels,
        "result_df": result_df,
        "scaled_centers": scaled_centers,
        "original_centers": original_centers,
    }


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, ID_COLUMN, FEATURE_COLUMNS, and K_VALUE before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    results = kmeans_workflow(raw_df)

    save_table(results["feature_df"], OUTPUT_DIR / "01_clean_feature_data.csv")
    save_table(results["scaled_df"], OUTPUT_DIR / "02_standardized_feature_data.csv")
    save_table(results["k_eval_df"], OUTPUT_DIR / "03_k_value_evaluation.csv")
    save_table(results["result_df"], OUTPUT_DIR / "04_kmeans_cluster_result.csv")
    save_table(results["scaled_centers"], OUTPUT_DIR / "05_scaled_cluster_centers.csv")
    save_table(results["original_centers"], OUTPUT_DIR / "06_original_cluster_centers.csv")

    plot_k_selection(results["k_eval_df"])
    plot_pca_cluster(results["scaled_df"], results["labels"])
    plot_cluster_profile(results["scaled_centers"])

    print("K-means clustering result count:")
    print(results["result_df"]["聚类类别"].value_counts().sort_index())
    print(f"\nK-means workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
