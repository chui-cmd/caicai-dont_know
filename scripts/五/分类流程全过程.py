"""
Classification modeling full workflow template.

Outputs:
1. Cleaned modeling data.
2. Train/test metrics.
3. Classification report.
4. Prediction result table.
5. Confusion matrix, ROC curve, feature importance plot.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, TARGET_COLUMN, and FEATURE_COLUMNS.
    3. Run:
       python scripts/五/分类流程全过程.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


INPUT_FILE = Path("data/your_classification_data.csv")
OUTPUT_DIR = Path("outputs/五/分类流程")

# Optional ID/name column. Set to None if there is no ID column.
ID_COLUMN = "样本"

TARGET_COLUMN = "类别"

# If this list is empty, all columns except ID_COLUMN and TARGET_COLUMN will be used.
FEATURE_COLUMNS = [
    "特征1",
    "特征2",
    "特征3",
]

# Model type:
#   "logistic_regression"
#   "random_forest"
MODEL_TYPE = "random_forest"
TEST_SIZE = 0.25
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


def get_model_data(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if FEATURE_COLUMNS:
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        feature_columns = FEATURE_COLUMNS
    else:
        excluded = {TARGET_COLUMN}
        if ID_COLUMN:
            excluded.add(ID_COLUMN)
        feature_columns = [col for col in df.columns if col not in excluded]

    model_df = df[feature_columns + [TARGET_COLUMN]].copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN])

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(model_df[col]):
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df[col] = model_df[col].fillna(model_df[col].median())
        else:
            mode_value = model_df[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "缺失"
            model_df[col] = model_df[col].fillna(fill_value).astype(str)

    model_df = pd.get_dummies(model_df, columns=[col for col in feature_columns if not pd.api.types.is_numeric_dtype(model_df[col])])
    encoded_feature_columns = [col for col in model_df.columns if col != TARGET_COLUMN]

    return model_df, encoded_feature_columns


def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    return y_encoded, encoder


def build_model():
    if MODEL_TYPE == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
    if MODEL_TYPE == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    raise ValueError("MODEL_TYPE must be 'logistic_regression' or 'random_forest'.")


def calculate_metrics(y_true, y_pred, y_prob, class_names):
    average_method = "binary" if len(class_names) == 2 else "weighted"
    metrics = {
        "模型": MODEL_TYPE,
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, average=average_method, zero_division=0),
        "召回率": recall_score(y_true, y_pred, average=average_method, zero_division=0),
        "F1值": f1_score(y_true, y_pred, average=average_method, zero_division=0),
    }

    if y_prob is not None and len(class_names) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics["AUC"] = None

    return pd.DataFrame([metrics])


def get_predict_proba(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)
    return None


def build_feature_importance(model, feature_columns):
    if MODEL_TYPE == "random_forest":
        importance = model.feature_importances_
    elif MODEL_TYPE == "logistic_regression":
        clf = model.named_steps["model"]
        if clf.coef_.shape[0] == 1:
            importance = abs(clf.coef_[0])
        else:
            importance = abs(clf.coef_).mean(axis=0)
    else:
        return pd.DataFrame()

    return pd.DataFrame({
        "特征": feature_columns,
        "重要性": importance,
    }).sort_values("重要性", ascending=False)


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=科研顺序色带, xticklabels=class_names, yticklabels=class_names)
    plt.title("分类混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob):
    if y_prob is None:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_value = roc_auc_score(y_true, y_prob[:, 1])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=科研配色["深蓝"], linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color=科研配色["红"], linestyle="--")
    plt.title("二分类ROC曲线")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_df):
    if importance_df.empty:
        return

    plot_df = importance_df.head(20)
    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="重要性", y="特征", color=科研配色["深蓝"])
    plt.title("分类模型特征重要性")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def classification_workflow(df):
    model_df, feature_columns = get_model_data(df)
    x = model_df[feature_columns]
    y, encoder = encode_target(model_df[TARGET_COLUMN])

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = build_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = get_predict_proba(model, x_test)
    class_names = encoder.classes_

    metrics_df = calculate_metrics(y_test, y_pred, y_prob, class_names)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)).T
    importance_df = build_feature_importance(model, feature_columns)

    prediction_df = x_test.copy()
    prediction_df["真实类别"] = encoder.inverse_transform(y_test)
    prediction_df["预测类别"] = encoder.inverse_transform(y_pred)
    if y_prob is not None:
        for idx, class_name in enumerate(class_names):
            prediction_df[f"预测概率_{class_name}"] = y_prob[:, idx]

    return {
        "model_df": model_df,
        "metrics_df": metrics_df,
        "report_df": report_df.reset_index().rename(columns={"index": "类别"}),
        "importance_df": importance_df,
        "prediction_df": prediction_df,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "class_names": class_names,
    }


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, TARGET_COLUMN, FEATURE_COLUMNS, and MODEL_TYPE before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    results = classification_workflow(raw_df)

    save_table(results["model_df"], OUTPUT_DIR / "01_clean_encoded_model_data.csv")
    save_table(results["metrics_df"], OUTPUT_DIR / "02_classification_metrics.csv")
    save_table(results["report_df"], OUTPUT_DIR / "03_classification_report.csv")
    save_table(results["importance_df"], OUTPUT_DIR / "04_feature_importance.csv")
    save_table(results["prediction_df"], OUTPUT_DIR / "05_prediction_result.csv")

    plot_confusion_matrix(results["y_test"], results["y_pred"], results["class_names"])
    if len(results["class_names"]) == 2:
        plot_roc_curve(results["y_test"], results["y_prob"])
    plot_feature_importance(results["importance_df"])

    print("Classification metrics:")
    print(results["metrics_df"])
    print(f"\nClassification workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
