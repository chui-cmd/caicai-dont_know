"""
Multiple linear regression full workflow template.

Outputs:
1. Regression coefficients, R2, adjusted R2, F-test p-value.
2. Coefficient table with t value and p value.
3. Prediction result table.
4. Residual plot, fitted-vs-actual plot, residual histogram, Q-Q plot.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, TARGET_COLUMN, and FEATURE_COLUMNS.
    3. Run:
       python scripts/三/多元线性回归全过程.py
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
import statsmodels.api as sm


INPUT_FILE = Path("data/your_data.csv")
OUTPUT_DIR = Path("outputs/三/多元线性回归")

# Optional ID/name column. Set to None if there is no ID column.
ID_COLUMN = None

# Edit these after you get the real data.
TARGET_COLUMN = "因变量"
FEATURE_COLUMNS = [
    "自变量1",
    "自变量2",
    "自变量3",
]

# Missing value strategy for numeric columns:
#   "median" or "mean"
MISSING_FILL_STRATEGY = "median"


def setup_plot_style():
    设置科研绘图风格(plt, sns)


def read_table(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Only CSV and Excel files are supported.")


def check_columns(df, target_column, feature_columns):
    required_columns = [target_column] + feature_columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")


def fill_missing_values(df, cols, strategy):
    result = df.copy()
    fill_records = []

    for col in cols:
        x = pd.to_numeric(result[col], errors="coerce")
        if strategy == "median":
            fill_value = x.median()
        elif strategy == "mean":
            fill_value = x.mean()
        else:
            raise ValueError("MISSING_FILL_STRATEGY must be 'median' or 'mean'.")

        before_missing = int(x.isna().sum())
        result[col] = x.fillna(fill_value)
        fill_records.append({
            "字段名": col,
            "填补方法": strategy,
            "填补值": fill_value,
            "填补前缺失数量": before_missing,
            "填补后缺失数量": int(result[col].isna().sum()),
        })

    return result, pd.DataFrame(fill_records)


def prepare_regression_data(df, target_column, feature_columns):
    y = pd.to_numeric(df[target_column], errors="coerce")
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_index = y.notna() & x.notna().all(axis=1)

    y = y.loc[valid_index]
    x = x.loc[valid_index]
    x = sm.add_constant(x)

    if len(y) <= len(feature_columns) + 1:
        raise ValueError("Sample size is too small for multiple linear regression.")

    return x, y, valid_index


def fit_ols_model(x, y):
    return sm.OLS(y, x).fit()


def build_model_summary(model):
    return pd.DataFrame([{
        "样本量": int(model.nobs),
        "R2": model.rsquared,
        "调整后R2": model.rsquared_adj,
        "F统计量": model.fvalue,
        "F检验p值": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
    }])


def build_coefficient_table(model):
    table = pd.DataFrame({
        "变量": model.params.index,
        "回归系数": model.params.values,
        "标准误": model.bse.values,
        "t值": model.tvalues.values,
        "p值": model.pvalues.values,
        "95%置信区间下限": model.conf_int()[0].values,
        "95%置信区间上限": model.conf_int()[1].values,
    })
    return table


def build_prediction_table(df, model, x, y, valid_index):
    result = df.loc[valid_index].copy()
    result["真实值"] = y.values
    result["预测值"] = model.predict(x).values
    result["残差"] = result["真实值"] - result["预测值"]
    result["标准化残差"] = model.get_influence().resid_studentized_internal
    return result


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_residual_vs_fitted(prediction_df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=prediction_df, x="预测值", y="残差", color=科研配色["深蓝"])
    plt.axhline(0, color=科研配色["红"], linestyle="--", linewidth=1.2)
    plt.title("残差-拟合值图")
    plt.xlabel("预测值")
    plt.ylabel("残差")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_vs_fitted.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(prediction_df):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=prediction_df, x="真实值", y="预测值", color=科研配色["浅蓝"])
    min_value = min(prediction_df["真实值"].min(), prediction_df["预测值"].min())
    max_value = max(prediction_df["真实值"].max(), prediction_df["预测值"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color=科研配色["红"], linestyle="--")
    plt.title("真实值与预测值对比")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_histogram(prediction_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(prediction_df["残差"], kde=True, color=科研配色["深蓝"])
    plt.title("残差分布直方图")
    plt.xlabel("残差")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_qq(model):
    fig = sm.qqplot(model.resid, line="45", fit=True)
    fig.set_size_inches(6, 6)
    plt.title("残差Q-Q图")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_qq_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_coefficient_bar(coefficient_df):
    plot_df = coefficient_df[coefficient_df["变量"] != "const"].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.45)))
    sns.barplot(data=plot_df, x="回归系数", y="变量", color=科研配色["深蓝"])
    plt.axvline(0, color=科研配色["红"], linestyle="--", linewidth=1)
    plt.title("多元线性回归系数")
    plt.xlabel("回归系数")
    plt.ylabel("变量")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coefficient_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def regression_workflow(df, target_column, feature_columns):
    check_columns(df, target_column, feature_columns)
    clean_df, fill_log = fill_missing_values(
        df,
        [target_column] + feature_columns,
        MISSING_FILL_STRATEGY,
    )
    x, y, valid_index = prepare_regression_data(clean_df, target_column, feature_columns)
    model = fit_ols_model(x, y)
    model_summary = build_model_summary(model)
    coefficient_df = build_coefficient_table(model)
    prediction_df = build_prediction_table(clean_df, model, x, y, valid_index)

    return {
        "clean_df": clean_df,
        "fill_log": fill_log,
        "model": model,
        "model_summary": model_summary,
        "coefficient_df": coefficient_df,
        "prediction_df": prediction_df,
    }


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, TARGET_COLUMN, and FEATURE_COLUMNS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    results = regression_workflow(raw_df, TARGET_COLUMN, FEATURE_COLUMNS)

    save_table(results["clean_df"], OUTPUT_DIR / "01_missing_filled_data.csv")
    save_table(results["fill_log"], OUTPUT_DIR / "02_missing_fill_log.csv")
    save_table(results["model_summary"], OUTPUT_DIR / "03_model_summary.csv")
    save_table(results["coefficient_df"], OUTPUT_DIR / "04_coefficient_pvalue_table.csv")
    save_table(results["prediction_df"], OUTPUT_DIR / "05_prediction_residual_result.csv")

    plot_residual_vs_fitted(results["prediction_df"])
    plot_actual_vs_predicted(results["prediction_df"])
    plot_residual_histogram(results["prediction_df"])
    plot_qq(results["model"])
    plot_coefficient_bar(results["coefficient_df"])

    with open(OUTPUT_DIR / "06_statsmodels_summary.txt", "w", encoding="utf-8") as file:
        file.write(str(results["model"].summary()))

    print("Model summary:")
    print(results["model_summary"])
    print("\nCoefficient table:")
    print(results["coefficient_df"])
    print(f"\nMultiple linear regression workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
