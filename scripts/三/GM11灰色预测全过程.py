"""
GM(1,1) gray prediction full workflow template.

Outputs:
1. Level ratio test result.
2. GM(1,1) parameters a and b.
3. Fitted values, residuals, relative errors.
4. Future prediction values.
5. Fitting and forecasting plot.

Usage:
    1. Put your time series data file in the data/ folder.
    2. Edit INPUT_FILE, TIME_COLUMN, VALUE_COLUMN, and PREDICT_STEPS.
    3. Run:
       python scripts/三/GM11灰色预测全过程.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("data/your_time_series.csv")
OUTPUT_DIR = Path("outputs/三/GM11灰色预测")

# Edit these after you get the real data.
TIME_COLUMN = "年份"
VALUE_COLUMN = "指标值"
PREDICT_STEPS = 5

EPS = 1e-12


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


def check_columns(df):
    missing = [col for col in [TIME_COLUMN, VALUE_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")


def prepare_series(df):
    check_columns(df)
    series_df = df[[TIME_COLUMN, VALUE_COLUMN]].copy()
    series_df[VALUE_COLUMN] = pd.to_numeric(series_df[VALUE_COLUMN], errors="coerce")
    series_df = series_df.dropna(subset=[VALUE_COLUMN]).reset_index(drop=True)

    if len(series_df) < 4:
        raise ValueError("GM(1,1) usually requires at least 4 observations.")
    if (series_df[VALUE_COLUMN] <= 0).any():
        raise ValueError("GM(1,1) requires all original values to be positive.")

    return series_df


def level_ratio_test(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    ratios = values[:-1] / values[1:]
    lower_bound = np.exp(-2 / (n + 1))
    upper_bound = np.exp(2 / (n + 1))
    passed_mask = (ratios > lower_bound) & (ratios < upper_bound)

    result = pd.DataFrame({
        "序号": np.arange(2, n + 1),
        "级比": ratios,
        "下界": lower_bound,
        "上界": upper_bound,
        "是否通过": passed_mask,
    })
    all_passed = bool(passed_mask.all())
    return result, all_passed


def fit_gm11(values):
    x0 = np.asarray(values, dtype=float)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])

    b_matrix = np.column_stack((-z1, np.ones(len(z1))))
    y_vector = x0[1:]

    parameter = np.linalg.inv(b_matrix.T @ b_matrix) @ b_matrix.T @ y_vector
    a, b = parameter[0], parameter[1]

    return a, b, x1, z1


def predict_gm11(first_value, a, b, total_steps):
    k = np.arange(total_steps)

    if abs(a) < EPS:
        x1_hat = first_value + b * k
    else:
        x1_hat = (first_value - b / a) * np.exp(-a * k) + b / a

    x0_hat = np.empty(total_steps)
    x0_hat[0] = first_value
    x0_hat[1:] = np.diff(x1_hat)
    return x0_hat


def evaluate_fit(actual, fitted):
    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    residual = actual - fitted
    relative_error = np.abs(residual) / (np.abs(actual) + EPS)

    mae = np.mean(np.abs(residual))
    mape = np.mean(relative_error)
    rmse = np.sqrt(np.mean(residual ** 2))

    posterior_error_ratio = np.std(residual, ddof=0) / (np.std(actual, ddof=0) + EPS)
    small_error_probability = np.mean(np.abs(residual - residual.mean()) < 0.6745 * np.std(actual, ddof=0))

    summary = pd.DataFrame([{
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "后验差比C": posterior_error_ratio,
        "小误差概率P": small_error_probability,
    }])
    return residual, relative_error, summary


def build_future_time_labels(time_values, predict_steps):
    time_series = pd.Series(time_values)
    numeric_time = pd.to_numeric(time_series, errors="coerce")

    if numeric_time.notna().all() and len(numeric_time) >= 2:
        step = numeric_time.iloc[-1] - numeric_time.iloc[-2]
        return [numeric_time.iloc[-1] + step * i for i in range(1, predict_steps + 1)]

    return [f"预测{i}" for i in range(1, predict_steps + 1)]


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_fit_and_forecast(result_df):
    plt.figure(figsize=(10, 5.5))

    history_df = result_df[result_df["数据类型"] == "历史"]
    forecast_df = result_df[result_df["数据类型"] == "预测"]

    plt.plot(history_df[TIME_COLUMN], history_df["真实值"], marker="o", label="真实值", color="#3A6EA5")
    plt.plot(history_df[TIME_COLUMN], history_df["GM11预测值"], marker="s", label="拟合值", color="#D95F02")
    if not forecast_df.empty:
        plt.plot(forecast_df[TIME_COLUMN], forecast_df["GM11预测值"], marker="^", label="未来预测", color="#1B9E77")

    plt.title("GM(1,1)灰色预测拟合与外推")
    plt.xlabel(TIME_COLUMN)
    plt.ylabel(VALUE_COLUMN)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gm11_fit_forecast.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual(result_df):
    history_df = result_df[result_df["数据类型"] == "历史"].copy()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=history_df, x=TIME_COLUMN, y="残差", color="#3A6EA5")
    plt.axhline(0, color="#D95F02", linestyle="--", linewidth=1)
    plt.title("GM(1,1)拟合残差图")
    plt.xlabel(TIME_COLUMN)
    plt.ylabel("残差")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gm11_residual_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def gm11_workflow(df):
    series_df = prepare_series(df)
    values = series_df[VALUE_COLUMN].to_numpy(dtype=float)

    ratio_df, ratio_passed = level_ratio_test(values)
    a, b, _, _ = fit_gm11(values)

    total_steps = len(values) + PREDICT_STEPS
    predicted = predict_gm11(values[0], a, b, total_steps)
    fitted = predicted[:len(values)]
    future_predicted = predicted[len(values):]

    residual, relative_error, evaluation_df = evaluate_fit(values, fitted)

    history_result = series_df.copy()
    history_result["真实值"] = values
    history_result["GM11预测值"] = fitted
    history_result["残差"] = residual
    history_result["相对误差"] = relative_error
    history_result["数据类型"] = "历史"

    future_time = build_future_time_labels(series_df[TIME_COLUMN], PREDICT_STEPS)
    forecast_result = pd.DataFrame({
        TIME_COLUMN: future_time,
        VALUE_COLUMN: np.nan,
        "真实值": np.nan,
        "GM11预测值": future_predicted,
        "残差": np.nan,
        "相对误差": np.nan,
        "数据类型": "预测",
    })

    result_df = pd.concat([history_result, forecast_result], ignore_index=True)
    parameter_df = pd.DataFrame([{
        "发展系数a": a,
        "灰作用量b": b,
        "预测步数": PREDICT_STEPS,
        "级比检验是否全部通过": "通过" if ratio_passed else "未通过",
    }])

    return result_df, ratio_df, parameter_df, evaluation_df


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, TIME_COLUMN, VALUE_COLUMN, and PREDICT_STEPS before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    result_df, ratio_df, parameter_df, evaluation_df = gm11_workflow(raw_df)

    save_table(result_df, OUTPUT_DIR / "01_gm11_fit_forecast_result.csv")
    save_table(ratio_df, OUTPUT_DIR / "02_level_ratio_test.csv")
    save_table(parameter_df, OUTPUT_DIR / "03_gm11_parameters.csv")
    save_table(evaluation_df, OUTPUT_DIR / "04_fit_error_evaluation.csv")

    plot_fit_and_forecast(result_df)
    plot_residual(result_df)

    print("GM(1,1) parameters:")
    print(parameter_df)
    print("\nFit error evaluation:")
    print(evaluation_df)
    print(f"\nGM(1,1) workflow completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
