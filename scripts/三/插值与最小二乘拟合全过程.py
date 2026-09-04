"""
Interpolation and least squares fitting full workflow template.

Outputs:
1. Interpolation result table and plot.
2. Polynomial least squares fitting result table and plot.
3. Optional nonlinear curve fitting result table and plot.
4. Error metrics: MAE, RMSE, MAPE, R2.

Usage:
    1. Put your data file in the data/ folder.
    2. Edit INPUT_FILE, X_COLUMN, Y_COLUMN, and method settings.
    3. Run:
       python scripts/三/插值与最小二乘拟合全过程.py
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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


INPUT_FILE = Path("data/your_xy_data.csv")
OUTPUT_DIR = Path("outputs/三/插值与最小二乘拟合")

# Edit these after you get the real data.
X_COLUMN = "x"
Y_COLUMN = "y"

# Interpolation method:
#   "linear": linear interpolation
#   "quadratic": quadratic interpolation
#   "cubic": cubic interpolation
INTERPOLATION_METHOD = "cubic"
INTERPOLATION_POINTS = 200

# Polynomial least squares degree.
POLYNOMIAL_DEGREE = 2

# Optional nonlinear fitting model:
#   "exponential": y = a * exp(b*x) + c
#   "logistic": y = L / (1 + exp(-k * (x - x0)))
#   None: skip nonlinear fitting
NONLINEAR_MODEL = "exponential"

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


def prepare_xy_data(df):
    missing = [col for col in [X_COLUMN, Y_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")

    xy_df = df[[X_COLUMN, Y_COLUMN]].copy()
    xy_df[X_COLUMN] = pd.to_numeric(xy_df[X_COLUMN], errors="coerce")
    xy_df[Y_COLUMN] = pd.to_numeric(xy_df[Y_COLUMN], errors="coerce")
    xy_df = xy_df.dropna().sort_values(X_COLUMN).drop_duplicates(subset=[X_COLUMN])

    if len(xy_df) < 3:
        raise ValueError("At least 3 valid points are required.")

    return xy_df.reset_index(drop=True)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    mape = np.mean(np.abs(residual) / (np.abs(y_true) + EPS))
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }


def interpolation_workflow(x, y):
    min_points = {"linear": 2, "quadratic": 3, "cubic": 4}
    if INTERPOLATION_METHOD not in min_points:
        raise ValueError("INTERPOLATION_METHOD must be 'linear', 'quadratic', or 'cubic'.")
    if len(x) < min_points[INTERPOLATION_METHOD]:
        raise ValueError(f"{INTERPOLATION_METHOD} interpolation needs at least {min_points[INTERPOLATION_METHOD]} points.")

    x_dense = np.linspace(x.min(), x.max(), INTERPOLATION_POINTS)
    interpolator = interp1d(x, y, kind=INTERPOLATION_METHOD, fill_value="extrapolate")
    y_dense = interpolator(x_dense)
    y_fitted = interpolator(x)

    result_df = pd.DataFrame({
        X_COLUMN: x_dense,
        f"{INTERPOLATION_METHOD}插值值": y_dense,
    })
    metric_df = pd.DataFrame([{
        "方法": f"{INTERPOLATION_METHOD}插值",
        **calculate_metrics(y, y_fitted),
    }])

    return result_df, y_fitted, metric_df


def polynomial_fit_workflow(x, y):
    if POLYNOMIAL_DEGREE < 1:
        raise ValueError("POLYNOMIAL_DEGREE must be at least 1.")
    if len(x) <= POLYNOMIAL_DEGREE:
        raise ValueError("Sample size must be greater than polynomial degree.")

    coefficients = np.polyfit(x, y, POLYNOMIAL_DEGREE)
    polynomial = np.poly1d(coefficients)

    x_dense = np.linspace(x.min(), x.max(), INTERPOLATION_POINTS)
    y_dense = polynomial(x_dense)
    y_fitted = polynomial(x)

    fit_df = pd.DataFrame({
        X_COLUMN: x_dense,
        "多项式最小二乘拟合值": y_dense,
    })
    coef_df = pd.DataFrame({
        "项": [f"x^{power}" for power in range(POLYNOMIAL_DEGREE, -1, -1)],
        "系数": coefficients,
    })
    metric_df = pd.DataFrame([{
        "方法": f"{POLYNOMIAL_DEGREE}次多项式最小二乘",
        **calculate_metrics(y, y_fitted),
    }])

    return fit_df, coef_df, y_fitted, metric_df


def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


def logistic_model(x, l_value, k_value, x0):
    return l_value / (1 + np.exp(-k_value * (x - x0)))


def nonlinear_fit_workflow(x, y):
    if NONLINEAR_MODEL is None:
        return None, None, None, None

    if NONLINEAR_MODEL == "exponential":
        model_func = exponential_model
        initial_guess = [1.0, 0.01, y.min()]
        param_names = ["a", "b", "c"]
        fitted_col = "指数函数最小二乘拟合值"
    elif NONLINEAR_MODEL == "logistic":
        model_func = logistic_model
        initial_guess = [y.max(), 1.0, np.median(x)]
        param_names = ["L", "k", "x0"]
        fitted_col = "Logistic函数最小二乘拟合值"
    else:
        raise ValueError("NONLINEAR_MODEL must be 'exponential', 'logistic', or None.")

    params, covariance = curve_fit(model_func, x, y, p0=initial_guess, maxfev=10000)
    param_std = np.sqrt(np.diag(covariance))

    x_dense = np.linspace(x.min(), x.max(), INTERPOLATION_POINTS)
    y_dense = model_func(x_dense, *params)
    y_fitted = model_func(x, *params)

    fit_df = pd.DataFrame({
        X_COLUMN: x_dense,
        fitted_col: y_dense,
    })
    param_df = pd.DataFrame({
        "参数": param_names,
        "估计值": params,
        "标准误": param_std,
    })
    metric_df = pd.DataFrame([{
        "方法": f"{NONLINEAR_MODEL}非线性最小二乘",
        **calculate_metrics(y, y_fitted),
    }])

    return fit_df, param_df, y_fitted, metric_df


def save_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_all_methods(xy_df, interpolation_df, polynomial_df, nonlinear_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(xy_df[X_COLUMN], xy_df[Y_COLUMN], label="原始数据", color=科研配色["红"], zorder=3)

    plt.plot(
        interpolation_df[X_COLUMN],
        interpolation_df.iloc[:, 1],
        label=f"{INTERPOLATION_METHOD}插值",
        color=科研配色["深蓝"],
        linewidth=2,
    )
    plt.plot(
        polynomial_df[X_COLUMN],
        polynomial_df["多项式最小二乘拟合值"],
        label=f"{POLYNOMIAL_DEGREE}次多项式最小二乘",
        color=科研配色["红"],
        linewidth=2,
    )
    if nonlinear_df is not None:
        plt.plot(
            nonlinear_df[X_COLUMN],
            nonlinear_df.iloc[:, 1],
            label=f"{NONLINEAR_MODEL}非线性最小二乘",
            color=科研配色["浅蓝"],
            linewidth=2,
        )

    plt.title("插值与最小二乘拟合对比")
    plt.xlabel(X_COLUMN)
    plt.ylabel(Y_COLUMN)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpolation_least_squares_compare.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_compare(x, interpolation_resid, polynomial_resid, nonlinear_resid):
    residual_data = [
        pd.DataFrame({X_COLUMN: x, "残差": interpolation_resid, "方法": f"{INTERPOLATION_METHOD}插值"}),
        pd.DataFrame({X_COLUMN: x, "残差": polynomial_resid, "方法": f"{POLYNOMIAL_DEGREE}次多项式最小二乘"}),
    ]
    if nonlinear_resid is not None:
        residual_data.append(pd.DataFrame({X_COLUMN: x, "残差": nonlinear_resid, "方法": f"{NONLINEAR_MODEL}非线性最小二乘"}))

    residual_df = pd.concat(residual_data, ignore_index=True)

    plt.figure(figsize=(10, 5.5))
    sns.lineplot(data=residual_df, x=X_COLUMN, y="残差", hue="方法", marker="o", palette=科研调色板)
    plt.axhline(0, color=科研配色["红"], linestyle="--", linewidth=1)
    plt.title("拟合残差对比")
    plt.xlabel(X_COLUMN)
    plt.ylabel("残差")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_compare.png", dpi=300, bbox_inches="tight")
    plt.close()


def interpolation_and_fit_workflow(df):
    xy_df = prepare_xy_data(df)
    x = xy_df[X_COLUMN].to_numpy(dtype=float)
    y = xy_df[Y_COLUMN].to_numpy(dtype=float)

    interpolation_df, y_interp, interp_metric = interpolation_workflow(x, y)
    polynomial_df, polynomial_coef_df, y_poly, poly_metric = polynomial_fit_workflow(x, y)
    nonlinear_df, nonlinear_param_df, y_nonlinear, nonlinear_metric = nonlinear_fit_workflow(x, y)

    metric_parts = [interp_metric, poly_metric]
    if nonlinear_metric is not None:
        metric_parts.append(nonlinear_metric)
    metrics_df = pd.concat(metric_parts, ignore_index=True)

    fitted_point_df = xy_df.copy()
    fitted_point_df[f"{INTERPOLATION_METHOD}插值拟合值"] = y_interp
    fitted_point_df["多项式最小二乘拟合值"] = y_poly
    fitted_point_df[f"{INTERPOLATION_METHOD}插值残差"] = y - y_interp
    fitted_point_df["多项式最小二乘残差"] = y - y_poly
    if y_nonlinear is not None:
        fitted_point_df[f"{NONLINEAR_MODEL}非线性最小二乘拟合值"] = y_nonlinear
        fitted_point_df[f"{NONLINEAR_MODEL}非线性最小二乘残差"] = y - y_nonlinear

    return {
        "xy_df": xy_df,
        "interpolation_df": interpolation_df,
        "polynomial_df": polynomial_df,
        "polynomial_coef_df": polynomial_coef_df,
        "nonlinear_df": nonlinear_df,
        "nonlinear_param_df": nonlinear_param_df,
        "metrics_df": metrics_df,
        "fitted_point_df": fitted_point_df,
    }


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please edit INPUT_FILE, X_COLUMN, Y_COLUMN, and method settings before running this script.")
        return

    raw_df = read_table(INPUT_FILE)
    results = interpolation_and_fit_workflow(raw_df)

    save_table(results["xy_df"], OUTPUT_DIR / "01_clean_xy_data.csv")
    save_table(results["interpolation_df"], OUTPUT_DIR / "02_interpolation_result.csv")
    save_table(results["polynomial_df"], OUTPUT_DIR / "03_polynomial_fit_curve.csv")
    save_table(results["polynomial_coef_df"], OUTPUT_DIR / "04_polynomial_coefficients.csv")
    save_table(results["metrics_df"], OUTPUT_DIR / "05_error_metrics.csv")
    save_table(results["fitted_point_df"], OUTPUT_DIR / "06_fitted_points_and_residuals.csv")

    if results["nonlinear_df"] is not None:
        save_table(results["nonlinear_df"], OUTPUT_DIR / "07_nonlinear_fit_curve.csv")
        save_table(results["nonlinear_param_df"], OUTPUT_DIR / "08_nonlinear_parameters.csv")

    plot_all_methods(
        results["xy_df"],
        results["interpolation_df"],
        results["polynomial_df"],
        results["nonlinear_df"],
    )
    nonlinear_resid = None
    if NONLINEAR_MODEL is not None and f"{NONLINEAR_MODEL}非线性最小二乘残差" in results["fitted_point_df"]:
        nonlinear_resid = results["fitted_point_df"][f"{NONLINEAR_MODEL}非线性最小二乘残差"]

    plot_residual_compare(
        results["xy_df"][X_COLUMN],
        results["fitted_point_df"][f"{INTERPOLATION_METHOD}插值残差"],
        results["fitted_point_df"]["多项式最小二乘残差"],
        nonlinear_resid,
    )

    print("Error metrics:")
    print(results["metrics_df"])
    print(f"\nInterpolation and least squares fitting completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
