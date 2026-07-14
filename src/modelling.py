"""
Modelling pipeline for GKL housing price prediction.

Models: Ridge (baseline), LightGBM, XGBoost, CatBoost
Tuning: Optuna with 5-fold CV
Evaluation: R², MAE, RMSE on log-transformed prices + SHAP analysis
"""

import warnings
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

from src.config import (
    DATA_PROCESSED, FIGURES_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    TARGET_LOG, FEATURES_TO_DROP_BEFORE_MODELLING,
    PALETTE, MODEL_COLOURS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E0E0E0",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def load_model_data():
    df = pd.read_csv(DATA_PROCESSED / "housing_model.csv")
    drop_cols = [c for c in FEATURES_TO_DROP_BEFORE_MODELLING if c in df.columns]
    y = df[TARGET_LOG]
    X = df.drop(columns=drop_cols + [TARGET_LOG])
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse, "y_pred": y_pred}


def train_ridge(X_train_scaled, y_train, X_test_scaled, y_test):
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    result = evaluate(model, X_test_scaled, y_test, "Ridge")
    result["model"] = model
    print(f"Ridge       -> R²={result['R2']:.4f}  MAE={result['MAE']:.4f}  RMSE={result['RMSE']:.4f}")
    return result


def _lgb_objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    return -scores.mean()


def train_lightgbm(X_train, y_train, X_test, y_test, n_trials=50):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _lgb_objective(t, X_train, y_train), n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    model = lgb.LGBMRegressor(**best)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test, "LightGBM")
    result["model"] = model
    result["best_params"] = best
    print(f"LightGBM    -> R²={result['R2']:.4f}  MAE={result['MAE']:.4f}  RMSE={result['RMSE']:.4f}")
    return result


def _xgb_objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    return -scores.mean()


def train_xgboost(X_train, y_train, X_test, y_test, n_trials=50):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _xgb_objective(t, X_train, y_train), n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
    model = xgb.XGBRegressor(**best)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test, "XGBoost")
    result["model"] = model
    result["best_params"] = best
    print(f"XGBoost     -> R²={result['R2']:.4f}  MAE={result['MAE']:.4f}  RMSE={result['RMSE']:.4f}")
    return result


def _cat_objective(trial, X_train, y_train):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "random_seed": RANDOM_STATE,
        "verbose": 0,
    }
    model = cb.CatBoostRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    return -scores.mean()


def train_catboost(X_train, y_train, X_test, y_test, n_trials=50):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _cat_objective(t, X_train, y_train), n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"random_seed": RANDOM_STATE, "verbose": 0})
    model = cb.CatBoostRegressor(**best)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test, "CatBoost")
    result["model"] = model
    result["best_params"] = best
    print(f"CatBoost    -> R²={result['R2']:.4f}  MAE={result['MAE']:.4f}  RMSE={result['RMSE']:.4f}")
    return result


# ── Visualisation functions ─────────────────────────────────────────

def plot_model_comparison(results):
    _style()
    df = pd.DataFrame([{k: v for k, v in r.items() if k in ("Model", "R2", "MAE", "RMSE")} for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("R2", "R² (higher is better)"), ("MAE", "MAE (lower is better)"), ("RMSE", "RMSE (lower is better)")]

    for ax, (col, label) in zip(axes, metrics):
        colours = [MODEL_COLOURS.get(m, PALETTE["primary"]) for m in df["Model"]]
        bars = ax.bar(df["Model"], df[col], color=colours, edgecolor="white", width=0.6)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(label)
        ax.set_title(col)

    fig.suptitle("Model Performance Comparison (Log Scale)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_model_comparison.png")
    plt.close()
    print("Saved 08_model_comparison.png")


def plot_actual_vs_predicted(results, y_test):
    _style()
    top_models = [r for r in results if r["Model"] != "Ridge"]
    fig, axes = plt.subplots(1, len(top_models), figsize=(5 * len(top_models), 5))
    if len(top_models) == 1:
        axes = [axes]

    for ax, r in zip(axes, top_models):
        colour = MODEL_COLOURS.get(r["Model"], PALETTE["primary"])
        ax.scatter(y_test, r["y_pred"], alpha=0.1, s=4, color=colour)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "--", color=PALETTE["accent2"], linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Price (log)")
        ax.set_ylabel("Predicted Price (log)")
        ax.set_title(f"{r['Model']} (R²={r['R2']:.4f})")
        ax.legend(fontsize=8)

    fig.suptitle("Actual vs Predicted (Log Scale)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_actual_vs_predicted.png")
    plt.close()
    print("Saved 09_actual_vs_predicted.png")


def plot_residuals(results, y_test):
    _style()
    top_models = [r for r in results if r["Model"] != "Ridge"]
    fig, axes = plt.subplots(1, len(top_models), figsize=(5 * len(top_models), 5))
    if len(top_models) == 1:
        axes = [axes]

    for ax, r in zip(axes, top_models):
        residuals = y_test.values - r["y_pred"]
        colour = MODEL_COLOURS.get(r["Model"], PALETTE["primary"])
        ax.scatter(r["y_pred"], residuals, alpha=0.1, s=4, color=colour)
        ax.axhline(y=0, color=PALETTE["accent2"], linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted Price (log)")
        ax.set_ylabel("Residual")
        ax.set_title(r["Model"])

    fig.suptitle("Residual Plots (Log Scale)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_residual_plots.png")
    plt.close()
    print("Saved 10_residual_plots.png")


def plot_error_stratification(results, y_test):
    _style()
    top_models = [r for r in results if r["Model"] != "Ridge"]
    decile_labels = [f"D{i+1}" for i in range(10)]
    bins = pd.qcut(y_test, 10, labels=False, duplicates="drop")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(decile_labels))
    width = 0.25

    for i, r in enumerate(top_models):
        residuals = y_test.values - r["y_pred"]
        rmse_by_decile = []
        for d in range(10):
            mask = bins == d
            if mask.sum() > 0:
                rmse_by_decile.append(np.sqrt(np.mean(residuals[mask] ** 2)))
            else:
                rmse_by_decile.append(0)
        colour = MODEL_COLOURS.get(r["Model"], PALETTE["primary"])
        ax.bar(x + i * width, rmse_by_decile, width, label=r["Model"], color=colour, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(decile_labels)
    ax.set_xlabel("Price Decile (D1=cheapest, D10=most expensive)")
    ax.set_ylabel("RMSE")
    ax.set_title("Error Stratification by Price Decile", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_error_stratification.png")
    plt.close()
    print("Saved 11_error_stratification.png")


def plot_feature_importance(results, feature_names, top_n=15):
    _style()
    tree_models = [r for r in results if r["Model"] != "Ridge"]
    fig, axes = plt.subplots(1, len(tree_models), figsize=(6 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, r in zip(axes, tree_models):
        model = r["model"]
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            continue

        idx = np.argsort(importance)[-top_n:]
        colour = MODEL_COLOURS.get(r["Model"], PALETTE["primary"])

        names = [feature_names[i] for i in idx]
        names = [n.replace("Property_Type_Simplified_", "PT: ").replace("Furnishing_Cleaned_", "F: ")
                 for n in names]

        ax.barh(range(top_n), importance[idx], color=colour, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"{r['Model']}")

    fig.suptitle(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_feature_importance.png")
    plt.close()
    print("Saved 12_feature_importance.png")


def plot_shap_summary(best_result, X_test):
    _style()
    model = best_result["model"]
    name = best_result["Model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    display_names = [
        c.replace("Property_Type_Simplified_", "PT: ").replace("Furnishing_Cleaned_", "F: ")
        for c in X_test.columns
    ]
    X_display = X_test.copy()
    X_display.columns = display_names

    fig, ax = plt.subplots(figsize=(9, 8))
    shap.summary_plot(shap_values, X_display, max_display=15, show=False, plot_size=None)
    ax.set_title(f"SHAP Summary -- {name}", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "13_shap_summary.png")
    plt.close()
    print("Saved 13_shap_summary.png")


def save_results_summary(results):
    summary = []
    for r in results:
        row = {"Model": r["Model"], "R2": round(r["R2"], 4),
               "MAE": round(r["MAE"], 4), "RMSE": round(r["RMSE"], 4)}
        if "best_params" in r:
            row["best_params"] = r["best_params"]
        summary.append(row)
    out_path = DATA_PROCESSED / "model_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved results to {out_path}")


def run_pipeline(n_trials=50):
    print("Loading data...")
    X, y = load_model_data()
    print(f"Features: {X.shape[1]}, Samples: {len(X):,}\n")

    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}\n")

    print("Training models...")
    results = []
    results.append(train_ridge(X_train_s, y_train, X_test_s, y_test))
    results.append(train_lightgbm(X_train, y_train, X_test, y_test, n_trials))
    results.append(train_xgboost(X_train, y_train, X_test, y_test, n_trials))
    results.append(train_catboost(X_train, y_train, X_test, y_test, n_trials))

    print("\nGenerating figures...")
    plot_model_comparison(results)
    plot_actual_vs_predicted(results, y_test)
    plot_residuals(results, y_test)
    plot_error_stratification(results, y_test)
    plot_feature_importance(results, X.columns.tolist())

    best = max(results, key=lambda r: r["R2"])
    print(f"\nBest model: {best['Model']} (R²={best['R2']:.4f})")
    plot_shap_summary(best, X_test)

    save_results_summary(results)
    print("\nDone.")
    return results


if __name__ == "__main__":
    run_pipeline()
