"""
modeling.py
===========
Train, evaluate, and compare multiple ML models for student performance prediction.

Models trained:
  1. Linear Regression       — interpretable baseline
  2. Decision Tree Regressor — non-linear, prone to overfit (good teaching moment)
  3. Random Forest Regressor — ensemble, strong performance
  4. Gradient Boosting       — often best; shows advanced knowledge
  5. Ridge Regression        — regularised linear (prevents overfit)

Evaluation:
  - RMSE, MAE, R² on test set
  - 5-fold cross-validation (anti-overfit guard)
  - Learning curves (detect overfit/underfit)
  - Feature importance plot (Random Forest)
  - Residuals analysis

Author: Student Performance Project
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import cross_val_score, learning_curve
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("reports/figures")
MODEL_DIR  = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    """
    Return a dict of models with production-ready hyperparameters.
    
    Hyperparameter rationale:
    - DT max_depth=8: unconstrained trees massively overfit tabular data
    - RF n_estimators=200: more trees → lower variance, marginal cost after ~200
    - RF max_features='sqrt': standard best practice for RF
    - GB n_estimators=200, lr=0.05: slow learning rate prevents overfit
    - Ridge alpha=1.0: light L2 regularisation; tune via CV in production
    """
    return {
        "Linear Regression":      LinearRegression(),
        "Ridge Regression":       Ridge(alpha=1.0),
        "Decision Tree":          DecisionTreeRegressor(
                                      max_depth=8,
                                      min_samples_leaf=10,
                                      random_state=42),
        "Random Forest":          RandomForestRegressor(
                                      n_estimators=200,
                                      max_depth=12,
                                      max_features="sqrt",
                                      min_samples_leaf=5,
                                      random_state=42,
                                      n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(
                                      n_estimators=200,
                                      learning_rate=0.05,
                                      max_depth=4,
                                      subsample=0.8,
                                      random_state=42),
    }


# ── Evaluation metrics ────────────────────────────────────────────────────────
def evaluate_model(
    model, X_train, X_test, y_train, y_test
) -> dict:
    """
    Compute training and test metrics.
    
    Returns a dict with:
      - rmse_train, rmse_test: Root Mean Squared Error (same unit as target)
      - mae_test:  Mean Absolute Error (interpretable: avg points off)
      - r2_train, r2_test:  R² score (% variance explained)
      - cv_r2_mean, cv_r2_std: 5-fold cross-validation R² (anti-overfit check)
      - overfit_gap: r2_train - r2_test (>0.10 → overfit warning)
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    # Cross-validation on FULL training data (5 folds)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )

    return {
        "rmse_train":   round(rmse_train,  3),
        "rmse_test":    round(rmse_test,   3),
        "mae_test":     round(mae_test,    3),
        "r2_train":     round(r2_train,    4),
        "r2_test":      round(r2_test,     4),
        "cv_r2_mean":   round(cv_scores.mean(), 4),
        "cv_r2_std":    round(cv_scores.std(),  4),
        "overfit_gap":  round(r2_train - r2_test, 4),
        "y_pred_test":  y_pred_test,
    }


# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison_table(results: dict) -> pd.DataFrame:
    """Pretty-print model comparison and return as DataFrame."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":         name,
            "R² Test":       m["r2_test"],
            "R² Train":      m["r2_train"],
            "CV R² (mean)":  m["cv_r2_mean"],
            "CV R² (±std)":  m["cv_r2_std"],
            "RMSE Test":     m["rmse_test"],
            "MAE Test":      m["mae_test"],
            "Overfit Gap":   m["overfit_gap"],
        })

    df_results = pd.DataFrame(rows).sort_values("R² Test", ascending=False)

    print("\n" + "=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("\n  Overfit Gap = R²_train − R²_test | >0.10 = overfit warning")
    print("=" * 80)

    return df_results


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_model_comparison(df_results: pd.DataFrame) -> None:
    """Bar chart comparing R² and RMSE across models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # R² comparison
    colors = ["#27AE60" if g < 0.05 else "#E74C3C"
              for g in df_results["Overfit Gap"]]
    bars = axes[0].barh(df_results["Model"], df_results["R² Test"],
                        color=colors, alpha=0.85, edgecolor="white")
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set_xlabel("R² Score (Test Set)")
    axes[0].set_title("Model R² Comparison\n(Green = overfit gap <5%)", fontweight="bold")
    axes[0].set_xlim(0, 1.05)

    # RMSE comparison
    bars2 = axes[1].barh(df_results["Model"], df_results["RMSE Test"],
                         color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.2f", padding=4)
    axes[1].set_xlabel("RMSE (lower is better)")
    axes[1].set_title("RMSE Comparison\n(Test Set)", fontweight="bold")

    fig.suptitle("Model Performance Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: reports/figures/09_model_comparison.png")


def plot_residuals(y_test, results: dict) -> None:
    """
    Residual analysis for each model.
    
    Good model → residuals scattered randomly around 0 (no pattern).
    Bad model  → residuals show a curve (model misses non-linearity).
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, (name, m) in enumerate(results.items()):
        residuals = y_test - m["y_pred_test"]

        # Scatter: actual vs predicted
        axes[0, i].scatter(m["y_pred_test"], y_test,
                           alpha=0.3, s=15, color="#4C72B0")
        mn = min(y_test.min(), m["y_pred_test"].min())
        mx = max(y_test.max(), m["y_pred_test"].max())
        axes[0, i].plot([mn, mx], [mn, mx], "r--", lw=1.5)
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("Actual")
        axes[0, i].set_title(f"{name}\nActual vs Predicted", fontsize=10)

        # Residual plot
        axes[1, i].scatter(m["y_pred_test"], residuals,
                           alpha=0.3, s=15, color="#E74C3C")
        axes[1, i].axhline(0, color="black", lw=1.5, ls="--")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("Residual")
        axes[1, i].set_title("Residuals", fontsize=10)

    fig.suptitle("Residual Analysis — All Models", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: reports/figures/10_residuals.png")


def plot_feature_importance(rf_model, feature_names: list) -> pd.DataFrame:
    """
    Random Forest feature importance (mean decrease in impurity).
    
    INSIGHT: This tells us which features the model relies on most.
    In production, you can use this to:
      1. Drop low-importance features (faster inference)
      2. Focus data collection on high-importance ones
      3. Explain the model to stakeholders ("the #1 predictor is...")
    """
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.35)))
    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(importances))
    )
    ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title("Random Forest — Feature Importance", fontweight="bold", fontsize=13)
    ax.axvline(importances.values.mean(), color="navy", ls="--", lw=1.5,
               label=f"Mean = {importances.values.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: reports/figures/11_feature_importance.png")

    print("\n  Top 10 Features by Importance:")
    for feat, imp in importances.sort_values(ascending=False).head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:30s} {imp:.4f}  {bar}")

    return importances.sort_values(ascending=False)


def plot_learning_curves(model, X_train, y_train, model_name: str) -> None:
    """
    Learning curves show if we need more data or a better model.
    
    Pattern interpretation:
    - Train score >> Val score AND high variance → OVERFIT (need regularization/more data)
    - Both scores converge at low value → UNDERFIT (need more complex model/features)
    - Both scores converge at high value → GOOD FIT
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="r2", n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Training Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#4C72B0")
    ax.plot(train_sizes, val_mean, "o-", color="#E74C3C", label="Validation Score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#E74C3C")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve — {model_name}", fontweight="bold")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.8, color="gray", ls=":", alpha=0.7, label="Target R²=0.80")

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"12_learning_curve_{safe_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reports/figures/12_learning_curve_{safe_name}.png")


# ── Best model selection & saving ─────────────────────────────────────────────
def select_and_save_best_model(
    trained_models: dict, results: dict, feature_names: list
) -> tuple:
    """
    Select best model by CV R² (not just test R² — prevents selection bias).
    Save model + metadata.
    """
    best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    best_model = trained_models[best_name]

    print(f"\n  ★ Best model: {best_name}")
    print(f"    CV R²: {results[best_name]['cv_r2_mean']:.4f} ± {results[best_name]['cv_r2_std']:.4f}")
    print(f"    Test R²: {results[best_name]['r2_test']:.4f}")

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    metadata = {
        "model_name":   best_name,
        "cv_r2_mean":   results[best_name]["cv_r2_mean"],
        "test_r2":      results[best_name]["r2_test"],
        "test_rmse":    results[best_name]["rmse_test"],
        "test_mae":     results[best_name]["mae_test"],
        "feature_names": feature_names,
    }
    joblib.dump(metadata, MODEL_DIR / "model_metadata.pkl")
    print(f"  ✓ Model saved: {MODEL_DIR}/best_model.pkl")

    return best_name, best_model


# ── Master training pipeline ──────────────────────────────────────────────────
def run_modeling_pipeline(
    X_train, X_test, y_train, y_test, feature_names: list
) -> dict:
    """
    Full modeling pipeline: train → evaluate → visualise → save best.
    Returns dict with all results.
    """
    print("\n" + "=" * 55)
    print("  MODELING PIPELINE")
    print("=" * 55)

    models = get_models()
    results = {}
    trained_models = {}

    print("\n[1/4] Training & evaluating models...")
    for name, model in models.items():
        print(f"      Training: {name}...", end=" ")
        m = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name]        = m
        trained_models[name] = model
        print(f"R²={m['r2_test']:.3f}  RMSE={m['rmse_test']:.2f}  "
              f"CV={m['cv_r2_mean']:.3f}±{m['cv_r2_std']:.3f}")

    print("\n[2/4] Comparison table...")
    df_results = print_comparison_table(results)

    print("\n[3/4] Generating plots...")
    plot_model_comparison(df_results)
    plot_residuals(y_test, results)

    rf_model = trained_models["Random Forest"]
    importances = plot_feature_importance(rf_model, feature_names)

    print("\n[4/4] Learning curves (Random Forest)...")
    plot_learning_curves(rf_model, X_train, y_train, "Random Forest")

    best_name, best_model = select_and_save_best_model(
        trained_models, results, feature_names
    )

    print("\n" + "=" * 55)

    return {
        "models":       trained_models,
        "results":      results,
        "df_results":   df_results,
        "best_name":    best_name,
        "best_model":   best_model,
        "importances":  importances,
    }


if __name__ == "__main__":
    from pathlib import Path
    from generate_dataset import generate_student_data
    from preprocessing    import run_preprocessing_pipeline

    df      = generate_student_data(2000)
    pp      = run_preprocessing_pipeline(df)
    outputs = run_modeling_pipeline(
        pp["X_train"], pp["X_test"],
        pp["y_train"], pp["y_test"],
        pp["feature_names"],
    )
