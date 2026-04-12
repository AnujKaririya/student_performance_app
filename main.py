"""
main.py
=======
Master orchestrator — runs the complete Student Performance pipeline:
  1. Generate dataset
  2. EDA
  3. Preprocessing
  4. Modeling & evaluation
  5. Prints summary + next steps

Usage:
    python main.py              # full pipeline
    python main.py --skip-eda  # skip EDA plots (faster)

Author: Student Performance Project
"""

import argparse
import time
from pathlib import Path

import pandas as pd


def main(skip_eda: bool = False):
    print("\n" + "█" * 60)
    print("  STUDENT PERFORMANCE ANALYSIS & PREDICTION PIPELINE")
    print("█" * 60)
    t_start = time.time()

    # ── Step 1: Generate Dataset ──────────────────────────────────────────────
    print("\n[STEP 1] Generating synthetic dataset...")
    from generate_dataset import generate_student_data
    Path("data").mkdir(exist_ok=True)

    df = generate_student_data(n=2000)
    df.to_csv("data/student_performance_raw.csv", index=False)
    print(f"  ✓ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  ✓ Saved: data/student_performance_raw.csv")

    # ── Step 2: EDA ────────────────────────────────────────────────────────────
    if not skip_eda:
        print("\n[STEP 2] Running Exploratory Data Analysis...")
        from eda import run_full_eda
        run_full_eda(df)
    else:
        print("\n[STEP 2] EDA skipped (--skip-eda flag set)")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    print("\n[STEP 3] Running preprocessing pipeline...")
    from preprocessing import run_preprocessing_pipeline
    pp = run_preprocessing_pipeline(df, save_artifacts=True, artifact_dir="models")

    # ── Step 4: Modeling ──────────────────────────────────────────────────────
    print("\n[STEP 4] Training & evaluating models...")
    from modeling import run_modeling_pipeline
    outputs = run_modeling_pipeline(
        pp["X_train"], pp["X_test"],
        pp["y_train"], pp["y_test"],
        pp["feature_names"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    best    = outputs["best_name"]
    res     = outputs["results"][best]

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  ✓ Best model:    {best}")
    print(f"  ✓ Test R²:       {res['r2_test']:.4f}")
    print(f"  ✓ Test RMSE:     {res['rmse_test']:.2f} points")
    print(f"  ✓ Test MAE:      {res['mae_test']:.2f} points")
    print(f"  ✓ CV R²:         {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")
    print(f"  ✓ Overfit gap:   {res['overfit_gap']:.4f} {'(OK)' if res['overfit_gap'] < 0.05 else '(⚠ review)'}")
    print(f"  ✓ Runtime:       {elapsed:.1f}s")

    print("\n  Output files:")
    print("    models/best_model.pkl          — trained model")
    print("    models/scaler.pkl              — feature scaler")
    print("    models/feature_names.pkl       — feature list")
    print("    models/model_metadata.pkl      — metrics snapshot")
    print("    reports/figures/               — all EDA & model plots")
    print("    data/student_performance_raw.csv")

    print("\n  Next steps:")
    print("    streamlit run app.py           — launch the web app")
    print("    Open http://localhost:8501")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA visualisations for faster run")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda)
