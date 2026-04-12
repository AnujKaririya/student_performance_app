"""
preprocessing.py
================
Production-grade data preprocessing pipeline for Student Performance data.

Design philosophy:
  - Every transformation is a named, testable function
  - No magic numbers — constants are defined at the top
  - The pipeline returns both the processed data and a 'report' dict
    so callers can log what changed (useful for auditing in production)

Author: Student Performance Project
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20       # 80/20 split
IQR_MULTIPLIER = 1.5        # Tukey's fence for outlier detection
TARGET_COL     = "exam_score"
ID_COL         = "student_id"

# Columns that need imputation (numerical → median, categorical → mode)
NUMERIC_FEATURES = [
    "age", "hours_studied", "attendance", "sleep_hours",
    "previous_scores", "internet_usage", "tutoring_sessions",
]
CATEGORICAL_FEATURES = ["gender", "parental_education", "extracurricular"]

# Columns to scale (don't scale target or binary encoded columns)
SCALE_COLS = NUMERIC_FEATURES  # all numeric features will be scaled


# ── Step 1: Basic cleaning ────────────────────────────────────────────────────
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID and any fully-null columns."""
    df = df.drop(columns=[ID_COL], errors="ignore")
    fully_null = df.columns[df.isnull().all()].tolist()
    if fully_null:
        df = df.drop(columns=fully_null)
        print(f"  Dropped fully-null columns: {fully_null}")
    return df


# ── Step 2: Missing value imputation ─────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values:
      - Numeric  → median  (robust to outliers vs mean)
      - Categorical → mode  (most frequent category)

    Returns modified df and a report of what was imputed.
    """
    report = {}

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report[col] = {"missing": int(n_missing), "imputed_with": f"median={median_val:.2f}"}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report[col] = {"missing": int(n_missing), "imputed_with": f"mode={mode_val}"}

    return df, report


# ── Step 3: Outlier detection & capping ───────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    """
    Winsorize outliers using Tukey's IQR fence.
    
    Strategy: CAP rather than DROP.
    - Dropping loses information (especially if outliers are real students)
    - Capping preserves the row while limiting distortion
    - In a real project, you'd also flag these for human review
    """
    report = {}
    for col in cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR

        n_low  = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()

        if n_low > 0 or n_high > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            report[col] = {
                "n_capped_low":  int(n_low),
                "n_capped_high": int(n_high),
                "bounds":        (round(lower, 2), round(upper, 2)),
            }
    return df, report


# ── Step 4: Feature engineering ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed composite features.

    Why these features help:
    1. study_efficiency: High attendance + study hours → interaction captures
       students who maximise both dimensions (more predictive than either alone).
    2. sleep_study_balance: Real research shows sleep deprivation hurts learning
       even if students study many hours. The ratio captures this trade-off.
    3. academic_support_index: Aggregates external support signals (tutoring +
       parental edu) into a single score; reduces dimensionality.
    4. distraction_ratio: Internet usage relative to study time; low ratio → 
       focused student. Non-linear effect caught by this ratio.
    5. performance_trend: Compares previous score to attendance — a proxy for
       whether the student is 'improving' or 'declining'.
    """
    # Encode parental_education to numeric for use in composite features
    edu_order = {"none": 0, "high_school": 1, "graduate": 2, "post_graduate": 3}
    df["parental_edu_num"] = df["parental_education"].map(edu_order)

    # 1. Study efficiency (hours × attendance normalised)
    df["study_efficiency"] = (df["hours_studied"] * df["attendance"]) / 100

    # 2. Sleep-to-study balance (optimal ~1:2 ratio: 7h sleep / 14h study)
    df["sleep_study_balance"] = df["sleep_hours"] / (df["hours_studied"] + 1e-5)

    # 3. Academic support index (tutoring + parental edu, weighted)
    df["academic_support"] = (
        0.6 * df["tutoring_sessions"] + 0.4 * df["parental_edu_num"] * 2
    )

    # 4. Distraction ratio (internet per study hour)
    df["distraction_ratio"] = df["internet_usage"] / (df["hours_studied"] + 1e-5)

    # 5. Performance trend (attendance as % of previous score)
    df["performance_trend"] = df["attendance"] - df["previous_scores"]

    return df


# ── Step 5: Categorical encoding ──────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns:
    - Binary columns (gender, extracurricular) → Label Encoding (0/1)
    - Multi-class (parental_education) → One-Hot Encoding
      (avoids ordinal assumption; drop_first prevents multicollinearity)
    """
    encoders = {}

    # Binary encode gender & extracurricular
    for col in ["gender", "extracurricular"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # One-hot encode parental_education
    if "parental_education" in df.columns:
        dummies = pd.get_dummies(
            df["parental_education"], prefix="edu", drop_first=True
        )
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["parental_education", "parental_edu_num"])

    return df, encoders


# ── Step 6: Feature scaling ────────────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    scale_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    StandardScaler: fit ONLY on training data, then transform both.
    
    CRITICAL: Never fit the scaler on test data — that would leak information.
    This is one of the most common data leakage mistakes in ML.
    """
    # Only scale columns that exist in the data
    cols_to_scale = [c for c in scale_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, scaler


# ── Master pipeline ─────────────────────────────────────────────────────────
def run_preprocessing_pipeline(
    df: pd.DataFrame,
    save_artifacts: bool = True,
    artifact_dir: str = "models",
) -> dict:
    """
    Run the full preprocessing pipeline and return a results dict.

    Returns:
        {
          "X_train": pd.DataFrame,
          "X_test":  pd.DataFrame,
          "y_train": pd.Series,
          "y_test":  pd.Series,
          "scaler":  StandardScaler,
          "encoders": dict,
          "feature_names": list[str],
          "report":  dict,
        }
    """
    print("=" * 55)
    print("  PREPROCESSING PIPELINE")
    print("=" * 55)
    report = {}

    print(f"\n[1/6] Raw data shape: {df.shape}")
    df = drop_irrelevant_columns(df)

    print(f"[2/6] Imputing missing values...")
    df, impute_report = impute_missing(df)
    report["imputation"] = impute_report
    for col, info in impute_report.items():
        print(f"      {col}: {info['missing']} missing → {info['imputed_with']}")

    print(f"[3/6] Capping outliers (IQR method)...")
    df, outlier_report = cap_outliers_iqr(df, NUMERIC_FEATURES)
    report["outliers"] = outlier_report
    for col, info in outlier_report.items():
        print(f"      {col}: capped {info['n_capped_low']} low, {info['n_capped_high']} high")

    print(f"[4/6] Engineering features...")
    df = engineer_features(df)
    new_features = [
        "study_efficiency", "sleep_study_balance",
        "academic_support", "distraction_ratio", "performance_trend",
    ]
    print(f"      Created: {new_features}")

    print(f"[5/6] Encoding categorical variables...")
    df, encoders = encode_categoricals(df)
    print(f"      Encoded: gender, extracurricular (label), parental_education (OHE)")

    # ── Train/test split ────────────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n      Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print(f"[6/6] Scaling numeric features...")
    # Rebuild scale_cols to include engineered numeric features
    all_scale_cols = SCALE_COLS + [
        "study_efficiency", "sleep_study_balance",
        "academic_support", "distraction_ratio", "performance_trend",
    ]
    X_train, X_test, scaler = scale_features(X_train, X_test, all_scale_cols)

    # ── Save artifacts for Streamlit app ────────────────────────────────────
    if save_artifacts:
        import os
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(scaler,   f"{artifact_dir}/scaler.pkl")
        joblib.dump(encoders, f"{artifact_dir}/encoders.pkl")
        print(f"\n  ✓ Artifacts saved to '{artifact_dir}/'")

    print(f"\n  Final feature set ({X_train.shape[1]} features):")
    for i, col in enumerate(X_train.columns, 1):
        print(f"    {i:2d}. {col}")
    print("=" * 55)

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "encoders":      encoders,
        "feature_names": list(X_train.columns),
        "report":        report,
    }


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("data/student_performance_raw.csv")
    if not data_path.exists():
        print("Dataset not found. Run generate_dataset.py first.")
    else:
        df = pd.read_csv(data_path)
        results = run_preprocessing_pipeline(df)
        print(f"\nX_train sample:\n{results['X_train'].head(3)}")
