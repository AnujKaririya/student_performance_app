# 🎓 Student Performance Analysis & Prediction

> End-to-end ML project — Data Science Internship Portfolio

## Project Overview

Predicts student exam scores using study habits, attendance, sleep, and socioeconomic
factors. Demonstrates a complete production-grade data science workflow.

---

## Project Structure

```
student_performance/
├── generate_dataset.py   # Synthetic dataset generation with realistic correlations
├── eda.py                # Exploratory Data Analysis (8 plot functions + insights)
├── preprocessing.py      # Full preprocessing pipeline (imputation, outliers, encoding, scaling)
├── modeling.py           # 5 ML models, cross-validation, feature importance
├── app.py                # Streamlit web app (deployment)
├── main.py               # Master orchestrator — runs everything
├── requirements.txt
├── data/
│   └── student_performance_raw.csv
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── model_metadata.pkl
└── reports/
    └── figures/          # All EDA and model plots
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# 3. Launch the web app
streamlit run app.py
```

---

## Features

### Input Features
| Feature | Type | Description |
|---|---|---|
| `hours_studied` | Numeric | Weekly study hours (1–14) |
| `attendance` | Numeric | Class attendance % (50–100) |
| `sleep_hours` | Numeric | Nightly sleep (4–9 hrs) |
| `previous_scores` | Numeric | Last exam score (0–100) |
| `internet_usage` | Numeric | Daily internet hours (0.5–8) |
| `tutoring_sessions` | Numeric | Monthly tutoring sessions (0–8) |
| `parental_education` | Categorical | none / hs / grad / post-grad |
| `extracurricular` | Categorical | yes / no |

### Engineered Features
| Feature | Formula | Why it helps |
|---|---|---|
| `study_efficiency` | `hours × attendance / 100` | Captures the interaction: studying without attending class is less effective |
| `distraction_ratio` | `internet / (hours + ε)` | High internet relative to study → distraction signal |
| `sleep_study_balance` | `sleep / (hours + ε)` | Captures whether the student sacrifices sleep for study |
| `academic_support` | `0.6×tutoring + 0.4×edu_level×2` | Aggregates external help signals |
| `performance_trend` | `attendance − previous_score` | Proxy for improvement vs decline |

---

## Models Trained

| Model | Purpose |
|---|---|
| Linear Regression | Interpretable baseline |
| Ridge Regression | Regularised linear (prevents overfit) |
| Decision Tree | Non-linear, prone to overfit — good teaching example |
| Random Forest | Ensemble; strong generalisation |
| Gradient Boosting | Often best; sequentially corrects errors |

**Selection criterion:** Best 5-fold cross-validation R² (not just test R²)

---

## Interview Talking Points

### "Walk me through this project"
> "I built an end-to-end ML pipeline to predict student exam scores. I started with
> EDA to understand which factors matter most — previous scores and study hours had
> the strongest correlations. I engineered 5 new features, particularly a
> study-efficiency metric combining hours and attendance, which improved R² by ~3%.
> I trained 5 models and selected by cross-validation to avoid selection bias.
> Finally, I deployed it as a Streamlit app with personalised improvement tips."

### "How did you prevent overfitting?"
> "Three ways: (1) regularisation — Ridge penalises large coefficients; (2)
> hyperparameter constraints — max_depth=8 for Decision Tree, min_samples_leaf=5
> for RF; (3) 5-fold cross-validation for model selection — I never chose based on
> raw test score. I also plot learning curves to visually diagnose overfit."

### "What's the biggest limitation?"
> "The dataset is synthetic, so real-world deployment would need validation on
> actual student records. Also, the model can't capture factors like test anxiety,
> learning disabilities, or teaching quality. For production, I'd add SHAP values
> for per-prediction explanations and set up MLflow for experiment tracking."

---

## Advanced Extensions (Bonus)

### 1. SHAP Values (Explainability)
```python
import shap
explainer  = shap.TreeExplainer(best_model)
shap_vals  = explainer.shap_values(X_test)
shap.summary_plot(shap_vals, X_test)
# Shows exactly how much each feature pushes each prediction up or down
```

### 2. MLflow Experiment Tracking
```python
import mlflow
with mlflow.start_run(run_name="RandomForest_v1"):
    mlflow.log_params({"n_estimators": 200, "max_depth": 12})
    mlflow.log_metrics({"r2": 0.89, "rmse": 4.2})
    mlflow.sklearn.log_model(model, "model")
# Track all experiments, compare runs, reproduce results
```

### 3. Replace with Neural Network
```python
from sklearn.neural_network import MLPRegressor
nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    max_iter=500,
    early_stopping=True,  # prevents overfit
    random_state=42,
)
```

---

## Tech Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn · streamlit · joblib
