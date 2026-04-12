"""
app.py — Streamlit Web App
===========================
Student Performance Prediction Dashboard

Features:
  - Student input form with sliders and dropdowns
  - Live prediction with confidence range
  - Performance category classification
  - Feature importance visualisation
  - How-it-works explanation

Run: streamlit run app.py

Author: Student Performance Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (minimal, professional) ────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #4C72B0;
        margin-bottom: 0.5rem;
    }
    .score-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .grade-A { background: #d4edda; color: #155724; }
    .grade-B { background: #d1ecf1; color: #0c5460; }
    .grade-C { background: #fff3cd; color: #856404; }
    .grade-D { background: #f8d7da; color: #721c24; }
    .insight-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #3498db;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Cache model artifacts — loaded only once per session."""
    model_dir = Path("models")
    try:
        model    = joblib.load(model_dir / "best_model.pkl")
        scaler   = joblib.load(model_dir / "scaler.pkl")
        features = joblib.load(model_dir / "feature_names.pkl")
        metadata = joblib.load(model_dir / "model_metadata.pkl")
        return model, scaler, features, metadata
    except FileNotFoundError:
        return None, None, None, None


# ── Preprocessing helper ───────────────────────────────────────────────────────
def preprocess_input(raw_inputs: dict, scaler, feature_names: list) -> pd.DataFrame:
    """
    Apply the same transformations as training preprocessing.
    This function must mirror preprocessing.py exactly.
    """
    d = raw_inputs.copy()

    # Parental education → numeric
    edu_map = {"none": 0, "high_school": 1, "graduate": 2, "post_graduate": 3}
    d["parental_edu_num"] = edu_map.get(d["parental_education"], 0)

    # Feature engineering (must match preprocessing.py)
    d["study_efficiency"]    = (d["hours_studied"] * d["attendance"]) / 100
    d["sleep_study_balance"] = d["sleep_hours"] / (d["hours_studied"] + 1e-5)
    d["academic_support"]    = 0.6 * d["tutoring_sessions"] + 0.4 * d["parental_edu_num"] * 2
    d["distraction_ratio"]   = d["internet_usage"] / (d["hours_studied"] + 1e-5)
    d["performance_trend"]   = d["attendance"] - d["previous_scores"]

    # Encode categoricals
    d["gender"]         = 1 if d["gender"] == "male" else 0
    d["extracurricular"] = 1 if d["extracurricular"] == "yes" else 0

    # One-hot encode parental_education (drop_first=True → 'none' is reference)
    edu_dummies = {"edu_graduate": 0, "edu_high_school": 0, "edu_post_graduate": 0}
    if d["parental_education"] == "graduate":
        edu_dummies["edu_graduate"] = 1
    elif d["parental_education"] == "high_school":
        edu_dummies["edu_high_school"] = 1
    elif d["parental_education"] == "post_graduate":
        edu_dummies["edu_post_graduate"] = 1

    d.update(edu_dummies)

    # Remove raw categorical columns
    for col in ["parental_education", "parental_edu_num"]:
        d.pop(col, None)

    # Build DataFrame with correct column order
    df_input = pd.DataFrame([d])
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    # Scale numeric features (same columns as training)
    scale_cols = [
        "age", "hours_studied", "attendance", "sleep_hours", "previous_scores",
        "internet_usage", "tutoring_sessions", "study_efficiency",
        "sleep_study_balance", "academic_support", "distraction_ratio",
        "performance_trend",
    ]
    scale_cols = [c for c in scale_cols if c in df_input.columns]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])

    return df_input


# ── Grade classification ───────────────────────────────────────────────────────
def classify_performance(score: float) -> tuple[str, str, str]:
    """Return (grade, label, css_class) based on predicted score."""
    if score >= 80:
        return "A", "Excellent 🌟", "grade-A"
    elif score >= 65:
        return "B", "Good 👍", "grade-B"
    elif score >= 50:
        return "C", "Average ⚠️", "grade-C"
    else:
        return "D", "Needs Improvement 📚", "grade-D"


# ── AI-powered insights helper ─────────────────────────────────────────────────
def get_improvement_tips(inputs: dict, score: float) -> list[str]:
    """Generate contextual improvement tips based on input values."""
    tips = []
    if inputs["hours_studied"] < 5:
        tips.append("📖 Increase weekly study hours to at least 6–8 hours for significant improvement.")
    if inputs["attendance"] < 75:
        tips.append("🏫 Attendance below 75% strongly correlates with lower scores. Prioritise class attendance.")
    if inputs["internet_usage"] > 5:
        tips.append("📵 High internet usage may be reducing focus. Consider using app blockers during study time.")
    if inputs["sleep_hours"] < 6:
        tips.append("😴 Sleep deprivation impairs memory consolidation. Aim for 7–8 hours.")
    if inputs["tutoring_sessions"] == 0:
        tips.append("🎓 Even 1–2 tutoring sessions/month can improve understanding in weak topics.")
    if score >= 80:
        tips.append("🌟 Outstanding performance! Consider mentoring peers or taking advanced electives.")
    return tips if tips else ["✅ You're doing well across all dimensions! Keep it up."]


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    model, scaler, feature_names, metadata = load_artifacts()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎓 Student Performance Predictor")
    st.markdown("*Enter your details below to predict your exam performance using Machine Learning.*")

    if model is None:
        st.error(
            "⚠️ Model artifacts not found. "
            "Please run `python main.py` first to train the model."
        )
        return

    # Model info banner
    with st.expander("ℹ️ About this model", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metadata.get("model_name", "N/A"))
        col2.metric("Test R²",  f"{metadata.get('test_r2',  0):.3f}")
        col3.metric("Test RMSE", f"{metadata.get('test_rmse', 0):.2f} pts")
        col4.metric("Test MAE",  f"{metadata.get('test_mae',  0):.2f} pts")

    st.divider()

    # ── Sidebar: input form ───────────────────────────────────────────────────
    st.sidebar.header("📝 Student Profile")
    st.sidebar.markdown("Adjust the sliders to match your study habits:")

    with st.sidebar:
        gender = st.selectbox("Gender", ["female", "male"])
        age    = st.slider("Age", 16, 23, 18)
        parental_education = st.selectbox(
            "Parental Education",
            ["none", "high_school", "graduate", "post_graduate"],
            index=1,
            help="Highest education level of either parent",
        )

        st.markdown("---")
        hours_studied = st.slider(
            "📚 Weekly Study Hours", 1.0, 14.0, 6.0, 0.5,
            help="Average hours studied per week"
        )
        attendance = st.slider(
            "🏫 Attendance (%)", 50, 100, 80,
            help="Percentage of classes attended"
        )
        sleep_hours = st.slider(
            "😴 Sleep Hours / Night", 4.0, 9.0, 7.0, 0.5,
        )
        previous_scores = st.slider(
            "📊 Previous Exam Score", 20, 100, 60,
            help="Your score in the last exam (0–100)"
        )
        internet_usage = st.slider(
            "📱 Daily Internet/Social Media (hrs)", 0.5, 8.0, 3.0, 0.5,
        )
        tutoring_sessions = st.slider(
            "🎓 Tutoring Sessions / Month", 0, 8, 2,
        )
        extracurricular = st.selectbox(
            "⚽ Extracurricular Activities", ["no", "yes"]
        )

    # ── Prediction ────────────────────────────────────────────────────────────
    raw_inputs = {
        "age":                age,
        "gender":             gender,
        "parental_education": parental_education,
        "hours_studied":      hours_studied,
        "attendance":         attendance,
        "sleep_hours":        sleep_hours,
        "previous_scores":    previous_scores,
        "internet_usage":     internet_usage,
        "tutoring_sessions":  tutoring_sessions,
        "extracurricular":    extracurricular,
    }

    try:
        df_input = preprocess_input(raw_inputs, scaler, feature_names)
        predicted_score = float(model.predict(df_input)[0])
        predicted_score = np.clip(predicted_score, 0, 100)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    grade, label, css_class = classify_performance(predicted_score)

    # ── Results area ─────────────────────────────────────────────────────────
    col_result, col_breakdown = st.columns([1, 2])

    with col_result:
        st.markdown("### 🔮 Predicted Score")
        st.markdown(
            f"""<div class="score-display {css_class}">
                {predicted_score:.1f}<br>
                <small style="font-size:1rem">{label}</small>
            </div>""",
            unsafe_allow_html=True,
        )

        # Confidence range (±MAE)
        mae = metadata.get("test_mae", 5)
        st.markdown(
            f"**Confidence Range:** {max(0, predicted_score - mae):.1f} – "
            f"{min(100, predicted_score + mae):.1f} pts"
        )
        st.caption(f"Range = prediction ± model MAE ({mae:.1f} pts)")

    with col_breakdown:
        st.markdown("### 💡 Personalised Insights")
        tips = get_improvement_tips(raw_inputs, predicted_score)
        for tip in tips:
            st.markdown(
                f'<div class="insight-box">{tip}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Feature breakdown radar chart ─────────────────────────────────────────
    st.markdown("### 📊 Your Study Profile vs Average Student")

    # Benchmark averages (from dataset)
    benchmarks = {
        "Study Hours":     (hours_studied,    6.5,  14),
        "Attendance":      (attendance,        78,   100),
        "Sleep":           (sleep_hours,       6.8,  9),
        "Previous Score":  (previous_scores,   65,   100),
        "Internet Usage":  (10 - internet_usage, 7.0, 9.5),  # inverted
        "Tutoring":        (tutoring_sessions, 2.5,  8),
    }

    fig, ax = plt.subplots(figsize=(9, 3))
    labels = list(benchmarks.keys())
    your_vals = [v[0] / v[2] * 100 for v in benchmarks.values()]
    avg_vals  = [v[1] / v[2] * 100 for v in benchmarks.values()]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, your_vals, width, label="You", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, avg_vals,  width, label="Avg Student", color="#AAAAAA", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of Maximum")
    ax.set_title("Study Profile Comparison (% of max possible)")
    ax.legend()
    ax.set_ylim(0, 115)
    ax.bar_label(bars1, fmt="%.0f%%", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.0f%%", padding=2, fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("### 🔑 What Drives Your Score? (Model Feature Importance)")
    importance_path = Path("reports/figures/11_feature_importance.png")
    if importance_path.exists():
        st.image(str(importance_path), use_column_width=True)
    else:
        st.info("Feature importance chart will appear after running the full training pipeline.")

    # ── How it works ──────────────────────────────────────────────────────────
    with st.expander("🤖 How does this model work?", expanded=False):
        st.markdown(f"""
        **Algorithm:** {metadata.get('model_name', 'Machine Learning Model')}

        **Training data:** 2,000 synthetic student records with realistic correlations.

        **Features used ({len(feature_names)}):**
        - *Raw features:* study hours, attendance, sleep, internet usage, previous scores, tutoring
        - *Engineered features:* study efficiency (hours × attendance), distraction ratio, sleep-study balance, academic support index

        **Evaluation:**
        - Test R² = {metadata.get('test_r2', 0):.3f} (explains {metadata.get('test_r2', 0)*100:.1f}% of score variance)
        - RMSE = {metadata.get('test_rmse', 0):.1f} points (average prediction error)
        - 5-fold cross-validation used to prevent overfitting

        **Disclaimer:** This is a demonstration project. Real exam outcomes depend on many
        additional factors not captured here. Use predictions as guidance, not guarantees.
        """)

    st.markdown("---")
    st.caption("Built with Python · scikit-learn · Streamlit | Student Performance Analysis Project")


if __name__ == "__main__":
    main()
