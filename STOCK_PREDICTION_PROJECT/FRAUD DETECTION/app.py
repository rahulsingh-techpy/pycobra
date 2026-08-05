"""
Credit Card Fraud Detection Dashboard
--------------------------------------
A Python + data-analytics Streamlit application that generates a realistic
synthetic credit-card-transaction dataset (same schema as the popular
Kaggle "creditcard.csv": Time, V1-V28 PCA features, Amount, Class),
performs EDA, trains and compares two ML models, and lets you interactively
score transactions for fraud risk.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

# ----------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Shield | Credit Card Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#6C5CE7"
ACCENT_DARK = "#4834D4"
GOOD = "#00B894"
BAD = "#D63031"
BG_CARD = "#161B29"

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#11151F",
    "figure.facecolor": "#11151F",
    "axes.edgecolor": "#2A3145",
    "axes.labelcolor": "#E4E7F1",
    "text.color": "#E4E7F1",
    "xtick.color": "#B8BCD4",
    "ytick.color": "#B8BCD4",
    "grid.color": "#232A3D",
})
plt.rcParams["font.family"] = "DejaVu Sans"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 15% 0%, #1B2136 0%, #0B0E17 55%);
    }}
    section[data-testid="stSidebar"] {{
        background: #0D111C;
        border-right: 1px solid #232A3D;
    }}
    h1, h2, h3 {{
        font-family: 'Trebuchet MS', sans-serif;
        letter-spacing: 0.3px;
    }}
    .hero {{
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(108,92,231,0.25);
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p {{ margin: 6px 0 0 0; opacity: 0.9; font-size: 0.95rem; }}
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid #232A3D;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }}
    .metric-card .label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9AA0C0;
        margin-bottom: 6px;
    }}
    .metric-card .value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
    }}
    .badge-safe {{
        display:inline-block; padding: 10px 18px; border-radius: 999px;
        background: rgba(0,184,148,0.15); color: {GOOD}; font-weight:700;
        border: 1px solid rgba(0,184,148,0.4);
    }}
    .badge-fraud {{
        display:inline-block; padding: 10px 18px; border-radius: 999px;
        background: rgba(214,48,49,0.15); color: {BAD}; font-weight:700;
        border: 1px solid rgba(214,48,49,0.4);
    }}
    div[data-testid="stMetricValue"] {{ color: white; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA GENERATION (cached so it's stable across reruns / widget interactions)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n_samples: int, fraud_ratio: float, seed: int = 42) -> pd.DataFrame:
    """Synthesize a Kaggle-creditcard-style transaction dataset."""
    rng = np.random.default_rng(seed)

    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    time_normal = rng.uniform(0, 172800, n_normal)  # 2-day window, seconds
    time_fraud = rng.uniform(0, 172800, n_fraud)

    amount_normal = rng.gamma(2.0, 50, n_normal)
    amount_fraud = rng.gamma(1.2, 250, n_fraud)

    pca_normal = rng.normal(0, 1, (n_normal, 28))
    shift = rng.choice([-1, 1], 28) * rng.uniform(0.6, 1.6, 28)
    pca_fraud = rng.normal(0, 1.3, (n_fraud, 28)) + shift

    df_normal = pd.DataFrame(pca_normal, columns=[f"V{i}" for i in range(1, 29)])
    df_normal["Time"] = time_normal
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(pca_fraud, columns=[f"V{i}" for i in range(1, 29)])
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return df[cols]


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, seed: int = 42):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=seed
    )

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    results = {}
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {"model": model, "proba": proba}

    return {
        "scaler": scaler, "X_test": X_test, "y_test": y_test,
        "results": results, "feature_cols": list(X.columns),
    }


def metrics_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pred": pred,
    }


# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Fraud Shield")
    st.caption("Synthetic-data credit card fraud analytics")
    st.divider()

    n_samples = st.slider("Number of transactions", 5000, 30000, 15000, step=1000)
    fraud_ratio = st.slider("Fraud rate (%)", 0.5, 5.0, 1.7, step=0.1) / 100
    st.divider()

    model_choice = st.selectbox("Model for scoring", ["Random Forest", "Logistic Regression"])
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.01,
                           help="Probability above which a transaction is flagged as fraud.")
    st.divider()
    st.caption("Dataset schema mirrors the well-known Kaggle 'creditcard.csv' "
               "(Time, V1–V28 PCA components, Amount, Class) so you can swap in "
               "real data later with zero code changes.")

df = generate_data(n_samples, fraud_ratio)
bundle = train_models(df)
proba = bundle["results"][model_choice]["proba"]
y_test = bundle["y_test"]
m = metrics_at_threshold(y_test, proba, threshold)

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>🛡️ Credit Card Fraud Detection Analytics</h1>
    <p>Synthetic-data pipeline · Exploratory analysis · Model comparison · Live transaction scoring</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value in zip(
    [k1, k2, k3, k4, k5],
    ["Transactions", "Fraud Cases", "Fraud Rate", f"{model_choice} ROC-AUC", "F1 @ threshold"],
    [
        f"{len(df):,}",
        f"{int(df['Class'].sum()):,}",
        f"{df['Class'].mean()*100:.2f}%",
        f"{roc_auc_score(y_test, proba):.3f}",
        f"{m['f1']:.3f}",
    ],
):
    col.markdown(f"""<div class="metric-card"><div class="label">{label}</div>
    <div class="value">{value}</div></div>""", unsafe_allow_html=True)

st.write("")

tab_eda, tab_models, tab_predict, tab_data = st.tabs(
    ["📊 Explore Data", "🤖 Model Performance", "🔍 Score a Transaction", "🗂️ Raw Data"]
)

# ----------------------------------------------------------------------------
# TAB 1 — EDA
# ----------------------------------------------------------------------------
with tab_eda:
    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.subheader("Class Balance")
        fig, ax = plt.subplots(figsize=(4.2, 4))
        counts = df["Class"].value_counts().sort_index()
        colors = [GOOD, BAD]
        ax.bar(["Legitimate", "Fraud"], counts.values, color=colors, width=0.55)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10, color="white")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        st.pyplot(fig, use_container_width=True)
        st.caption("Log scale — fraud is a small minority, as in real-world data.")

    with c2:
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(6.2, 4))
        sns.kdeplot(df.loc[df.Class == 0, "Amount"].clip(upper=800), fill=True,
                    color=GOOD, alpha=0.4, label="Legitimate", ax=ax)
        sns.kdeplot(df.loc[df.Class == 1, "Amount"].clip(upper=800), fill=True,
                    color=BAD, alpha=0.4, label="Fraud", ax=ax)
        ax.set_xlabel("Amount ($, clipped at 800)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Transactions Over Time")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        hours = (df["Time"] / 3600).astype(int)
        hourly = df.groupby([hours, "Class"]).size().unstack(fill_value=0)
        ax.plot(hourly.index, hourly.get(0, 0), color=GOOD, label="Legitimate", linewidth=1.6)
        ax2 = ax.twinx()
        ax2.plot(hourly.index, hourly.get(1, 0), color=BAD, label="Fraud", linewidth=1.6)
        ax.set_xlabel("Hour of window")
        ax.set_ylabel("Legitimate count", color=GOOD)
        ax2.set_ylabel("Fraud count", color=BAD)
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)

    with c4:
        st.subheader("Top Feature Correlations with Fraud")
        corr = df.drop(columns=["Time"]).corr(numeric_only=True)["Class"].drop("Class")
        top = corr.abs().sort_values(ascending=False).head(10).index
        fig, ax = plt.subplots(figsize=(6, 3.6))
        vals = corr.loc[top]
        bar_colors = [BAD if v > 0 else ACCENT for v in vals]
        ax.barh(top[::-1], vals.reindex(top[::-1]), color=bar_colors)
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.set_xlabel("Correlation with Class")
        st.pyplot(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab_models:
    st.subheader(f"Evaluation — {model_choice} @ threshold {threshold:.2f}")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    mc2.metric("Precision", f"{m['precision']*100:.2f}%")
    mc3.metric("Recall", f"{m['recall']*100:.2f}%")
    mc4.metric("F1-score", f"{m['f1']*100:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, m["pred"])
        fig, ax = plt.subplots(figsize=(4.4, 4))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="mako",
                    xticklabels=["Pred: Legit", "Pred: Fraud"],
                    yticklabels=["True: Legit", "True: Fraud"], ax=ax, cbar=False,
                    annot_kws={"size": 13})
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("**ROC Curve — both models**")
        fig, ax = plt.subplots(figsize=(4.6, 4))
        for name, color in [("Logistic Regression", ACCENT), ("Random Forest", GOOD)]:
            p = bundle["results"][name]["proba"]
            fpr, tpr, _ = roc_curve(y_test, p)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test,p):.3f})",
                    color=color, linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig, use_container_width=True)

    st.markdown("**Precision–Recall Trade-off**")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    prec, rec, thr = precision_recall_curve(y_test, proba)
    ax.plot(thr, prec[:-1], label="Precision", color=ACCENT, linewidth=2)
    ax.plot(thr, rec[:-1], label="Recall", color=BAD, linewidth=2)
    ax.axvline(threshold, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Decision threshold")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.caption("Move the **Decision threshold** slider in the sidebar to see the "
               "precision/recall trade-off — lower thresholds catch more fraud but "
               "raise false alarms, and vice versa.")

# ----------------------------------------------------------------------------
# TAB 3 — LIVE SCORING
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Score a Transaction")
    st.caption("Pick a real row from the test set, or generate a random one, and see the model's fraud probability.")

    colA, colB = st.columns([1, 2])
    with colA:
        mode = st.radio("Sample source", ["Random legitimate", "Random fraud", "Fully random"])
        if st.button("🎲 Draw sample", use_container_width=True):
            X_test = bundle["X_test"]
            if mode == "Random legitimate":
                idx = y_test[y_test == 0].sample(1).index
            elif mode == "Random fraud":
                idx = y_test[y_test == 1].sample(1).index
            else:
                idx = y_test.sample(1).index
            st.session_state["sample_idx"] = idx[0]

    idx = st.session_state.get("sample_idx", bundle["X_test"].index[0])
    row = bundle["X_test"].loc[[idx]]
    true_label = y_test.loc[idx]
    model = bundle["results"][model_choice]["model"]
    prob = model.predict_proba(row)[0, 1]

    with colB:
        raw_amount = df.loc[idx, "Amount"]
        st.markdown(f"**Transaction Amount:** ${raw_amount:,.2f}  |  **True label:** "
                     f"{'🚩 Fraud' if true_label == 1 else '✅ Legitimate'}")
        st.progress(min(max(prob, 0.0), 1.0))
        if prob >= threshold:
            st.markdown(f'<span class="badge-fraud">⚠️ FLAGGED AS FRAUD — {prob*100:.1f}% risk</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-safe">✅ LOOKS LEGITIMATE — {prob*100:.1f}% risk</span>',
                        unsafe_allow_html=True)

        with st.expander("View raw feature values"):
            st.dataframe(row.T.rename(columns={row.index[0]: "value"}), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — RAW DATA
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Underlying Dataset")
    st.caption("Synthetic data generated fresh for this session (schema-compatible with Kaggle's creditcard.csv). "
               "Swap `generate_data()` for `pd.read_csv('creditcard.csv')` to use real data.")
    show_fraud_only = st.checkbox("Show fraud transactions only")
    view = df[df.Class == 1] if show_fraud_only else df
    st.dataframe(view.head(500), use_container_width=True, height=420)
    st.download_button("⬇️ Download this dataset as CSV", df.to_csv(index=False),
                        file_name="synthetic_credit_card_transactions.csv", mime="text/csv")
