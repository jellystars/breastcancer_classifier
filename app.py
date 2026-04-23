"""
Breast Cancer KNN Classifier — Streamlit App
Run with: streamlit run app.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,100..700;1,100..700&display=swap');

:root {
  --c1: #e5f3f8; --c2: #9ED7E6; --c3: #7eafbb;
  --c4: #618892; --c5: #46626a; --c6: #2c4045; --c7: #142023;
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', sans-serif !important;
  background-color: #f4f8f9 !important;
  color: var(--c7) !important;
}

/* Hide toolbar, toggle button, footer — sidebar stays open via initial_sidebar_state */
[data-testid="stToolbar"] { display: visible !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
footer { visibility: hidden !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background-color: var(--c7) !important;
  min-width: 210px !important;
  max-width: 210px !important;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
  display: none !important;
}
[data-testid="stSidebar"] .stRadio label {
  display: block !important;
  padding: 9px 14px !important;
  border-radius: 7px !important;
  font-size: 13px !important;
  cursor: pointer !important;
  margin: 2px 0 !important;
  color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio label * {
  color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: var(--c5) !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio > label {
  display: none !important;
}
[data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Metric cards */
[data-testid="metric-container"] {
  background: #ffffff !important;
  border: 1px solid #e2eaec !important;
  border-radius: 10px !important;
  padding: 1rem 1.1rem !important;
}
[data-testid="metric-container"] label {
  color: #8a9fa5 !important; font-size: 11px !important;
  text-transform: uppercase !important; letter-spacing: 0.05em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--c7) !important; font-size: 26px !important; font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  color: var(--c4) !important; font-size: 11px !important;
}

/* Slider label */
[data-testid="stSlider"] label p {
  font-size: 12px !important;
  color: var(--c5) !important;
  font-family: 'IBM Plex Sans', monospace !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  display: none !important;
}
[data-testid="stSlider"] p {
  color: var(--c7) !important;
}
                       
/* Slider filled track */
[data-testid="stSlider"] > div > div > div > div {
  background: var(--c3) !important;
}
/* Slider thumb */
[data-testid="stSlider"] > div > div > div > div > div {
  background: var(--c5) !important;
  border: 2px solid var(--c2) !important;
}

/* Button */
.stButton > button {
  background: var(--c6) !important; color: var(--c1) !important;
  border: none !important; border-radius: 7px !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover { background: var(--c7) !important; border: none !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: #edf4f6 !important; border-radius: 7px !important;
  padding: 3px !important; gap: 3px !important; border: 1px solid #e2eaec !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 5px !important; color: #8a9fa5 !important;
  font-size: 12px !important; font-weight: 500 !important; padding: 5px 14px !important;
}
.stTabs [aria-selected="true"] { background: #ffffff !important; color: var(--c6) !important; }

a.dataset-link {
  color: #7eafbb !important;
  text-decoration: none !important;
  font-weight: 600 !important;
  transition: color 0.2s !important;
}
a.dataset-link:hover {
  color: #46626a !important;
}            

hr { border-color: #e2eaec !important; }
[data-testid="stCaptionContainer"] p { color: #a0b4b9 !important; font-size: 11px !important; }
code, pre { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e2eaec", "axes.labelcolor": "#618892",
    "axes.titlecolor": "#2c4045", "xtick.color": "#8a9fa5",
    "ytick.color": "#8a9fa5", "grid.color": "#edf4f6",
    "text.color": "#2c4045", "font.family": "sans-serif",
})

# ── Train ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    k_range = range(1, 31)
    cv_scores = [
        cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, cv=5).mean()
        for k in k_range
    ]
    best_k = list(k_range)[np.argmax(cv_scores)]
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return data, model, scaler, X_test, y_test, y_pred, best_k, list(k_range), cv_scores

data, model, scaler, X_test, y_test, y_pred, best_k, k_range, cv_scores = load_and_train()
feature_names = data.feature_names
class_names   = data.target_names

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    page = st.radio("nav", ["Predictor", "Results"], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predictor
# ══════════════════════════════════════════════════════════════════════════════
if page == "Predictor":
    st.markdown("<h2 style='color:#142023;font-size:24px;font-weight:600;margin-bottom:4px;'>Tumor Classification</h2>", unsafe_allow_html=True)
    st.markdown("<div style='width:180px;height:3px;background:#7eafbb;border-radius:2px;margin-bottom:24px;'></div>", unsafe_allow_html=True)

    # About card
    with st.container(border=True):
        st.markdown("""
        <p style='font-size:14px;font-weight:600;color:#618892;margin-bottom:8px;'>Overview</p>
        <p style='font-size:14px;color:#46626a;line-height:1.7;'>
          A KNN classifier that predicts whether a breast tumor is malignant or benign 
          based on 30 clinical measurements from the <a href='https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic' target='_blank' class='dataset-link'>Wisconsin Breast Cancer Dataset</a> 
          (569 samples). Adjust the sliders below to run a live prediciton.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

    # Predictor card
    with st.container(border=True):
        SHOW = {
            "mean radius": "Mean Radius",
            "mean texture": "Mean Texture",
            "mean perimeter": "Mean Perimeter",
            "mean area": "Mean Area",
            "mean smoothness": "Mean Smoothness",
            "mean concavity": "Mean Concavity",
        }

        mins  = data.data.min(axis=0)
        maxs  = data.data.max(axis=0)
        means = data.data.mean(axis=0)

        user_input = []
        for i, name in enumerate(feature_names):
            if name in SHOW:
                val = st.slider(SHOW[name], float(mins[i]), float(maxs[i]), float(means[i]), key=name)
                user_input.append((i, val))
            else:
                user_input.append((i, float(means[i])))

        fv = np.array([v for _, v in sorted(user_input)]).reshape(1, -1)
        fs = scaler.transform(fv)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        run = st.button("Run", use_container_width=True)

        if run:
            pred  = model.predict(fs)[0]
            proba = model.predict_proba(fs)[0]
            conf  = max(proba)
            label = class_names[pred]
            if pred == 1:
                st.success(f"{label} — {conf:.1%} confidence")
            else:
                st.error(f"{label} — {conf:.1%} confidence")
            
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.markdown("<h2 style='color:#142023;font-size:24px;font-weight:600;margin-bottom:4px;'>Results</h2>", unsafe_allow_html=True)
    st.markdown("<div style='width:60px;height:3px;background:#7eafbb;border-radius:2px;margin-bottom:24px;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    c2.metric("Best K", str(best_k))
    c3.metric("Train Samples", "455")
    c4.metric("Features", "30")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("<p style='font-size:14px;font-weight:600;color:#2c4045;margin-bottom:14px;'>Confusion Matrix</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap=sns.light_palette("#7eafbb", as_cmap=True),
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="#f4f8f9",
                annot_kws={"size": 14, "weight": "bold", "color": "#2c4045"},
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual", fontsize=11)
            ax.tick_params(colors="#618892", labelsize=11)
            fig.tight_layout()
            st.pyplot(fig)

    with right:
        with st.container(border=True):
            st.markdown("<p style='font-size:14px;font-weight:600;color:#2c4045;margin-bottom:14px;'>CV Accuracy vs K</p>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
            ax2.plot(k_range, cv_scores, color="#7eafbb", linewidth=2,
                     marker="o", markersize=4,
                     markerfacecolor="#ffffff", markeredgewidth=1.5, markeredgecolor="#7eafbb")
            ax2.axvline(best_k, color="#46626a", linestyle="--", linewidth=1.5, label=f"Best K = {best_k}")
            ax2.fill_between(k_range, cv_scores, min(cv_scores) - 0.005, alpha=0.08, color="#7eafbb")
            ax2.set_xlabel("K", fontsize=11)
            ax2.set_ylabel("CV Accuracy", fontsize=11)
            ax2.legend(fontsize=10, framealpha=0.5)
            ax2.grid(True, alpha=0.4)
            ax2.spines[["top", "right"]].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

    left2, right2 = st.columns(2, gap="medium")

    with left2:
        with st.container(border=True):
            st.markdown("<p style='font-size:14px;font-weight:600;color:#2c4045;margin-bottom:14px;'>Class Distribution</p>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
            counts = [(data.target == 0).sum(), (data.target == 1).sum()]
            bars = ax3.bar(class_names, counts, color=["#e24b4a", "#7eafbb"], width=0.5, edgecolor="none")
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(count), ha="center", va="bottom", fontsize=12, fontweight="bold", color="#2c4045")
            ax3.set_ylabel("Count", fontsize=11)
            ax3.spines[["top", "right"]].set_visible(False)
            ax3.grid(axis="y", alpha=0.3)
            fig3.tight_layout()
            st.pyplot(fig3)

    with right2:
        with st.container(border=True):
            st.markdown("<p style='font-size:14px;font-weight:600;color:#2c4045;margin-bottom:14px;'>Classification Report</p>", unsafe_allow_html=True)
            st.code(classification_report(y_test, y_pred, target_names=class_names))