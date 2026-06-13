import streamlit as st
from utils import apply_theme

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

apply_theme()

st.title("🛡️ Network Security Threat Detection")
st.markdown("### AI-Powered Intrusion Detection System (NIDS)")

st.markdown("""
Welcome to the **Network Intrusion Detection App**.

This application uses **Machine Learning** trained on the **NSL-KDD** dataset
to detect whether network traffic is **normal** or a potential **attack/intrusion**.

---

### 📋 Navigate using the sidebar:
1. **Home** — Overview of the project
2. **Dataset Overview** — Explore the NSL-KDD dataset
3. **EDA & Visualizations** — Charts and traffic distributions
4. **Model Comparison** — Compare performance of all trained models
5. **Make a Prediction** — Enter traffic features and detect threats
6. **Model Insights** — Confusion matrices, clustering & detailed metrics

---

### 🛠️ Tech Stack
Python · Scikit-learn (Logistic Regression, Random Forest, MLP, K-Means) ·
Pandas · Seaborn · Matplotlib · Streamlit

### 📊 Dataset
**NSL-KDD** — ~125,000 network connection records, 41 features
(duration, protocol, bytes transferred, error rates, login attempts, etc.),
labeled as **normal** or **attack**.

---

⚠️ **Disclaimer:** This tool is for **educational purposes only** —
a demonstration of an ML-based intrusion detection pipeline, not a
production-grade security system.
""")

st.success("Use the sidebar (left) to navigate through the app pages.")
