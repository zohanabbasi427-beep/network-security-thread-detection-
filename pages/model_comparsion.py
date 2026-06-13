import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import apply_theme, PLOT_STYLE, PALETTE

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
apply_theme()

st.title("⚖️ Model Comparison")

st.markdown("""
Below is the performance comparison of all trained machine learning models
based on **Accuracy, Precision, Recall, and F1-Score** for detecting
network intrusions.
""")

try:
    results_df = pd.read_csv("results_df.csv", index_col=0)

    st.subheader("📋 Results Table")
    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("📊 Comparison Chart")
    plt.rcParams.update(PLOT_STYLE)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind="bar", ax=ax, color=PALETTE[:4])
    ax.set_title("Model Comparison - Intrusion Detection")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("🏆 Best Model")
    best_model = results_df["F1"].idxmax()
    st.success(f"The best performing model based on F1-Score is: **{best_model}**")
    st.dataframe(results_df.loc[[best_model]].style.format("{:.4f}"), use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ `results_df.csv` not found. Please upload it to the app directory.")
