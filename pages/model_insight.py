import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

# System path taaki baahar padi utlis.py mil sake
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utlis import apply_theme, COLUMNS, DATA_URL, PLOT_STYLE

st.set_page_config(page_title="Model Insights", page_icon="🔬", layout="wide")
apply_theme()

st.title("🔬 Model Insights")

st.markdown("""
This page shows the **confusion matrix**, **classification report**, and **K-Means clustering** view for the trained models.
""")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)
    with open("all_features.json", "r") as f:
        all_features = json.load(f)
    with open("categorical_cols.json", "r") as f:
        categorical_cols = json.load(f)

    # Models dictionary with safe bypass for MLP
    models = {}
    
    try:
        models["Logistic Regression"] = joblib.load("log_model.pkl")
    except Exception as e:
        st.warning(f"Logistic Regression load nahi ho saka: {e}")

    try:
        models["Random Forest"] = joblib.load("rf_model.pkl")
    except Exception as e:
        st.warning(f"Random Forest load nahi ho saka: {e}")

    # MLP ko safe block mein rakha taaki app crash na ho
    try:
        models["Neural Network (MLP)"] = joblib.load("mlp_model.pkl")
    except Exception as e:
        st.sidebar.error("⚠️ MLP Model version mismatch ki wajah se disabled hai.")
        models["Neural Network (MLP)"] = None

    return (scaler, selector, label_encoders, target_encoder, kmeans,
            categorical_cols, selected_features, models)

try:
    (scaler, selector, label_encoders, target_encoder, kmeans,
     categorical_cols, selected_features, models) = load_artifacts()

    # Filter out models jo sahi se load nahi hue (MLP agar None hai toh dropdown mein nahi aayega)
    available_models = [m for m, obj in models.items() if obj is not None]
    
    st.subheader("📊 Select Model for Insights")
    model_choice = st.selectbox("🤖 Choose a Model", available_models)

    # --- Yahan aapka baaki ka Confusion Matrix aur Clustering ka evaluation code aayega ---
    st.info(f"Showing insights for: **{model_choice}**")
    
    # K-Means section chalane ke liye
    if kmeans is not None:
        st.subheader("🎯 K-Means Clustering View")
        # Aapka K-Means plotting ka code yahan perfectly run karega...

except FileNotFoundError as e:
    st.error(f"⚠️ Required model file not found: {e}. Please check your repository.")
