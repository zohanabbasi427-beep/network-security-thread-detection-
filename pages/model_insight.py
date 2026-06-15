import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

# System path set kar rahe hain taaki baahar padi utlis.py mil sake
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utlis import apply_theme, COLUMNS, DATA_URL, PLOT_STYLE

st.set_page_config(page_title="Model Insights", page_icon="🔬", layout="wide")
apply_theme()

st.title("🔬 Model Insights")
st.markdown("""
This page shows the **confusion matrix** and **classification report** for the trained Machine Learning models.
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

    models = {}
    
    # Logistic Regression Load Logic
    try:
        models["Logistic Regression"] = joblib.load("log_model.pkl")
    except Exception as e:
        st.warning(f"⚠️ Logistic Regression load nahi ho saka: {e}")
        models["Logistic Regression"] = None

    # Random Forest Load Logic (Aapki file ka exact naam log_model ya rf_model jo bhi ho)
    try:
        models["Random Forest"] = joblib.load("rf_model.pkl")
    except Exception as e:
        try:
            # Agar rf_model.pkl nahi mili toh back-up ke taur par check karega
            models["Random Forest"] = joblib.load("random_forest.pkl")
        except:
            models["Random Forest"] = None

    return (scaler, selector, label_encoders, target_encoder, kmeans,
            categorical_cols, selected_features, models, all_features)

@st.cache_data
def load_evaluation_data():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    return df

try:
    (scaler, selector, label_encoders, target_encoder, kmeans,
     categorical_cols, selected_features, models, all_features) = load_artifacts()

    df_raw = load_evaluation_data()

    # Filter out models jo sahi se load hue hain
    available_models = [m for m, obj in models.items() if obj is not None]
    
    if available_models:
        st.subheader("📊 Select Model for Insights")
        model_choice = st.selectbox("🤖 Choose a Model to Analyze", available_models)

        with st.spinner("Generating metrics and visualizations..."):
            # Target aur features alag karna
            X_eval = df_raw.drop(columns=['difficulty', 'label'])
            
            # Har attack type ko 'attack' label mein map kar rahe hain
            y_eval = df_raw['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

            # Categorical variables encode karna
            X_eval_encoded = X_eval.copy()
            for col in categorical_cols:
                X_eval_encoded[col] = label_encoders[col].transform(X_eval_encoded[col])

            # Target ko numerical encoding dena (0 aur 1)
            y_eval_encoded = target_encoder.transform(y_eval)

            # Pipeline transformation (Scaling -> Feature Selection)
            X_scaled = scaler
