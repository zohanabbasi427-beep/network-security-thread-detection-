mport streamlit as st

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



    models = {}

    try:

        models["Logistic Regression"] = joblib.load("log_model.pkl")

    except Exception as e:

        st.warning(f"Logistic Regression load nahi ho saka: {e}")



    try:

        models["Random Forest"] = joblib.load("rf_model.pkl")

    except Exception as e:

        st.warning(f"Random Forest load nahi ho saka: {e}")



    # MLP ko safe block mein rakha taaki app crash na ho (NumPy version mismatch fix)

    try:

        models["Neural Network (MLP)"] = joblib.load("mlp_model.pkl")

    except Exception as e:

        st.sidebar.error("⚠️ MLP Model version mismatch ki wajah se disabled hai.")

        models["Neural Network (MLP)"] = None



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

    

    st.subheader("📊 Select Model for Insights")

    model_choice = st.selectbox("🤖 Choose a Model", available_models)



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

        X_scaled = scaler.transform(X_eval_encoded[all_features])

        X_selected = selector.transform(X_scaled)



        # Model predict logic

        current_model = models[model_choice]

        y_pred = current_model.predict(X_selected)



        # --- Visuals Generation ---

        col1, col2 = st.columns(2)



        with col1:

            st.subheader("🎯 Confusion Matrix")

            cm = confusion_matrix(y_eval_encoded, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(

                cm, annot=True, fmt="d", cmap="Blues",

                xticklabels=target_encoder.classes_,

                yticklabels=target_encoder.classes_, ax=ax

            )

            plt.ylabel('Actual Label')

            plt.xlabel('Predicted Label')

            st.pyplot(fig)



        with col2:

            st.subheader("📋 Classification Report")

            report_dict = classification_report(y_eval_encoded, y_pred, target_names=target_encoder.classes_, output_dict=True)

            report_df = pd.DataFrame(report_dict).transpose()

            st.dataframe(report_df.style.format(precision=4), use_container_width=True)



    # --- K-Means Clustering Section ---

    if kmeans is not None:

        st.markdown("---")

        st.subheader("🎯 K-Means Clustering View")

        

        # 🔥 FIX: KMeans ko poore 41 features (X_scaled) ke bajaye selected 15 features (X_selected) de rahe hain

        X_sample_selected = X_selected[:1000] 

        cluster_labels = kmeans.predict(X_sample_selected)

        

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        # Plotting first two columns of selected features

        scatter = ax2.scatter(X_sample_selected[:, 0], X_sample_selected[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)

        ax2.set_title("K-Means Cluster Assignments (First 2 Selected Features)")

        fig2.colorbar(scatter, ax=ax2)

        st.pyplot(fig2)



except FileNotFoundError as e:

    st.error(f"⚠️ Required model file not found: {e}. Please check your repository files.")

except Exception as e:

    st.error(f"⚠️ Unexpected Error: {e}")
