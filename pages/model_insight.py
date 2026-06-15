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
        models["Logistic Regression"] = None

    # Random Forest Load Logic
    try:
        models["Random Forest"] = joblib.load("rf_model.pkl")
    except Exception as e:
        try:
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

    # Sirf wahi models dropdown mein aayenge jo actually list mein hain
    available_models = ["Logistic Regression", "Random Forest"]
    
    st.subheader("📊 Select Model for Insights")
    model_choice = st.selectbox("🤖 Choose a Model to Analyze", available_models)

    # Pehle check karo ke jo model select hua hai kya uski file mili thi?
    if models.get(model_choice) is None:
        st.error(f"⚠️ **{model_choice}** ki model file (.pkl) folder mein nahi mili! Is model ke graphs dekhne ke liye kripya pehle iski file ko folder mein rakhein.")
    else:
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

            # Model predict logic safely execution
            current_model = models[model_choice]
            y_pred = current_model.predict(X_selected)

            # --- Visuals Generation ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"🎯 Confusion Matrix ({model_choice})")
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
                st.subheader(f"📋 Classification Report ({model_choice})")
                report_dict = classification_report(y_eval_encoded, y_pred, target_names=target_encoder.classes_, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)

    # --- K-Means Clustering Section ---
    if kmeans is not None:
        st.markdown("---")
        st.subheader("🎯 K-Means Clustering View")
        
        X_sample_selected = X_selected if 'X_selected' in locals() else scaler.transform(df_raw.drop(columns=['difficulty', 'label']).copy().pipe(lambda d: d.assign(**{c: label_encoders[c].transform(d[c]) for c in categorical_cols}))[all_features])
        X_sample_selected = selector.transform(X_sample_selected)[:1000]
        
        cluster_labels = kmeans.predict(X_sample_selected)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        scatter = ax2.scatter(X_sample_selected[:, 0], X_sample_selected[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax2.set_title("K-Means Cluster Assignments (First 2 Selected Features)")
        fig2.colorbar(scatter, ax=ax2)
        st.pyplot(fig2)

except FileNotFoundError as e:
    st.error(f"⚠️ Required data file not found: {e}. Please check your dataset path.")
except Exception as e:
    st.error(f"⚠️ Unexpected Error: {e}")
