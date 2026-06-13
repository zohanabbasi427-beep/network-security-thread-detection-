import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils import apply_theme, COLUMNS, DATA_URL, PLOT_STYLE

st.set_page_config(page_title="Model Insights", page_icon="🔬", layout="wide")
apply_theme()

st.title("🔬 Model Insights")

st.markdown("""
This page shows the **confusion matrix**, **classification report**, and
**K-Means clustering** view for the trained models.
""")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    with open("categorical_cols.json", "r") as f:
        categorical_cols = json.load(f)
    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)

    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "Neural Network (MLP)": joblib.load("mlp_model.pkl"),
    }
    return scaler, selector, label_encoders, target_encoder, kmeans, categorical_cols, selected_features, models

@st.cache_data
def get_data():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    df = df.drop(columns=['difficulty'])
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

try:
    (scaler, selector, label_encoders, target_encoder, kmeans,
     categorical_cols, selected_features, models) = load_artifacts()

    X, y = get_data()
    X_enc = X.copy()
    for col in categorical_cols:
        X_enc[col] = label_encoders[col].transform(X_enc[col])

    y_enc = target_encoder.transform(y)

    X_scaled = scaler.transform(X_enc)
    X_selected = selector.transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    plt.rcParams.update(PLOT_STYLE)

    model_choice = st.selectbox("🤖 Select a Model to Inspect", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = list(target_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧮 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_choice} - Confusion Matrix")
        st.pyplot(fig)

    with col2:
        st.subheader("📄 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("🌀 K-Means Clustering (Unsupervised View)")
    st.markdown("""
    This shows how the **K-Means** algorithm groups traffic into 2 clusters
    *without using the actual labels* — useful for exploring whether
    attack and normal traffic naturally separate.
    """)

    clusters = kmeans.predict(X_selected)
    comparison_df = pd.DataFrame({
        'Actual': target_encoder.inverse_transform(y_enc),
        'Cluster': clusters
    })

    st.write("**Cluster vs Actual Label Crosstab:**")
    st.dataframe(pd.crosstab(comparison_df['Cluster'], comparison_df['Actual']),
                  use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sample_idx = np.random.choice(X_selected.shape[0], size=min(3000, X_selected.shape[0]), replace=False)
    ax2.scatter(X_selected[sample_idx, 0], X_selected[sample_idx, 1],
                c=clusters[sample_idx], cmap='cool', alpha=0.5, s=5)
    ax2.set_title("K-Means Clusters (sampled)")
    ax2.set_xlabel(selected_features[0])
    ax2.set_ylabel(selected_features[1])
    st.pyplot(fig2)

except FileNotFoundError as e:
    st.error(f"⚠️ Required file not found: {e}. "
             "Please make sure all .pkl/.json files are in the app directory.")
