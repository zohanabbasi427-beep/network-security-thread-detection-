import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from utils import apply_theme, COLUMNS, DATA_URL

st.set_page_config(page_title="Make a Prediction", page_icon="🩺", layout="wide")
apply_theme()

st.title("🛡️ Make a Prediction")

st.markdown("""
Enter network connection feature values below to predict whether the
traffic is **Normal** or a potential **Attack/Intrusion**.
""")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")

    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)
    with open("all_features.json", "r") as f:
        all_features = json.load(f)
    with open("categorical_cols.json", "r") as f:
        categorical_cols = json.load(f)

    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "Neural Network (MLP)": joblib.load("mlp_model.pkl"),
    }
    return (scaler, selector, label_encoders, target_encoder,
            selected_features, all_features, categorical_cols, models)

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    df = df.drop(columns=['difficulty', 'label'])
    return df

try:
    (scaler, selector, label_encoders, target_encoder,
     selected_features, all_features, categorical_cols, models) = load_artifacts()

    X_ref = load_reference_data()

    X_ref_encoded = X_ref.copy()
    for col in categorical_cols:
        X_ref_encoded[col] = label_encoders[col].transform(X_ref_encoded[col])

    st.subheader("🔢 Enter Feature Values")
    st.caption("Only the most relevant features (selected via SelectKBest) are shown. "
                "Other features are auto-filled with dataset mean/mode values.")

    col1, col2 = st.columns(2)
    user_inputs = {}

    for i, feat in enumerate(selected_features):
        col = col1 if i % 2 == 0 else col2

        if feat in categorical_cols:
            options = list(label_encoders[feat].classes_)
            choice = col.selectbox(f"{feat}", options)
            user_inputs[feat] = label_encoders[feat].transform([choice])[0]
        else:
            min_val = float(X_ref_encoded[feat].min())
            max_val = float(X_ref_encoded[feat].max())
            mean_val = float(X_ref_encoded[feat].mean())
            user_inputs[feat] = col.number_input(
                f"{feat}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.5f"
            )

    model_choice = st.selectbox("🤖 Choose a Model", list(models.keys()))

    if st.button("🔍 Detect Threat"):
        full_input = []
        for feat in all_features:
            if feat in selected_features:
                full_input.append(user_inputs[feat])
            elif feat in categorical_cols:
                full_input.append(float(X_ref_encoded[feat].mode()[0]))
            else:
                full_input.append(float(X_ref_encoded[feat].mean()))

        full_input = np.array(full_input).reshape(1, -1)

        scaled_input = scaler.transform(full_input)
        selected_input = selector.transform(scaled_input)

        model = models[model_choice]
        prediction = model.predict(selected_input)[0]
        label_name = target_encoder.inverse_transform([prediction])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(selected_input)[0]

        st.markdown("---")
        st.subheader("🧾 Prediction Result")

        if label_name == "attack":
            st.error("🚨 The model predicts: **ATTACK / INTRUSION DETECTED**")
        else:
            st.success("✅ The model predicts: **NORMAL TRAFFIC**")

        if proba is not None:
            classes = target_encoder.inverse_transform(model.classes_)
            proba_str = " | ".join(
                [f"{c}: {p*100:.2f}%" for c, p in zip(classes, proba)]
            )
            st.write(f"**Confidence:** {proba_str}")

except FileNotFoundError as e:
    st.error(f"⚠️ Required model file not found: {e}. "
             "Please make sure all .pkl and .json files are in the app directory.")
