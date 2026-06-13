import streamlit as st
import pandas as pd
from utils import apply_theme, COLUMNS, DATA_URL

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")
apply_theme()

st.title("📊 Dataset Overview")

st.markdown("""
The **NSL-KDD** dataset is a benchmark dataset for network intrusion detection.
It contains network connection records, each labeled as either **normal**
traffic or one of several **attack types** (DoS, Probe, R2L, U2R).
""")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    df['label_binary'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    return df

df = load_data()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", df.shape[0])
with col2:
    st.metric("Total Features", df.shape[1] - 3)
with col3:
    st.metric("Classes", df["label_binary"].nunique())

st.subheader("🔍 Preview of Dataset")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("📈 Statistical Summary")
st.dataframe(df.describe(), use_container_width=True)

st.subheader("🧮 Missing Values")
st.dataframe(df.isnull().sum().rename("Missing Count"), use_container_width=True)

st.subheader("🏷️ Original Attack Type Distribution (Top 10)")
st.dataframe(df['label'].value_counts().head(10).rename("Count"), use_container_width=True)

st.subheader("ℹ️ Column Information")
info_df = pd.DataFrame({
    "Column": df.columns,
    "Dtype": [str(t) for t in df.dtypes]
})
st.dataframe(info_df, use_container_width=True)
