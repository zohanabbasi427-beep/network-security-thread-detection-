import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utlis import apply_theme, COLUMNS, DATA_URL, PLOT_STYLE, PALETTE

st.set_page_config(page_title="EDA & Visualizations", page_icon="📉", layout="wide")
apply_theme()

st.title("📉 Exploratory Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    df['label_binary'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    return df

df = load_data()
plt.rcParams.update(PLOT_STYLE)

# ---------- Class Distribution ----------
st.subheader("🟦 Normal vs Attack Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="label_binary", data=df, palette=[PALETTE[0], PALETTE[1]], ax=ax)
ax.set_title("Normal vs Attack Traffic")
st.pyplot(fig)

# ---------- Protocol Type ----------
st.subheader("🌐 Protocol Type Distribution")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.countplot(x="protocol_type", data=df, hue="label_binary",
               palette=[PALETTE[0], PALETTE[1]], ax=ax2)
ax2.set_title("Protocol Type vs Traffic Label")
st.pyplot(fig2)

# ---------- Feature Distribution ----------
st.subheader("📊 Numeric Feature Distribution")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
feature = st.selectbox("Select a feature to visualize:", numeric_cols)

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x=feature, hue="label_binary", kde=True,
              palette=[PALETTE[0], PALETTE[1]], ax=ax3, bins=40)
ax3.set_title(f"Distribution of {feature}")
st.pyplot(fig3)

# ---------- Correlation Heatmap ----------
st.subheader("🔥 Correlation Heatmap (Numeric Features)")
fig4, ax4 = plt.subplots(figsize=(16, 12))
sns.heatmap(df[numeric_cols].corr(), cmap="mako", annot=False, ax=ax4)
st.pyplot(fig4)

# ---------- Boxplot ----------
st.subheader("📦 Boxplot by Label")
feature2 = st.selectbox("Select feature for boxplot:", numeric_cols, key="box")
fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.boxplot(x="label_binary", y=feature2, data=df, palette=[PALETTE[0], PALETTE[1]], ax=ax5)
st.pyplot(fig5)
