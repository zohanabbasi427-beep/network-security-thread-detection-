import streamlit as st

def apply_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0a0e17;
            color: #e6f1ff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00bfff !important;
        }
        .stButton>button {
            background-color: #00bfff;
            color: #0a0e17;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            padding: 0.5em 1.5em;
        }
        .stButton>button:hover {
            background-color: #0099cc;
            color: white;
        }
        [data-testid="stSidebar"] {
            background-color: #0d1320;
            border-right: 2px solid #00bfff;
        }
        .stMetric {
            background-color: #131a2b;
            border: 1px solid #00bfff;
            border-radius: 10px;
            padding: 10px;
        }
        .stDataFrame {
            border: 1px solid #00bfff;
        }
        a {
            color: #00bfff !important;
        }
        .stAlert {
            background-color: #131a2b;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #00bfff;
        }
        </style>
    """, unsafe_allow_html=True)

PLOT_STYLE = {
    "axes.facecolor": "#0a0e17",
    "figure.facecolor": "#0a0e17",
    "axes.edgecolor": "#00bfff",
    "axes.labelcolor": "#e6f1ff",
    "xtick.color": "#e6f1ff",
    "ytick.color": "#e6f1ff",
    "text.color": "#e6f1ff",
    "axes.titlecolor": "#00bfff",
    "grid.color": "#1e2a3f"
}

# Blue / cyan / orange accent palette for charts
PALETTE = ["#00bfff", "#ff6b35", "#3a86ff", "#ffbe0b", "#06d6a0", "#ef476f"]

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

DATA_URL = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv"
