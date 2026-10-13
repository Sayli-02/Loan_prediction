import json
import os
import re
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

MODEL_PATH = "loan_model.pkl"
DATASET_PATH = "loan_approval_dataset.csv"

FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]

# (display name, type / encoding, description)
FEATURE_INFO = {
    "no_of_dependents": ("Dependents", "Whole number", "Number of financial dependents"),
    "education": ("Education", "Category: Graduate = 1, Not Graduate = 0", "Applicant education level"),
    "self_employed": ("Self Employed", "Category: Yes = 1, No = 0", "Employment category"),
    "income_annum": ("Annual Income", "Number", "Annual applicant income"),
    "loan_amount": ("Loan Amount", "Number", "Requested loan amount"),
    "loan_term": ("Loan Term", "Number", "Repayment term of the loan"),
    "cibil_score": ("CIBIL Score", "Number from 0 to 900", "Applicant credit score"),
    "residential_assets_value": ("Residential Assets", "Number", "Value of residential assets"),
    "commercial_assets_value": ("Commercial Assets", "Number", "Value of commercial assets"),
    "luxury_assets_value": ("Luxury Assets", "Number", "Value of luxury assets"),
    "bank_asset_value": ("Bank Assets", "Number", "Value of bank assets"),
}

PAGES = ["Overview", "Dataset Preview", "Manual Prediction", "Bulk Prediction", "Analytics", "About"]

# chart colours
NAVY = "#0f2a47"
BLUE = "#1d4ed8"
TEAL = "#0d9488"
GREEN = "#15803d"
RED = "#b91c1c"

# ---------------- CSS ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, button, input, select, textarea {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ================= PAGE ================= */
    .stApp { background-color: #f6f8fb; color: #0f2a47; }
    .block-container { max-width: 1280px; padding: 3.2rem 3rem 2rem 3rem; }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent; }
    /* remove the unused Deploy / menu buttons at the top right */
    div[data-testid="stToolbar"], div[data-testid="stDeployButton"], .stDeployButton,
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display: none !important; }

    h1, h2, h3, h4 { color: #0f2a47; font-weight: 700 !important; letter-spacing: -0.3px; }
    h2 { font-size: 24px !important; }
    h3 {
        font-size: 19px !important;
        border-left: 4px solid #0d9488;
        padding-left: 12px !important;
        margin: 6px 0 8px 0 !important;
        line-height: 1.3 !important;
    }
    p, li { font-size: 15px; line-height: 1.7; color: #475569; }
    .stMarkdown p { text-align: justify; }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2a47 0%, #0a1e34 100%);
        border-right: 0;
    }
    section[data-testid="stSidebar"] > div { padding: 1.8rem 1.1rem; }

    .brand {
        display: flex; align-items: center; gap: 14px;
        padding: 6px 10px 22px 10px; margin-bottom: 18px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }
    .brand-logo {
        width: 46px; height: 46px; flex: 0 0 46px; border-radius: 12px;
        background: linear-gradient(135deg, #2563eb, #0d9488);
        color: #ffffff; font-size: 17px; font-weight: 800; letter-spacing: 0.5px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.30);
    }
    .brand-title { font-size: 19px; font-weight: 800; color: #ffffff; line-height: 1.2; letter-spacing: -0.3px; }
    .brand-sub { font-size: 12.5px; color: #9fb3c8; margin-top: 3px; line-height: 1.4; }

    /* the widget title ("Navigation") was being styled like a menu item, so hide it */
    section[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stRadio > label { display: none !important; }

    section[data-testid="stSidebar"] .stRadio > div { gap: 6px; }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        display: flex !important; align-items: center; width: 100%;
        padding: 12px 14px !important; border-radius: 10px;
        border: 1px solid transparent; cursor: pointer; transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child { display: none; }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]::before {
        content: ""; width: 14px; height: 14px; flex: 0 0 14px; border-radius: 50%;
        border: 2px solid #5b7794; background: transparent; margin-right: 14px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] p {
        font-size: 15px; font-weight: 500; color: #c5d3e3; text-align: left; margin: 0;
    }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover { background: rgba(255, 255, 255, 0.07); border-color: rgba(255, 255, 255, 0.10); }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover::before { border-color: #9fb3c8; }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(255, 255, 255, 0.10); border-color: rgba(255, 255, 255, 0.18); box-shadow: none;
    }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked)::before {
        background: #3b82f6; border-color: #3b82f6; box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.28);
    }
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) p { color: #ffffff; font-weight: 700; }

    /* ================= PAGE HEADER ================= */
    .page-header {
        background: #ffffff; border: 1px solid #dde5ef; border-left: 6px solid #1d4ed8;
        border-radius: 12px; padding: 24px 30px; margin-bottom: 26px;
        box-shadow: 0 2px 8px rgba(15, 42, 71, 0.05);
    }
    .page-title { font-size: 32px; font-weight: 800; color: #0f2a47; letter-spacing: -0.8px; line-height: 1.2; }
    .page-subtitle { font-size: 15.5px; color: #64748b; margin-top: 8px; line-height: 1.6; }

    /* ================= CARDS ================= */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border: 1px solid #dde5ef !important; border-radius: 12px;
        padding: 6px 8px; box-shadow: 0 2px 8px rgba(15, 42, 71, 0.04);
    }

    /* ================= METRICS ================= */
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dde5ef; border-top: 4px solid #1d4ed8;
        border-radius: 12px; padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(15, 42, 71, 0.05); transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { box-shadow: 0 8px 20px rgba(29, 78, 216, 0.12); }
    div[data-testid="stMetricLabel"] p { font-size: 14px !important; font-weight: 600; color: #64748b !important; }
    div[data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 800 !important; color: #0f2a47 !important; }

    /* ================= INPUTS ================= */
    .stNumberInput label p, .stSelectbox label p, .stFileUploader label p {
        font-size: 14px !important; font-weight: 600 !important; color: #334155 !important;
    }
    .stNumberInput input, div[data-baseweb="select"] > div {
        border-radius: 8px !important; border-color: #c9d5e3 !important;
        font-size: 15px !important; background: #f8fafc !important;
    }
    .stNumberInput input:focus { border-color: #1d4ed8 !important; background: #ffffff !important; }
    div[data-testid="stFileUploader"] section {
        background: #f8fafc; border: 2px dashed #93c5fd; border-radius: 12px; padding: 22px;
    }

    /* ================= BUTTONS ================= */
    .stButton > button[kind="primary"], .stDownloadButton > button {
        background: linear-gradient(135deg, #0f2a47, #1d4ed8); color: #ffffff; border: none;
        height: 3.2em; font-size: 16px; font-weight: 700; border-radius: 10px;
        box-shadow: 0 6px 14px rgba(15, 42, 71, 0.22); transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0a1e34, #1e40af); color: #ffffff; border: none;
        transform: translateY(-1px);
    }

    /* ================= RESULT BOX ================= */
    .result-approved, .result-rejected { padding: 24px 28px; border-radius: 12px; margin: 10px 0 20px 0; }
    .result-approved { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 7px solid #15803d; }
    .result-rejected { background: #fef2f2; border: 1px solid #fecaca; border-left: 7px solid #b91c1c; }
    .result-title { font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }
    .result-approved .result-title { color: #15803d; }
    .result-rejected .result-title { color: #b91c1c; }
    .result-text { font-size: 15px; color: #475569; margin-top: 5px; }

    /* ================= TABLES AND CHARTS ================= */
    div[data-testid="stDataFrame"] {
        border: 1px solid #dde5ef; border-radius: 10px; overflow: hidden; background: #ffffff;
    }
    div[data-testid="stImage"] img, div[data-testid="stPyplot"] img {
        background: #ffffff; border: 1px solid #dde5ef; border-radius: 12px; padding: 8px;
        box-shadow: 0 2px 8px rgba(15, 42, 71, 0.04);
    }
    .stAlert { border-radius: 10px; font-size: 14.5px; }
    div[data-testid="stVerticalBlock"] { gap: 0.9rem; }

    /* ================= ABOUT PAGE ================= */
    .about-intro { font-size: 16px; line-height: 1.8; color: #334155; text-align: justify; margin: 0; }
    .about-intro b { color: #0f2a47; }

    .grid-3, .grid-4, .flow { display: grid; gap: 16px; }
    .grid-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-4 { grid-template-columns: repeat(4, 1fr); }
    .flow { grid-template-columns: repeat(6, 1fr); gap: 12px; }

    .feat-card {
        background: #ffffff; border: 1px solid #dde5ef; border-top: 4px solid #0d9488;
        border-radius: 12px; padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(15, 42, 71, 0.04);
    }
    .feat-title { font-size: 16px; font-weight: 700; color: #0f2a47; margin-bottom: 6px; }
    .feat-text { font-size: 14px; line-height: 1.65; color: #64748b; }

    .tech-card {
        background: #ffffff; border: 1px solid #dde5ef; border-radius: 12px;
        padding: 16px 18px; text-align: left;
    }
    .tech-name { font-size: 16px; font-weight: 800; color: #1d4ed8; }
    .tech-role { font-size: 13px; color: #64748b; margin-top: 3px; line-height: 1.5; }

    .flow-step {
        background: #ffffff; border: 1px solid #dde5ef; border-radius: 12px;
        padding: 18px 16px; box-shadow: 0 2px 8px rgba(15, 42, 71, 0.04);
    }
    .flow-num {
        width: 30px; height: 30px; border-radius: 50%; background: #0f2a47; color: #ffffff;
        font-size: 14px; font-weight: 700; display: flex; align-items: center; justify-content: center;
        margin-bottom: 12px;
    }
    .flow-title { font-size: 14.5px; font-weight: 700; color: #0f2a47; margin-bottom: 4px; }
    .flow-text { font-size: 12.8px; line-height: 1.55; color: #64748b; }

    .file-row {
        display: flex; align-items: center; gap: 18px; padding: 14px 4px;
        border-bottom: 1px solid #eef2f7;
    }
    .file-row:last-child { border-bottom: 0; }
    .file-name {
        flex: 0 0 280px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13.5px; font-weight: 600;
        color: #0f2a47; background: #eef4fb; border-radius: 6px; padding: 6px 12px;
    }
    .file-desc { font-size: 14px; color: #475569; line-height: 1.55; }

    .note-box {
        background: #f0fdfa; border: 1px solid #99f6e4; border-left: 6px solid #0d9488;
        border-radius: 12px; padding: 20px 26px;
    }
    .note-title { font-size: 16px; font-weight: 700; color: #0f2a47; margin-bottom: 6px; }
    .note-text { font-size: 14.5px; line-height: 1.7; color: #475569; text-align: justify; }


    /* fixed sidebar: remove the collapse arrow and empty header */
    div[data-testid="stSidebarNav"], div[data-testid="stSidebarNavSeparator"],
    ul[data-testid="stSidebarNavItems"], div[data-testid="stSidebarNavLink"] { display: none !important; }

    /* remove Streamlit's top bar buttons (Deploy, menu, status) in all versions */
    div[data-testid="stToolbarActions"],
    div[data-testid="stAppDeployButton"], div[data-testid="stMainMenu"], div[data-testid="stDeployButton"],
    .stDeployButton, div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display: none !important; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    .kv-row { display: flex; justify-content: space-between; gap: 16px; padding: 11px 2px; border-bottom: 1px solid #eef2f7; font-size: 14.5px; }
    .kv-row:last-child { border-bottom: 0; }
    .kv-key { color: #64748b; }
    .kv-val { color: #0f2a47; font-weight: 700; text-align: right; }
    .pill { margin-left: auto; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 99px; white-space: nowrap; }
    .pill-ok { background: #dcfce7; color: #15803d; }
    .pill-bad { background: #fee2e2; color: #b91c1c; }
    .pill-grey { background: #e8eef5; color: #475569; }


    /* ================= SIDEBAR OPEN / CLOSE BUTTON ================= */
    div[data-testid="stSidebarHeader"] {
        background: transparent; height: auto; min-height: 0;
        padding: 0.9rem 1rem 0 1rem; margin-bottom: 0;
        display: flex; justify-content: flex-end;
    }
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarExpandButton"],
    div[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid="stExpandSidebarButton"] button, button[data-testid="stExpandSidebarButton"] {
        visibility: visible !important; opacity: 1 !important; display: flex !important;
        align-items: center; justify-content: center;
        width: 40px; height: 40px; border-radius: 10px;
        background: #ffffff !important; color: #0f2a47 !important;
        border: 1px solid #dde5ef !important;
        box-shadow: 0 2px 8px rgba(15, 42, 71, 0.10);
        transition: all 0.15s ease;
    }
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stSidebarExpandButton"]:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover,
    div[data-testid="stExpandSidebarButton"] button:hover, button[data-testid="stExpandSidebarButton"]:hover {
        background: #0f2a47 !important; color: #ffffff !important; border-color: #0f2a47 !important;
    }


    /* reopen button: always visible at the top-left while the sidebar is closed */
    div[data-testid="stSidebarCollapsedControl"], div[data-testid="collapsedControl"],
    div[data-testid="stExpandSidebarButton"], button[data-testid="stExpandSidebarButton"] {
        position: fixed !important; top: 14px; left: 16px; z-index: 999999;
        visibility: visible !important; opacity: 1 !important; display: flex !important;
        padding: 0 !important; background: transparent;
    }
    div[data-testid="stSidebarCollapsedControl"] button, div[data-testid="collapsedControl"] button,
    div[data-testid="stExpandSidebarButton"] button {
        visibility: visible !important; opacity: 1 !important; display: flex !important;
        width: 40px; height: 40px; border-radius: 10px;
        background: #ffffff !important; color: #0f2a47 !important;
        border: 1px solid #dde5ef !important; box-shadow: 0 2px 8px rgba(15, 42, 71, 0.10);
    }

    /* ================= ALIGNMENT, JUSTIFY, EQUAL SIZES ================= */
    .stMarkdown p, .stMarkdown li, .page-subtitle, .about-intro, .feat-text, .tech-role,
    .flow-text, .file-desc, .note-text, .result-text, .stCaption, div[data-testid="stCaptionContainer"] {
        text-align: justify; text-justify: inter-word; hyphens: auto;
    }
    .stMarkdown li { margin-bottom: 4px; }

    /* one body size for card text */
    .feat-text, .tech-role, .file-desc, .note-text, .flow-text { font-size: 14px; line-height: 1.65; }
    .flow-text { font-size: 13.5px; }

    /* equal height cards in every row */
    div[data-testid="stHorizontalBlock"] { align-items: stretch; }
    div[data-testid="stColumn"], div[data-testid="column"] { display: flex; flex-direction: column; }
    div[data-testid="stColumn"] > div, div[data-testid="column"] > div { flex: 1 1 auto; }
    div[data-testid="stColumn"] > div > div[data-testid="stVerticalBlock"],
    div[data-testid="column"] > div > div[data-testid="stVerticalBlock"] { height: 100%; }
    div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }

    div[data-testid="stMetric"] { min-height: 112px; }
    .feat-card, .tech-card, .flow-step { height: 100%; box-sizing: border-box; }
    .tech-card { display: flex; flex-direction: column; justify-content: center; }
    .stButton > button[kind="primary"], .stDownloadButton > button { width: 100%; }


    /* ================= MENU ARROWS ================= */
    .st-key-close_menu, .st-key-open_menu { display: flex; justify-content: flex-end; }
    .st-key-open_menu { position: fixed; top: 14px; left: 16px; z-index: 999999; width: auto; justify-content: flex-start; }
    .st-key-close_menu button, .st-key-open_menu button {
        width: 40px; min-width: 40px; height: 40px; min-height: 40px; padding: 0; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        background: #ffffff; border: 1px solid #c9d5e3; box-shadow: 0 2px 8px rgba(15, 42, 71, 0.10);
        transition: all 0.15s ease;
    }
    .st-key-close_menu button p, .st-key-open_menu button p {
        font-size: 26px; line-height: 1; font-weight: 600; color: #0f2a47; margin: 0; transform: translateY(-2px);
    }
    .st-key-close_menu button:hover, .st-key-open_menu button:hover { background: #0f2a47; border-color: #0f2a47; }
    .st-key-close_menu button:hover p, .st-key-open_menu button:hover p { color: #ffffff; }

    /* on desktop we use our own arrows, so remove Streamlit's white top bar and its arrow */
    @media (min-width: 641px) {
        header[data-testid="stHeader"], div[data-testid="stSidebarHeader"],
        button[data-testid="stSidebarCollapseButton"] { display: none !important; }
        .block-container { padding-top: 1.8rem; }

        /* if Streamlit's own sidebar state is "collapsed", force it open again.
           From now on only our arrow buttons open and close the menu. */
        section[data-testid="stSidebar"], section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: none !important; margin-left: 0 !important; left: 0 !important;
            width: 336px !important; min-width: 336px !important; visibility: visible !important;
        }
        section[data-testid="stSidebar"] > div { visibility: visible !important; opacity: 1 !important; width: 336px !important; }
    }


    /* close arrow on the dark sidebar */
    section[data-testid="stSidebar"] .st-key-close_menu button {
        background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.22); box-shadow: none;
    }
    section[data-testid="stSidebar"] .st-key-close_menu button p { color: #ffffff; }
    section[data-testid="stSidebar"] .st-key-close_menu button:hover { background: #ffffff; border-color: #ffffff; }
    section[data-testid="stSidebar"] .st-key-close_menu button:hover p { color: #0f2a47; }

    /* ================= FOOTER ================= */
    .footer {
        border-top: 1px solid #dde5ef; margin-top: 40px; padding-top: 18px;
        text-align: center; font-size: 13px; color: #64748b; line-height: 1.7;
    }
    .footer b { color: #1d4ed8; }

    @media (max-width: 1100px) {
        .grid-4 { grid-template-columns: repeat(2, 1fr); }
        .flow { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 900px) {
        .block-container { padding: 1.4rem 1rem; }
        .page-title { font-size: 25px; }
        .grid-3, .grid-4, .flow { grid-template-columns: 1fr; }
        .file-row { flex-direction: column; align-items: flex-start; gap: 8px; }
        .file-name { flex: 0 0 auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({"font.family": "sans-serif", "axes.facecolor": "white", "figure.facecolor": "white"})


# ---------------- Load model and data ----------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_data():
    if os.path.exists(DATASET_PATH):
        data = pd.read_csv(DATASET_PATH)
        data.columns = data.columns.str.strip()
        for col in data.select_dtypes(include="object").columns:
            data[col] = data[col].astype(str).str.strip()
        return data
    return None


model = load_model()
df = load_data()


# ---------------- Helper functions ----------------
def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card():
    """White bordered box (needs Streamlit 1.29+, falls back to plain container)."""
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()
    with box:
        yield


def encode_input(data):
    """Convert text columns to numbers, the same way the model was trained."""
    data = data.copy()
    data["education"] = data["education"].astype(str).str.strip().map({"Graduate": 1, "Not Graduate": 0})
    data["self_employed"] = data["self_employed"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    for col in data.columns:
        if col not in ["education", "self_employed"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def approval_index():
    """Position of the 'approved' class (1) in the model output."""
    classes = list(model.classes_)
    return classes.index(1) if 1 in classes else 1


def read_pdf(file):
    """Read values like 'income_annum: 500000' from a PDF."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        row = {}
        for name in FEATURES:
            if name == "education":
                value_pattern = r"(Not Graduate|Graduate)"
            elif name == "self_employed":
                value_pattern = r"(Yes|No)"
            else:
                value_pattern = r"([\d,]+)"
            match = re.search(name + r"\s*[:\-]?\s*" + value_pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                row[name] = int(value) if value.isdigit() else value.title()
        return pd.DataFrame([row]) if row else None
    except Exception:
        return None


def read_uploaded_file(file):
    """Read a CSV, Excel, JSON or PDF file. Returns (dataframe, error_message)."""
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            data = pd.read_csv(file)

        elif name.endswith((".xlsx", ".xls")):
            data = pd.read_excel(file)  # first sheet

        elif name.endswith(".json"):
            content = json.load(file)
            # allow {"data": [...]}, {"records": [...]} or {"applicants": [...]}
            if isinstance(content, dict):
                for key in ("data", "records", "applicants"):
                    if isinstance(content.get(key), list):
                        content = content[key]
                        break
            if isinstance(content, dict):
                # one applicant: {"income_annum": 500000, ...}
                one_row = all(not isinstance(v, (list, dict)) for v in content.values())
                data = pd.DataFrame([content]) if one_row else pd.DataFrame(content)
            else:
                data = pd.DataFrame(content)

        elif name.endswith(".pdf"):
            data = read_pdf(file)
            if data is None:
                return None, "The PDF does not contain readable applicant information."

        else:
            return None, "This file type is not supported."

    except ImportError as e:
        return None, f"A package needed to read this file is missing: {e}. For Excel files run: pip install openpyxl"
    except Exception as e:
        return None, f"The file could not be read: {e}"

    if data is None or data.empty:
        return None, "The file does not contain any records."

    data.columns = data.columns.astype(str).str.strip()
    return data, None


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d5e3")
    ax.spines["bottom"].set_color("#c9d5e3")
    ax.tick_params(labelsize=9, colors="#475569")
    ax.grid(axis="y", color="#eef2f7")
    ax.set_axisbelow(True)


def show(fig, ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", color=NAVY, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#475569")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#475569")
    style_axes(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def bar_chart(labels, values, title, colors, ylabel=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#334155")
    show(fig, ax, title, ylabel=ylabel)


def hist_chart(values, title, xlabel, bins=20):
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white")
    show(fig, ax, title, xlabel, "Applicants")


def html_block(css_class, items, template):
    """Build a grid of small HTML cards (no indentation, so markdown keeps it as HTML)."""
    cards = "".join(template.format(*item) for item in items)
    st.markdown(f'<div class="{css_class}">{cards}</div>', unsafe_allow_html=True)


# ---------------- Menu open / close state ----------------
if "menu_open" not in st.session_state:
    st.session_state.menu_open = True


def set_menu(value):
    st.session_state.menu_open = value


# ---------------- Sidebar ----------------
with st.sidebar:
    _, close_col = st.columns([4, 1])
    close_col.button("‹", key="close_menu", on_click=set_menu, args=(False,), use_container_width=True)

    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">LE</div>
            <div>
                <div class="brand-title">Loan Eligibility</div>
                <div class="brand-sub">Machine learning loan decision system</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")


# ---------------- Reopen arrow (shown only while the menu is closed) ----------------
if not st.session_state.menu_open:
    # the sidebar is still rendered (so the selected page is kept), it is only hidden from view
    st.markdown(
        "<style>section[data-testid='stSidebar'] { display: none !important; }"
        " .block-container { padding-top: 4rem !important; }</style>",
        unsafe_allow_html=True,
    )
    open_col, _ = st.columns([1, 14])
    open_col.button("›", key="open_menu", on_click=set_menu, args=(True,), use_container_width=True)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    page_header(
        "Loan Eligibility Prediction",
        "Assess loan applications using applicant, credit and asset information, with the probability behind every decision.",
    )

    if df is not None:
        approved = int((df["loan_status"].str.lower() == "approved").sum())
        rejected = int((df["loan_status"].str.lower() == "rejected").sum())

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        c1.metric("Total Applications", f"{len(df):,}")
        c2.metric("Approved", f"{approved:,}")
        c3.metric("Rejected", f"{rejected:,}")
        c4.metric("Approval Rate", f"{approved / len(df) * 100:.1f}%")

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        with card():
            st.subheader("How it works")
            st.markdown(
                """
                1. Enter the applicant details.
                2. Text values such as education are converted to numbers.
                3. The trained model reads all eleven inputs.
                4. The model predicts whether the loan is approved or rejected.
                5. The approval probability is shown along with supporting charts.
                """
            )

    with right:
        with card():
            st.subheader("Modules")
            st.markdown(
                """
                - **Dataset Preview:** view the records, column details and statistics.
                - **Manual Prediction:** assess a single loan application.
                - **Bulk Prediction:** upload a CSV or PDF and predict many applicants at once.
                - **Analytics:** review trends and relationships in the loan data.
                - **About:** project summary, workflow, technologies and files.
                """
            )

    with card():
        st.subheader("Model input features")
        table = pd.DataFrame({
            "Feature": [FEATURE_INFO[f][0] for f in FEATURES],
            "Description": [FEATURE_INFO[f][2] for f in FEATURES],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)


# ---------------- DATASET PREVIEW ----------------
elif page == "Dataset Preview":
    page_header("Dataset Preview", "The loan data used to train and analyse the model.")

    if df is None:
        st.error("loan_approval_dataset.csv was not found. Keep it in the same folder as this file.")
    else:
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.write("")
        with card():
            st.subheader("Records")
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)

        with card():
            st.subheader("Column information")
            column_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [str(x) for x in df.dtypes],
                "Missing Values": [int(df[x].isnull().sum()) for x in df.columns],
                "Unique Values": [int(df[x].nunique()) for x in df.columns],
            })
            st.dataframe(column_info, use_container_width=True, hide_index=True)

        with card():
            st.subheader("Statistical summary")
            st.dataframe(df.describe().T, use_container_width=True)


# ---------------- MANUAL PREDICTION ----------------
elif page == "Manual Prediction":
    page_header("Manual Prediction", "Enter the applicant details below and run the assessment to get a result.")

    if model is None:
        st.error("loan_model.pkl was not found. Keep it in the same folder as this file.")
    else:
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            with card():
                st.subheader("Personal information")
                dependents = st.number_input("Number of dependents", 0, 20, 2)
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self employed", ["No", "Yes"])

        with c2:
            with card():
                st.subheader("Loan information")
                income = st.number_input("Annual income", 0, value=5000000, step=100000)
                loan_amount = st.number_input("Loan amount", 0, value=15000000, step=100000)
                loan_term = st.number_input("Loan term", 1, 50, 10)

        with c3:
            with card():
                st.subheader("Credit and assets")
                cibil = st.number_input("CIBIL score", 0, 900, 700)
                residential = st.number_input("Residential assets", 0, value=5000000, step=100000)
                commercial = st.number_input("Commercial assets", 0, value=3000000, step=100000)

        with card():
            st.subheader("Other assets")
            a1, a2 = st.columns(2, gap="large")
            luxury = a1.number_input("Luxury assets", 0, value=5000000, step=100000)
            bank = a2.number_input("Bank assets", 0, value=3000000, step=100000)

        if st.button("Run Eligibility Assessment", type="primary", use_container_width=True):
            row = pd.DataFrame([{
                "no_of_dependents": dependents,
                "education": 1 if education == "Graduate" else 0,
                "self_employed": 1 if self_employed == "Yes" else 0,
                "income_annum": income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil,
                "residential_assets_value": residential,
                "commercial_assets_value": commercial,
                "luxury_assets_value": luxury,
                "bank_asset_value": bank,
            }])

            try:
                result = model.predict(row)[0]
                approval = model.predict_proba(row)[0][approval_index()] * 100
                rejection = 100 - approval

                with card():
                    st.subheader("Assessment result")

                    if result == 1:
                        st.markdown(
                            """
                            <div class="result-approved">
                                <div class="result-title">Loan Approved</div>
                                <div class="result-text">The model predicts that this applicant is eligible for the requested loan.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="result-rejected">
                                <div class="result-title">Loan Rejected</div>
                                <div class="result-text">The model predicts that this applicant is not eligible for the requested loan.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    m1, m2, m3 = st.columns(3, gap="medium")
                    m1.metric("Approval Probability", f"{approval:.1f}%")
                    m2.metric("Rejection Probability", f"{rejection:.1f}%")
                    m3.metric("Loan to Income Ratio", f"{loan_amount / income:.2f}" if income else "N/A")

                with card():
                    st.subheader("Assessment analysis")
                    g1, g2 = st.columns(2, gap="large")

                    with g1:
                        bar_chart(["Approval", "Rejection"], [approval, rejection],
                                  "Prediction probability", [GREEN, RED], "Probability (%)")

                    with g2:
                        labels = ["Income", "Loan", "Residential", "Commercial", "Luxury", "Bank"]
                        values = [income, loan_amount, residential, commercial, luxury, bank]
                        fig, ax = plt.subplots(figsize=(5.5, 3.3))
                        ax.barh(labels[::-1], [v / 100000 for v in values][::-1], color=BLUE, height=0.55)
                        show(fig, ax, "Applicant financial profile", xlabel="Value in 100,000 units")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ---------------- BULK PREDICTION ----------------
elif page == "Bulk Prediction":
    page_header("Bulk Prediction", "Upload a CSV, Excel or JSON file with many applicants, or a structured PDF with one applicant.")

    with card():
        st.subheader("Upload file")
        file = st.file_uploader(
            "Select a CSV, Excel, JSON or PDF file",
            type=["csv", "xlsx", "xls", "json", "pdf"],
        )

    if file is not None:
        data, error = read_uploaded_file(file)
        if error:
            st.error(error)

        if data is not None:
            with card():
                st.subheader("Uploaded records")
                st.dataframe(data, use_container_width=True, hide_index=True, height=250)

            missing = [c for c in FEATURES if c not in data.columns]

            if missing:
                st.error("Required columns are missing: " + ", ".join(missing))
            elif model is None:
                st.error("loan_model.pkl was not found.")
            else:
                model_input = encode_input(data[FEATURES])

                if model_input.isnull().any().any():
                    st.error("The file contains missing or invalid values.")
                else:
                    try:
                        predictions = model.predict(model_input)
                        approval_probs = model.predict_proba(model_input)[:, approval_index()] * 100

                        results = data.copy()
                        results["Prediction"] = ["Approved" if p == 1 else "Rejected" for p in predictions]
                        results["Approval Probability"] = approval_probs.round(2)

                        approved = int((predictions == 1).sum())
                        rejected = int((predictions == 0).sum())

                        c1, c2, c3 = st.columns(3, gap="medium")
                        c1.metric("Total Records", len(results))
                        c2.metric("Approved", approved)
                        c3.metric("Rejected", rejected)

                        with card():
                            st.subheader("Prediction results")
                            st.dataframe(results, use_container_width=True, hide_index=True, height=300)
                            st.download_button(
                                "Download Prediction Results",
                                results.to_csv(index=False),
                                file_name="loan_prediction_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                        with card():
                            st.subheader("Bulk analysis")
                            g1, g2 = st.columns(2, gap="large")
                            with g1:
                                bar_chart(["Approved", "Rejected"], [approved, rejected],
                                          "Prediction distribution", [GREEN, RED], "Applications")
                            with g2:
                                hist_chart(approval_probs, "Approval probability distribution",
                                           "Approval probability (%)", 10)

                    except Exception as e:
                        st.error(f"Bulk prediction error: {e}")


# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    page_header("Analytics", "Explore loan outcomes, applicant profiles and financial relationships.")

    if df is None:
        st.error("loan_approval_dataset.csv was not found.")
    else:
        approved = int((df["loan_status"].str.lower() == "approved").sum())
        rejected = int((df["loan_status"].str.lower() == "rejected").sum())

        with card():
            st.subheader("Loan performance")
            g1, g2 = st.columns(2, gap="large")
            with g1:
                bar_chart(["Approved", "Rejected"], [approved, rejected],
                          "Loan status distribution", [GREEN, RED], "Applications")
            with g2:
                hist_chart(df["cibil_score"], "CIBIL score distribution", "CIBIL score")

        with card():
            st.subheader("Applicant profile")
            g3, g4 = st.columns(2, gap="large")
            with g3:
                counts = df["education"].value_counts()
                bar_chart(counts.index.astype(str), counts.values, "Education distribution", BLUE, "Applicants")
            with g4:
                counts = df["self_employed"].value_counts()
                bar_chart(counts.index.astype(str), counts.values, "Employment distribution", TEAL, "Applicants")

        with card():
            st.subheader("Financial analysis")
            g5, g6 = st.columns(2, gap="large")
            with g5:
                fig, ax = plt.subplots(figsize=(5.5, 3.3))
                ax.scatter(df["income_annum"], df["loan_amount"], alpha=0.35, color=BLUE, s=12, edgecolors="none")
                show(fig, ax, "Annual income vs loan amount", "Annual income", "Loan amount")
            with g6:
                avg = df.groupby("loan_status")["loan_amount"].mean()
                colors = [GREEN if str(s).strip().lower() == "approved" else RED for s in avg.index]
                bar_chart(avg.index.astype(str), avg.values, "Average loan amount by status",
                          colors, "Average loan amount")


# ---------------- ABOUT ----------------
elif page == "About":
    page_header(
        "About the Project",
        "A machine learning dashboard that helps assess loan applications quickly and consistently, with the probability behind every decision.",
    )

    # ---- facts read from the real model, dataset and files ----
    this_file = os.path.basename(__file__)

    model_type, model_classes, model_inputs = "Not loaded", "N/A", "N/A"
    if model is not None:
        final_step = model.steps[-1][1] if hasattr(model, "steps") else model
        model_type = type(final_step).__name__
        classes = getattr(model, "classes_", None)
        if classes is not None:
            model_classes = ", ".join(str(c) for c in classes)
        n_in = getattr(model, "n_features_in_", None)
        model_inputs = str(n_in) if n_in is not None else str(len(FEATURES))

    if df is not None:
        status_col = df["loan_status"].str.lower() if "loan_status" in df.columns else pd.Series(dtype=str)
        data_rows = f"{len(df):,}"
        data_cols = str(len(df.columns))
        data_rate = f"{(status_col == 'approved').sum() / len(df) * 100:.1f}%" if len(df) else "N/A"
        data_missing = str(int(df.isnull().sum().sum()))
    else:
        data_rows = data_cols = data_rate = data_missing = "N/A"

    # quick facts
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Model Inputs", len(FEATURES))
    c2.metric("Dashboard Sections", len(PAGES))
    c3.metric("Upload Formats", "4 file types")
    c4.metric("Dataset Records", data_rows)

    st.write("")

    # project summary
    with card():
        st.subheader("Project summary")
        st.markdown(
            """
            <p class="about-intro">
            The <b>Loan Eligibility Prediction Dashboard</b> loads a pre-trained machine learning model
            (<b>loan_model.pkl</b>) and the historical loan dataset (<b>loan_approval_dataset.csv</b>) into a Streamlit
            web app. A user can enter the details of one applicant, or upload a CSV or PDF file, and the model predicts
            whether the loan is <b>approved or rejected</b> together with the probability of that outcome. The
            dashboard also lets the user preview the dataset and review its trends through charts.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # model and dataset details (read from the loaded objects)
    left, right = st.columns(2, gap="large")

    def kv_rows(rows):
        return "".join(
            f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
            for k, v in rows
        )

    with left:
        with card():
            st.subheader("Model details")
            st.markdown(
                kv_rows([
                    ("Status", "Loaded" if model is not None else "Not found"),
                    ("Model type", model_type),
                    ("Output classes", model_classes),
                    ("Inputs expected", model_inputs),
                ]),
                unsafe_allow_html=True,
            )

    with right:
        with card():
            st.subheader("Dataset details")
            st.markdown(
                kv_rows([
                    ("Status", "Loaded" if df is not None else "Not found"),
                    ("Records", data_rows),
                    ("Columns", data_cols),
                    ("Historical approval rate", data_rate),
                    ("Missing values", data_missing),
                ]),
                unsafe_allow_html=True,
            )

    # key features
    st.subheader("Key features")
    html_block(
        "grid-3",
        [
            ("Manual prediction", "Enter the 11 applicant details and get an approved or rejected result with approval and rejection probability."),
            ("Bulk prediction", "Upload a CSV, Excel or JSON file with many applicants, or a text based PDF with one applicant, and download the results as a CSV."),
            ("Input validation", "Missing columns and missing or invalid values are reported before the model is run."),
            ("Dataset preview", "View the records, column information (type, missing and unique values) and a statistical summary."),
            ("Analytics", "Charts for loan status, CIBIL score, education, employment, income against loan amount and average loan amount."),
            ("Overview", "Summary counts of applications, approvals, rejections and the historical approval rate."),
        ],
        '<div class="feat-card"><div class="feat-title">{0}</div><div class="feat-text">{1}</div></div>',
    )

    # workflow
    st.subheader("Workflow")
    html_block(
        "flow",
        [
            (1, "Collect", "Applicant details are typed in or read from an uploaded CSV, Excel, JSON or PDF."),
            (2, "Validate", "All 11 required columns must be present and values must be valid."),
            (3, "Encode", "Education and self employed text is converted to 1 and 0."),
            (4, "Predict", "The loaded model predicts approved or rejected."),
            (5, "Probability", "predict_proba gives the approval and rejection probability."),
            (6, "Present", "Result, charts and a results CSV download are shown."),
        ],
        '<div class="flow-step"><div class="flow-num">{0}</div><div class="flow-title">{1}</div><div class="flow-text">{2}</div></div>',
    )

    # technologies
    st.subheader("Technologies used")
    html_block(
        "grid-4",
        [
            ("Python", "Core programming language of the project"),
            ("Streamlit", "Builds the dashboard, pages and input forms"),
            ("Pandas", "Reads CSV, Excel and JSON files and prepares the model input"),
            ("Joblib", "Loads the saved model from loan_model.pkl"),
            ("Matplotlib", "Draws the probability and analytics charts"),
            ("PyPDF2", "Extracts text from uploaded PDF files"),
            ("Regex", "Finds the column values inside the PDF text"),
            ("CSS", "Custom styling of the sidebar, cards and pages"),
        ],
        '<div class="tech-card"><div class="tech-name">{0}</div><div class="tech-role">{1}</div></div>',
    )

    # model inputs
    with card():
        st.subheader("Model inputs")
        st.caption("These 11 columns are passed to the model in this order.")
        inputs = pd.DataFrame({
            "No.": list(range(1, len(FEATURES) + 1)),
            "Input": [FEATURE_INFO[f][0] for f in FEATURES],
            "Column Name": FEATURES,
            "Accepted Value": [FEATURE_INFO[f][1] for f in FEATURES],
            "Description": [FEATURE_INFO[f][2] for f in FEATURES],
        })
        st.dataframe(inputs, use_container_width=True, hide_index=True)

    # project files (existence checked on disk)
    def file_row(name, desc, path=None):
        if path is None:
            pill = '<span class="pill pill-grey">Generated</span>'
        elif os.path.exists(path):
            pill = '<span class="pill pill-ok">Found</span>'
        else:
            pill = '<span class="pill pill-bad">Missing</span>'
        return (
            f'<div class="file-row"><div class="file-name">{name}</div>'
            f'<div class="file-desc">{desc}</div>{pill}</div>'
        )

    with card():
        st.subheader("Project files")
        st.markdown(
            "<div>"
            + file_row(this_file, "Main Streamlit application with all pages, styling, validation and the prediction flow.", this_file)
            + file_row(MODEL_PATH, "Pre-trained model, loaded once with Joblib and used for every prediction.", MODEL_PATH)
            + file_row(DATASET_PATH, "Historical loan records used by Overview, Dataset Preview and Analytics.", DATASET_PATH)
            + file_row("loan_prediction_results.csv", "Downloaded from Bulk Prediction. It contains the uploaded data plus Prediction and Approval Probability.")
            + "</div>",
            unsafe_allow_html=True,
        )

    # upload format guide
    with card():
        st.subheader("Upload format guide")
        st.caption("Every format needs the same 11 column names listed in the Model inputs table.")
        g1, g2, g3 = st.columns(3, gap="large")
        with g1:
            st.markdown("**CSV and Excel files**")
            st.markdown(
                "- Accepted types: `.csv`, `.xlsx` and `.xls`.\n"
                "- The first row must contain the column names.\n"
                "- Excel files are read from the first sheet and need the `openpyxl` package.\n"
                "- Extra columns such as `loan_id` or `loan_status` are ignored.\n"
                "- Education must be exactly `Graduate` or `Not Graduate`, and self employed `Yes` or `No`."
            )
        with g2:
            st.markdown("**JSON file**")
            st.markdown(
                "- A list of applicant objects, one object per applicant.\n"
                "- A single object is read as one applicant.\n"
                "- The list can also be inside a `data`, `records` or `applicants` key.\n"
                "- Each key is a column name, for example `\"cibil_score\": 720`.\n"
                "- The same Education and self employed values apply."
            )
        with g3:
            st.markdown("**PDF file**")
            st.markdown(
                "- Must contain selectable text, not a scanned image.\n"
                "- Each value is written as `column_name: value`, for example `income_annum: 5000000`.\n"
                "- Numbers may contain commas.\n"
                "- Only the first match of each column is read, so one applicant per PDF.\n"
                "- All 11 columns must be found, otherwise the missing ones are listed."
            )

    # responsible use
    st.markdown(
        """
        <div class="note-box">
            <div class="note-title">Responsible use</div>
            <div class="note-text">
            The prediction is a decision support tool. It is based on patterns in historical data and should be
            reviewed together with a loan officer's judgement and the lender's own policies. It should not be used
            as the only basis for approving or rejecting a real loan application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Footer ----------------
st.markdown(
    """
    <div class="footer">
        Loan Eligibility Prediction System, Machine Learning and Data Analytics<br>
        Developed by <b>Yash Joshi</b>
    </div>
    """,
    unsafe_allow_html=True,
)