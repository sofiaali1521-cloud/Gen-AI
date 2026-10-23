"""
app.py
Enso Group HR Policy Assistant — Luxury Enterprise Edition (Obsidian Black + Champagne Gold)
Features: Top-Right Theme Toggle, Slim Header, Strict High-Contrast Luxury Palettes,
Resumable ChromaDB Ingestion, Offline Explorer & QA.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import (
    DEFAULT_PDF_PATH,
    DEFAULT_PERSIST_DIR,
    build_vectorstore,
    check_index_status,
    load_and_split_pdf,
    get_clean_key,
)
from rag import answer_question

# 1. Page Configuration
st.set_page_config(
    page_title="Enso Group — HR Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Session State Initialization — Default to Obsidian Black + Champagne Gold
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = get_clean_key()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# 3. Luxury Color Tokens & Dynamic Theme Engine
def get_theme_css(theme: str) -> str:
    if theme == "dark":
        return """
        <style>
        /* === OBSIDIAN BLACK + CHAMPAGNE GOLD LUXURY DESIGN SYSTEM === */
        /* Background: #090909 | Cards: #151515 | Inputs: #1B1B1B | Borders: #303030 */
        /* Primary Text: #F5F1E8 | Secondary Text: #B8B1A4 | Accent Gold: #D6B878 | Hover: #E6CB91 */
        
        .stApp {
            background-color: #090909 !important;
            color: #F5F1E8 !important;
        }

        /* Top Bar & Container Layout */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* Slim Unified Top Navigation Bar (Dark Theme) */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stToggle"]) {
            background-color: #151515 !important;
            border: 1px solid #303030 !important;
            border-radius: 10px !important;
            padding: 12px 20px !important;
            margin-bottom: 20px !important;
            align-items: center !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        }

        .hr-top-navbar {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .hr-top-title {
            font-size: 20px !important;
            font-weight: 700 !important;
            color: #F5F1E8 !important;
            margin: 0 0 2px 0 !important;
            letter-spacing: -0.2px !important;
            line-height: 1.25 !important;
        }
        .hr-top-subtitle {
            font-size: 12.5px !important;
            color: #B8B1A4 !important;
            margin: 0 !important;
            line-height: 1.3 !important;
        }

        /* Top-Right Toggle Switch (Dark Theme) */
        div[data-testid="stToggle"] {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            margin: 0 !important;
        }
        div[data-testid="stToggle"] label {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            margin: 0 !important;
            cursor: pointer !important;
        }
        div[data-testid="stToggle"] label span {
            font-size: 13px !important;
            font-weight: 600 !important;
            color: #F5F1E8 !important;
            letter-spacing: 0.2px !important;
        }
        div[data-testid="stToggle"] div[role="switch"] {
            transform: scale(0.88) !important;
            transition: background-color 0.2s ease !important;
        }
        div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
            background-color: #D6B878 !important;
        }
        div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
            background-color: #303030 !important;
        }

        /* Sidebar Styling (Dark Theme: Sidebar #0D0D0D, Borders #303030) */
        section[data-testid="stSidebar"] {
            background-color: #0D0D0D !important;
            border-right: 1px solid #303030 !important;
            padding-top: 1rem !important;
        }
        section[data-testid="stSidebar"] div.stVerticalBlock {
            gap: 0.5rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 8px 0 !important;
            border-color: #303030 !important;
        }
        .sidebar-brand-card {
            background: #151515;
            border: 1px solid #303030;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        .sidebar-brand-title {
            color: #F5F1E8;
            font-weight: 700;
            font-size: 13.5px;
            letter-spacing: 0.6px;
        }
        .sidebar-brand-sub {
            color: #B8B1A4;
            font-size: 11px;
            margin-top: 2px;
        }
        .sidebar-sec-label {
            color: #D6B878;
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 10px;
            margin-bottom: 4px;
        }
        .status-card-ok {
            background: #121912;
            border: 1px solid #234023;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #D1E7DD;
        }
        .status-card-warn {
            background: #1E1A14;
            border: 1px solid #483A20;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #EBD8B8;
        }
        .status-card-err {
            background: #221414;
            border: 1px solid #4E2424;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #F1B5B5;
        }
        .status-sub {
            color: #B8B1A4;
            font-size: 11px;
        }
        .status-dot-ok { color: #5CB85C; }
        .status-dot-warn { color: #D6B878; }
        .status-dot-err { color: #D9534F; }
        .key-active-text { color: #5CB85C; font-size: 11.5px; margin-top: 3px; font-weight: 500; }
        .key-inactive-text { color: #D6B878; font-size: 11.5px; margin-top: 3px; font-weight: 500; }

        section[data-testid="stSidebar"] input {
            background-color: #1B1B1B !important;
            color: #F5F1E8 !important;
            border: 1.5px solid #303030 !important;
            border-radius: 6px !important;
            font-size: 12.5px !important;
        }
        section[data-testid="stSidebar"] input::placeholder {
            color: #6A655C !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: #151515 !important;
            border: 1.5px solid #D6B878 !important;
            color: #F5F1E8 !important;
            text-align: center !important;
            min-height: 36px !important;
            font-size: 12.5px !important;
        }
        section[data-testid="stSidebar"] div.stButton > button * {
            color: #F5F1E8 !important;
            text-align: center !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #D6B878 !important;
            border-color: #E6CB91 !important;
            color: #090909 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover * {
            color: #090909 !important;
        }

        /* Section Titles in Main Page */
        .quick-title {
            font-size: 11.5px;
            font-weight: 700;
            color: #D6B878 !important;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        /* Evaluation Question Buttons (Dark Theme) */
        div.stButton > button {
            background-color: #151515 !important;
            color: #F5F1E8 !important;
            border: 1.5px solid #303030 !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
            transition: all 0.2s ease-in-out !important;
            text-align: left !important;
            width: 100% !important;
            white-space: normal !important;
            height: auto !important;
            min-height: 44px !important;
        }
        div.stButton > button * {
            color: #F5F1E8 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }
        div.stButton > button:hover {
            background-color: #D6B878 !important;
            border-color: #E6CB91 !important;
            color: #090909 !important;
            box-shadow: 0 4px 14px rgba(214, 184, 120, 0.25) !important;
        }
        div.stButton > button:hover * {
            color: #090909 !important;
        }

        /* Tabs Styling (Dark Theme) */
        div[data-testid="stTabs"] button[role="tab"] {
            color: #B8B1A4 !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            border-bottom: 2.5px solid transparent !important;
            padding: 8px 16px !important;
        }
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] button[role="tab"] span {
            color: #B8B1A4 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #F5F1E8 !important;
            font-weight: 700 !important;
            border-bottom: 2.5px solid #D6B878 !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {
            color: #F5F1E8 !important;
            font-weight: 700 !important;
        }

        /* Chat Messages & Input (Dark Theme) */
        div[data-testid="stChatMessage"] {
            background-color: #151515 !important;
            border: 1px solid #303030 !important;
            color: #F5F1E8 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }
        div[data-testid="stChatMessage"] * {
            color: #F5F1E8 !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: #1B1B1B !important;
            color: #F5F1E8 !important;
            border: 1.5px solid #303030 !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #6A655C !important;
        }

        /* Citation Badge & Not Found (Dark Theme) */
        .citation-badge {
            display: inline-block;
            background-color: #1E1A14;
            color: #D6B878;
            font-size: 12px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 9999px;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #483A20;
        }
        .not-found-badge {
            font-size: 14px;
            font-weight: 700;
            color: #F1B5B5;
            background-color: #221414;
            padding: 8px 14px;
            border-radius: 6px;
            border-left: 4px solid #D9534F;
            display: inline-block;
        }

        /* Expanders & Popovers (Dark Theme) */
        div[data-testid="stExpander"] {
            background-color: #151515 !important;
            border: 1px solid #303030 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: #D6B878 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] * {
            color: #F5F1E8 !important;
        }
        </style>
        """
    else:
        return """
        <style>
        /* === LIGHT LUXURY DESIGN SYSTEM === */
        /* Background: #F7F4EC | Sidebar: #EFE9DC | Cards: #FFFFFF | Inputs: #FFFFFF */
        /* Primary Text: #242016 | Secondary Text: #625B4E | Borders: #DED5C3 | Gold Accent: #967641 */

        .stApp {
            background-color: #F7F4EC !important;
            color: #242016 !important;
        }

        /* Top Bar & Container Layout */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* Slim Unified Top Navigation Bar (Light Theme) */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stToggle"]) {
            background-color: #FFFFFF !important;
            border: 1px solid #DED5C3 !important;
            border-radius: 10px !important;
            padding: 12px 20px !important;
            margin-bottom: 20px !important;
            align-items: center !important;
            box-shadow: 0 2px 8px rgba(36, 32, 22, 0.04) !important;
        }

        .hr-top-navbar {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .hr-top-title {
            font-size: 20px !important;
            font-weight: 700 !important;
            color: #242016 !important;
            margin: 0 0 2px 0 !important;
            letter-spacing: -0.2px !important;
            line-height: 1.25 !important;
        }
        .hr-top-subtitle {
            font-size: 12.5px !important;
            color: #625B4E !important;
            margin: 0 !important;
            line-height: 1.3 !important;
        }

        /* Top-Right Toggle Switch (Light Theme) */
        div[data-testid="stToggle"] {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            margin: 0 !important;
        }
        div[data-testid="stToggle"] label {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            margin: 0 !important;
            cursor: pointer !important;
        }
        div[data-testid="stToggle"] label span {
            font-size: 13px !important;
            font-weight: 600 !important;
            color: #242016 !important;
            letter-spacing: 0.2px !important;
        }
        div[data-testid="stToggle"] div[role="switch"] {
            transform: scale(0.88) !important;
            transition: background-color 0.2s ease !important;
        }
        div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
            background-color: #967641 !important;
        }
        div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
            background-color: #DED5C3 !important;
        }

        /* Sidebar Styling (Light Theme: Sidebar #EFE9DC, Borders #DED5C3) */
        section[data-testid="stSidebar"] {
            background-color: #EFE9DC !important;
            border-right: 1px solid #DED5C3 !important;
            padding-top: 1rem !important;
        }
        section[data-testid="stSidebar"] div.stVerticalBlock {
            gap: 0.5rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 8px 0 !important;
            border-color: #DED5C3 !important;
        }
        .sidebar-brand-card {
            background: #FFFFFF;
            border: 1px solid #DED5C3;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(36, 32, 22, 0.04);
        }
        .sidebar-brand-title {
            color: #242016;
            font-weight: 700;
            font-size: 13.5px;
            letter-spacing: 0.6px;
        }
        .sidebar-brand-sub {
            color: #625B4E;
            font-size: 11px;
            margin-top: 2px;
        }
        .sidebar-sec-label {
            color: #967641;
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 10px;
            margin-bottom: 4px;
        }
        .status-card-ok {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #166534;
        }
        .status-card-warn {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #92400E;
        }
        .status-card-err {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 12px;
            color: #991B1B;
        }
        .status-sub {
            color: #625B4E;
            font-size: 11px;
        }
        .status-dot-ok { color: #059669; }
        .status-dot-warn { color: #D97706; }
        .status-dot-err { color: #DC2626; }
        .key-active-text { color: #059669; font-size: 11.5px; margin-top: 3px; font-weight: 500; }
        .key-inactive-text { color: #D97706; font-size: 11.5px; margin-top: 3px; font-weight: 500; }

        section[data-testid="stSidebar"] input {
            background-color: #FFFFFF !important;
            color: #242016 !important;
            border: 1.5px solid #DED5C3 !important;
            border-radius: 6px !important;
            font-size: 12.5px !important;
        }
        section[data-testid="stSidebar"] input::placeholder {
            color: #9E9585 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: #FFFFFF !important;
            border: 1.5px solid #967641 !important;
            color: #242016 !important;
            text-align: center !important;
            min-height: 36px !important;
            font-size: 12.5px !important;
        }
        section[data-testid="stSidebar"] div.stButton > button * {
            color: #242016 !important;
            text-align: center !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #967641 !important;
            border-color: #967641 !important;
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover * {
            color: #FFFFFF !important;
        }

        /* Section Titles in Main Page */
        .quick-title {
            font-size: 11.5px;
            font-weight: 700;
            color: #242016 !important;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        /* Evaluation Question Buttons (Light Theme) */
        div.stButton > button {
            background-color: #FFFFFF !important;
            color: #242016 !important;
            border: 1.5px solid #DED5C3 !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 1px 3px rgba(36, 32, 22, 0.05) !important;
            transition: all 0.2s ease-in-out !important;
            text-align: left !important;
            width: 100% !important;
            white-space: normal !important;
            height: auto !important;
            min-height: 44px !important;
        }
        div.stButton > button * {
            color: #242016 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }
        div.stButton > button:hover {
            background-color: #242016 !important;
            border-color: #242016 !important;
            color: #F7F4EC !important;
            box-shadow: 0 3px 8px rgba(36, 32, 22, 0.15) !important;
        }
        div.stButton > button:hover * {
            color: #F7F4EC !important;
        }

        /* Tabs Styling (Light Theme) */
        div[data-testid="stTabs"] button[role="tab"] {
            color: #625B4E !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            border-bottom: 2.5px solid transparent !important;
            padding: 8px 16px !important;
        }
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] button[role="tab"] span {
            color: #625B4E !important;
            font-weight: 600 !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #242016 !important;
            font-weight: 700 !important;
            border-bottom: 2.5px solid #967641 !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {
            color: #242016 !important;
            font-weight: 700 !important;
        }

        /* Chat Messages & Input (Light Theme) */
        div[data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DED5C3 !important;
            color: #242016 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }
        div[data-testid="stChatMessage"] * {
            color: #242016 !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important;
            color: #242016 !important;
            border: 1.5px solid #DED5C3 !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9E9585 !important;
        }

        /* Citation Badge & Not Found (Light Theme) */
        .citation-badge {
            display: inline-block;
            background-color: #EFE9DC;
            color: #967641;
            font-size: 12px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 9999px;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #DED5C3;
        }
        .not-found-badge {
            font-size: 14px;
            font-weight: 700;
            color: #991B1B;
            background-color: #FEF2F2;
            padding: 8px 14px;
            border-radius: 6px;
            border-left: 4px solid #DC2626;
            display: inline-block;
        }

        /* Expanders & Popovers (Light Theme) */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DED5C3 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: #967641 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] * {
            color: #242016 !important;
        }
        </style>
        """

# Render Theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# 4. Cached Policy Document Loader for Explorer
@st.cache_data(show_spinner=False)
def get_cached_policy_chunks():
    """Loads and caches PDF chunks for instant offline manual exploration."""
    if os.path.isfile(DEFAULT_PDF_PATH):
        try:
            return load_and_split_pdf(DEFAULT_PDF_PATH)
        except Exception:
            return []
    return []


# 5. Clean, Compact Sidebar Layout
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-title">🏢 ENSO GROUP</div>
            <div class="sidebar-brand-sub">HR Policy Intelligence Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Status (Compact)
    st.markdown('<div class="sidebar-sec-label">📄 Document Status</div>', unsafe_allow_html=True)
    pdf_exists = os.path.isfile(DEFAULT_PDF_PATH)
    if pdf_exists:
        file_mb = os.path.getsize(DEFAULT_PDF_PATH) / (1024 * 1024)
        st.markdown(
            f"""
            <div class="status-card-ok">
                <span class="status-dot-ok">●</span> <b>Authoritative Manual Loaded</b><br>
                <span class="status-sub"><code>{DEFAULT_PDF_PATH}</code> ({file_mb:.1f} MB, 79 pp.)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-card-err">
                <span class="status-dot-err">●</span> <b>Missing Manual</b><br>
                <span class="status-sub"><code>{DEFAULT_PDF_PATH}</code> not found</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Index Status (Compact)
    chunk_count = check_index_status(DEFAULT_PERSIST_DIR)
    st.markdown('<div class="sidebar-sec-label">🗄️ Vector Index Status</div>', unsafe_allow_html=True)
    if chunk_count >= 279:
        st.markdown(
            f"""
            <div class="status-card-ok">
                <span class="status-dot-ok">●</span> <b>ChromaDB Ready</b><br>
                <span class="status-sub">All {chunk_count}/279 policy chunks indexed</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif chunk_count > 0:
        st.markdown(
            f"""
            <div class="status-card-warn">
                <span class="status-dot-warn">●</span> <b>Partially Indexed</b><br>
                <span class="status-sub">{chunk_count}/279 chunks persisted (resumable)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card-warn">
                <span class="status-dot-warn">●</span> <b>Index Empty</b><br>
                <span class="status-sub">Enter API key & index below</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # API Key Configuration
    st.markdown('<div class="sidebar-sec-label">🔑 Model Credentials</div>', unsafe_allow_html=True)
    current_key = st.session_state.gemini_api_key or get_clean_key()
    key_input = st.text_input(
        "Google Gemini API Key",
        value=current_key,
        type="password",
        placeholder="Paste AIzaSy... here",
        help="Obtain your API key at https://aistudio.google.com/",
        label_visibility="collapsed",
        key="gemini_api_key_input",
    )
    clean_val = get_clean_key(key_input)
    if clean_val != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = clean_val
        if clean_val:
            os.environ["GEMINI_API_KEY"] = clean_val
            os.environ["GOOGLE_API_KEY"] = clean_val

    active_key = st.session_state.gemini_api_key or get_clean_key()
    if active_key:
        masked_hint = f"({active_key[:4]}...{active_key[-2:]})" if len(active_key) > 8 else ""
        st.markdown(f'<div class="key-active-text">🟢 API Key is configured {masked_hint}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-inactive-text">⚠️ API Key required. Paste above or save in .env</div>', unsafe_allow_html=True)

    # Indexing Action with Safe Resumption & Verification
    st.markdown('<div class="sidebar-sec-label">⚡ Index Actions</div>', unsafe_allow_html=True)
    if chunk_count == 0:
        index_btn_label = "⚡ Index HR Manual"
        is_force = False
    elif chunk_count < 279:
        index_btn_label = f"⚡ Resume Indexing ({chunk_count}/279 Done)"
        is_force = False
    else:
        index_btn_label = "🔄 Re-Index Manual (Fresh Index)"
        is_force = True

    if st.button(index_btn_label, use_container_width=True, key="index_manual_btn"):
        if not active_key:
            st.error("❌ Action Required: Please enter your Google Gemini API Key above before indexing.")
        elif not pdf_exists:
            st.error(f"❌ Missing manual: `{DEFAULT_PDF_PATH}` not found on disk.")
        else:
            init_frac = chunk_count / 279.0 if 279 > 0 else 0.0
            progress_bar = st.progress(init_frac, text=f"Progress ({chunk_count}/279 chunks)...")
            status_text = st.empty()

            def on_progress(current, total, msg):
                frac = current / total if total > 0 else 0.0
                progress_bar.progress(min(frac, 1.0), text=f"Indexing ({current}/{total} chunks)...")
                status_text.caption(f"Status: {msg}")

            try:
                with st.spinner("Embedding chunks with Gemini rate-limiting & backoff guard..."):
                    vectorstore, indexed_count = build_vectorstore(
                        pdf_path=DEFAULT_PDF_PATH,
                        persist_dir=DEFAULT_PERSIST_DIR,
                        api_key=active_key,
                        force_reindex=is_force,
                        progress_callback=on_progress,
                    )

                # Verify that ChromaDB actually contains documents
                verified_count = check_index_status(DEFAULT_PERSIST_DIR)
                if verified_count >= 279:
                    progress_bar.progress(1.0, text="Indexing complete!")
                    st.success(f"✅ Verified: Successfully indexed all {verified_count}/279 chunks in ChromaDB!")
                    st.rerun()
                elif verified_count > 0:
                    progress_bar.progress(verified_count / 279.0, text=f"Saved {verified_count}/279 chunks.")
                    st.info(f"💾 Progress saved: {verified_count}/279 chunks persisted in ChromaDB. You can resume anytime.")
                    st.rerun()
                else:
                    st.error("❌ Verification failed: ChromaDB returned 0 chunks.")
            except Exception as e:
                err_msg = str(e)
                if active_key and active_key in err_msg:
                    err_msg = err_msg.replace(active_key, "[REDACTED_API_KEY]")
                if "API_KEY_INVALID" in err_msg or "api key not valid" in err_msg.lower():
                    st.error(
                        "❌ Invalid API Key: Google AI Studio rejected this key with `API_KEY_INVALID`. "
                        "Please verify your key at https://aistudio.google.com/ and replace it in the field above."
                    )
                else:
                    st.error(f"❌ Notice: {err_msg}")

    # Conversation Session Controls
    st.markdown('<div class="sidebar-sec-label">💬 Conversation</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
    with col_s2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()


# 6. Slim Top Navigation Bar with Integrated Top-Right Theme Toggle
header_col1, header_col2 = st.columns([4.6, 1.1], vertical_alignment="center")

with header_col1:
    st.markdown(
        """
        <div class="hr-top-navbar">
            <h1 class="hr-top-title">Enso Group — HR Policy Assistant</h1>
            <p class="hr-top-subtitle">Authoritative policy guidance grounded strictly in the 79-page HR Policy Manual</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    # Compact, elegant top-right toggle switch
    is_dark = st.toggle(
        "🌙 Dark" if st.session_state.theme == "dark" else "☀️ Light",
        value=(st.session_state.theme == "dark"),
        help="Switch theme",
        key="top_nav_theme_switch",
    )
    target_theme = "dark" if is_dark else "light"
    if target_theme != st.session_state.theme:
        st.session_state.theme = target_theme
        st.rerun()


# 7. Navigation Tabs: Assistant Chat vs. Manual Explorer
tab_assistant, tab_explorer = st.tabs(["💬 Policy Assistant & QA", "📖 HR Manual Explorer & Search"])


# ==============================================================================
# TAB 1: POLICY ASSISTANT & QA
# ==============================================================================
with tab_assistant:
    # Assigned Evaluation Questions Section with High-Contrast Layout
    st.markdown('<div class="quick-title">⚡ Quick Mid-Term Evaluation Questions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Q1: Privilege Leave (PL) days per year?", use_container_width=True, key="btn_q1"):
            st.session_state.pending_query = "How many days of Privilege Leave (PL) does a permanent employee get per year?"
        if st.button("Q2: Probation period up to Senior Manager?", use_container_width=True, key="btn_q2"):
            st.session_state.pending_query = "What is the probation period for employees up to Senior Manager?"
        if st.button("Q3: Health-insurance premium amount?", use_container_width=True, key="btn_q3"):
            st.session_state.pending_query = "What is the company's health-insurance premium amount?"

    with col2:
        if st.button("Q4: Current GST rate in India? (Test Boundary)", use_container_width=True, key="btn_q4"):
            st.session_state.pending_query = "What is the current GST rate in India?"
        if st.button("Q5: Maternity leave days given?", use_container_width=True, key="btn_q5"):
            st.session_state.pending_query = "How many days of maternity leave are given?"

    st.write("")

    # Query Handler
    def process_query(query_text: str):
        st.session_state.messages.append({"role": "user", "content": query_text})

        effective_key = st.session_state.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not effective_key:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ **Gemini API Key missing.** Please enter your Gemini API Key in the left sidebar to enable policy retrieval.",
                "sources": [],
                "is_found": False,
            })
            return

        with st.spinner("Consulting Enso Group HR Policy Manual..."):
            result = answer_question(
                question=query_text,
                api_key=effective_key,
                persist_dir=DEFAULT_PERSIST_DIR,
                top_k=4,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "is_found": result.get("is_found", False),
            "feedback": None,
        })

    # Trigger Pending Question from Button Click
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query(q)
        st.rerun()

    # Display Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            ans_text = msg["content"].strip()

            # Strict Policy Rule: If answer is 'Not found', display ONLY that answer without citations
            if ans_text == "Not found":
                st.markdown('<div class="not-found-badge">Not found</div>', unsafe_allow_html=True)
            else:
                st.markdown(ans_text)

                # Citations and Evidence Display
                sources = msg.get("sources", [])
                if sources:
                    pages = sorted(list(set(s["page"] for s in sources if "page" in s)))
                    badges = "".join([f'<span class="citation-badge">📄 Page {p}</span>' for p in pages])
                    st.markdown(f"<div style='margin-top: 10px; margin-bottom: 8px;'>{badges}</div>", unsafe_allow_html=True)

                    with st.expander(f"🔍 Verified Document Evidence ({len(sources)} Excerpts)"):
                        for e_idx, s in enumerate(sources):
                            st.markdown(f"**[Excerpt {e_idx + 1}] — Page {s.get('page', 'Unknown')}** (`{s.get('source', 'data/hr_manual.pdf')}`)")
                            st.text(s.get("content", "").strip())
                            if e_idx < len(sources) - 1:
                                st.divider()

            # Interactive Controls for Assistant Messages (Copy & Feedback)
            if msg["role"] == "assistant" and ans_text != "Not found" and not ans_text.startswith("⚠️"):
                col_c1, col_c2, col_c3, col_c4 = st.columns([1.5, 1, 1, 3])
                with col_c1:
                    with st.popover("📋 Copy Text", use_container_width=True):
                        st.text_area("Answer Text", value=ans_text, height=120)
                with col_c2:
                    if st.button("👍 Helpful", key=f"helpful_{idx}", use_container_width=True):
                        msg["feedback"] = "helpful"
                        st.toast("Thank you! Feedback recorded: Helpful.")
                        st.rerun()
                with col_c3:
                    if st.button("👎 Flag", key=f"unhelpful_{idx}", use_container_width=True):
                        msg["feedback"] = "unhelpful"
                        st.toast("Feedback noted: Flagged for HR review.")
                        st.rerun()
                with col_c4:
                    if msg.get("feedback") == "helpful":
                        st.caption("✅ Marked as helpful")
                    elif msg.get("feedback") == "unhelpful":
                        st.caption("⚠️ Flagged for HR policy review")

    # Interactive Chat Input Box
    user_query = st.chat_input("Ask any question regarding Enso Group HR policies...")
    if user_query:
        process_query(user_query)
        st.rerun()


# ==============================================================================
# TAB 2: HR MANUAL EXPLORER & SEARCH
# ==============================================================================
with tab_explorer:
    st.markdown("#### 📖 Browse & Search Manual Excerpts (79 Pages)")
    st.caption("Search through genuine chunks extracted directly from `data/hr_manual.pdf` with exact page numbers.")

    all_chunks = get_cached_policy_chunks()

    if not all_chunks:
        st.warning("⚠️ Could not load `data/hr_manual.pdf`. Ensure the file exists in the `data/` folder.")
    else:
        # Search & Filter Controls
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_query = st.text_input("🔍 Keyword Search", placeholder="e.g. leave, probation, insurance, notice, travel, maternity", key="explorer_search")
        with f_col2:
            page_options = ["All Pages (1-79)"] + [f"Page {i}" for i in range(1, 80)]
            selected_page = st.selectbox("📄 Filter by Page", page_options, key="explorer_page")
        with f_col3:
            section_presets = [
                "All Sections",
                "Table of Contents (pp. 2-4)",
                "Recruitment & Induction (pp. 5-11)",
                "Leave Policy (pp. 12-25)",
                "Probation & Confirmation (pp. 26-38)",
                "Employee Benefits (pp. 39-50)",
                "Travel & Claims (pp. 67-78)",
            ]
            selected_section = st.selectbox("📑 Quick Section Jump", section_presets, key="explorer_section")

        # Apply Filters
        filtered_chunks = all_chunks

        # Section preset filter
        if selected_section == "Table of Contents (pp. 2-4)":
            filtered_chunks = [c for c in filtered_chunks if 2 <= c.metadata.get("page", 0) <= 4]
        elif selected_section == "Recruitment & Induction (pp. 5-11)":
            filtered_chunks = [c for c in filtered_chunks if 5 <= c.metadata.get("page", 0) <= 11]
        elif selected_section == "Leave Policy (pp. 12-25)":
            filtered_chunks = [c for c in filtered_chunks if 12 <= c.metadata.get("page", 0) <= 25]
        elif selected_section == "Probation & Confirmation (pp. 26-38)":
            filtered_chunks = [c for c in filtered_chunks if 26 <= c.metadata.get("page", 0) <= 38]
        elif selected_section == "Employee Benefits (pp. 39-50)":
            filtered_chunks = [c for c in filtered_chunks if 39 <= c.metadata.get("page", 0) <= 50]
        elif selected_section == "Travel & Claims (pp. 67-78)":
            filtered_chunks = [c for c in filtered_chunks if 67 <= c.metadata.get("page", 0) <= 78]

        # Specific page filter
        if selected_page != "All Pages (1-79)":
            target_p = int(selected_page.replace("Page ", ""))
            filtered_chunks = [c for c in filtered_chunks if c.metadata.get("page", 0) == target_p]

        # Keyword filter
        if search_query.strip():
            sq = search_query.strip().lower()
            filtered_chunks = [c for c in filtered_chunks if sq in c.page_content.lower()]

        st.markdown(f"**Found `{len(filtered_chunks)}` matching passages** (out of {len(all_chunks)} total chunks):")

        # Render Matching Passages
        display_limit = 20
        for i, chunk in enumerate(filtered_chunks[:display_limit]):
            p_num = chunk.metadata.get("page", "Unknown")
            with st.expander(f"📄 Passage {i + 1} — Page {p_num} ({len(chunk.page_content)} characters)"):
                st.text(chunk.page_content.strip())

        if len(filtered_chunks) > display_limit:
            st.caption(f"Showing first {display_limit} of {len(filtered_chunks)} matching results. Refine your keyword or page filter above to view more specific passages.")
