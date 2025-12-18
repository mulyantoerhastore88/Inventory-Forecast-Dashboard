# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Judul ---
st.markdown('<h1 class="main-header">📊 Inventory Intelligence Pro</h1>', unsafe_allow_html=True)
st.caption(f"Dashboard for Inventory Control & Demand Planning | Last Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- KONEKSI KE GOOGLE SHEETS ---
@st.cache_resource
def init_gsheet_connection():
    """Inisialisasi koneksi ke Google Sheets menggunakan kredensial dari secrets.toml"""
    try:
        # Ambil kredensial dari secrets.toml[citation:2][citation:9]
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"❌ Gagal menginisialisasi koneksi Google Sheets: {str(e)}")
        st.info("Pastikan Anda telah menambahkan kredensial yang benar ke secrets.toml dan telah membagikan Google Sheet ke service account.")
        return None

@st.cache_data(ttl=600)
def load_sheet_data(_client, sheet_url, sheet_name):
    """Memuat data dari worksheet spesifik"""
    try:
        sh = _client.open_by_url(sheet_url)
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa memuat sheet '{sheet_name}': {str(e)}")
        return pd.DataFrame()

# --- Inisialisasi Koneksi ---
gsheet_url = st.secrets["gsheet_url"]
client = init_gsheet_connection()

if client is None:
    st.stop()

# --- Muat Semua Data ---
with st.spinner('🔄 Memuat data dari Google Sheets...'):
    # Mapping antara nama sheet di GSheet dengan variabel yang kita gunakan
    df_product = load_sheet_data(client, gsheet_url, "Product_Master")
    df_rofo = load_sheet_data(client, gsheet_url, "Rofo")
    df_sales = load_sheet_data(client, gsheet_url, "Sales")
    df_po = load_sheet_data(client, gsheet_url, "PO")
    df_stock = load_sheet_data(client, gsheet_url, "Stock_Onhand")

# ======================================================
# DI SINI ANDA DAPAT MENERAPKAN FUNGSI ANALISIS
# DAN VISUALISASI YANG TELAH KITA BUAT SEBELUMNYA
# ======================================================

# --- Tampilkan Preview Data untuk Verifikasi ---
st.subheader("📁 Preview Data yang Dimuat")

tab_names = ["Product Master", "Sales", "Rofo (Forecast)", "PO", "Stock"]
dfs = [df_product, df_sales, df_rofo, df_po, df_stock]

tabs = st.tabs(tab_names)

for i, (tab, df, name) in enumerate(zip(tabs, dfs, tab_names)):
    with tab:
        if not df.empty:
            st.write(f"**Jumlah Baris:** {len(df)}")
            st.write(f"**Kolom:** {', '.join(df.columns.tolist())}")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info(f"Data untuk {name} kosong atau tidak dapat dimuat.")

# --- Footer ---
st.divider()
st.caption("Inventory Intelligence Pro | Data sumber: Google Sheets")

# PERHATIAN PENTING:
# Pastikan Anda telah membagikan Google Sheet ke alamat email service account:
# gsheet-forcast-to-dashboard@inventoryforecast-479502.iam.gserviceaccount.com
# Tanpa ini, aplikasi tidak akan bisa mengakses data Anda![citation:4][citation:7]
