import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(
    page_title="Inventory Intelligence Pro AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .main-header {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 1.5rem;
    }
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00f2fe;
        border-radius: 10px; padding: 15px;
        margin: 10px 0; border: 1px solid rgba(255,255,255,0.1);
    }
    .status-indicator {
        border-radius: 12px; padding: 1.2rem;
        font-weight: 800; text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        color: white;
    }
    .status-under { background: linear-gradient(135deg, #FF5252, #B71C1C); }
    .status-accurate { background: linear-gradient(135deg, #4CAF50, #1B5E20); }
    .status-over { background: linear-gradient(135deg, #FF9800, #E65100); }
</style>
""", unsafe_allow_html=True)

# --- KONEKSI ---
@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {e}"); return None

def parse_month_label(label):
    try:
        label_str = str(label).strip().upper()
        month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
        for m_name, m_num in month_map.items():
            if m_name in label_str:
                year_part = label_str.replace(m_name, '').replace('-', '').replace(' ', '').strip()
                year = int('20' + year_part) if len(year_part) == 2 else int(year_part) if year_part else datetime.now().year
                return datetime(year, m_num, 1)
        return datetime.now()
    except: return datetime.now()

# --- LOAD DATA ---
@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(_client):
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        ws_p = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws_p.get_all_records())
        df_product.columns = [c.strip().replace(' ', '_') for c in df_product.columns]
        
        def process_sheet(sheet_name, val_name):
            ws = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            m_cols = [c for c in df.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            df_long = df.melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='Month_Label', value_name=val_name)
            df_long[val_name] = pd.to_numeric(df_long[val_name], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            return df_long

        data['sales'] = process_sheet("Sales", "Sales_Qty")
        data['forecast'] = process_sheet("Rofo", "Forecast_Qty")
        data['po'] = process_sheet("PO", "PO_Qty")
        
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock = pd.DataFrame(ws_s.get_all_records())
        df_stock.columns = [c.strip().replace(' ', '_') for c in df_stock.columns]
        s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_stock.columns), 'Stock_Qty')
        data['stock'] = df_stock[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
        data['product'] = df_product
        return data
    except Exception as e:
        st.error(f"Error: {e}"); return {}

# --- ANALYTICS ENGINE (FIXED MERGE) ---
def calculate_metrics(data):
    df_f, df_po, df_p = data['forecast'], data['po'], data['product']
    
    # 1. Forecast Accuracy Logic (Last 3 Months)
    all_months = sorted(df_f['Month'].unique())
    last_3m = all_months[-3:] if len(all_months) >= 3 else all_months
    
    # Merge Forecast & PO
    merged = pd.merge(df_f[df_f['Month'].isin(last_3m)], 
                      df_po[df_po['Month'].isin(last_3m)], 
                      on=['SKU_ID', 'Month'], how='inner')
    
    # KUNCI PERBAIKAN: Tarik SKU_Tier dari Product Master ke hasil merge
    merged = pd.merge(merged, df_p[['SKU_ID', 'SKU_Tier', 'Product_Name', 'Brand']], on='SKU_ID', how='left')
    
    merged['Ratio'] = np.where(merged['Forecast_Qty'] > 0, (merged['PO_Qty']/merged['Forecast_Qty'])*100, 0)
    merged['Accuracy_Status'] = np.select(
        [merged['Ratio'] < 80, (merged['Ratio'] >= 80) & (merged['Ratio'] <= 120), merged['Ratio'] > 120], 
        ['Under', 'Accurate', 'Over'], default='Unknown'
    )
    merged['APE'] = abs(merged['Ratio'] - 100)
    
    # 2. Inventory Health
    avg_sales = data['sales'].groupby('SKU_ID')['Sales_Qty'].mean().reset_index().rename(columns={'Sales_Qty': 'Avg_Sales'})
    inv = pd.merge(data['stock'], avg_sales, on='SKU_ID', how='left')
    inv = pd.merge(inv, df_p[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']], on='SKU_ID', how='left')
    inv['Cover'] = np.where(inv['Avg_Sales'] > 0, inv['Stock_Qty']/inv['Avg_Sales'], 99)
    inv['Status'] = np.select(
        [inv['Cover'] < 0.8, (inv['Cover'] >= 0.8) & (inv['Cover'] <= 1.5), inv['Cover'] > 1.5],
        ['Need Replenishment', 'Ideal', 'High Stock'], default='Unknown'
    )
    
    return {
        'forecast_merged': merged,
        'inventory_health': inv,
        'mape': merged['APE'].mean(),
        'period': [m.strftime('%b %Y') for m in last_3m]
    }

# --- MAIN APP ---
client = init_gsheet_connection()
if client:
    all_data = load_and_process_data(client)
    if all_data:
        res = calculate_metrics(all_data)
        f_data = res['forecast_merged']
        
        st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE PRO</h1>', unsafe_allow_html=True)
        
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="status-indicator status-under">UNDER FORECAST<br><span style="font-size:2rem">{len(f_data[f_data["Accuracy_Status"]=="Under"])/len(f_data)*100:.1f}%</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="status-indicator status-accurate">ACCURATE<br><span style="font-size:2rem">{len(f_data[f_data["Accuracy_Status"]=="Accurate"])/len(f_data)*100:.1f}%</span></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="status-indicator status-over">OVER FORECAST<br><span style="font-size:2rem">{len(f_data[f_data["Accuracy_Status"]=="Over"])/len(f_data)*100:.1f}%</span></div>', unsafe_allow_html=True)
        with c4: st.metric("Overall Accuracy", f"{(100-res['mape']):.1f}%")

        tabs = st.tabs(["🚀 Command Center", "🏷️ Tier Flow (Sankey)", "📦 Inventory Health", "🔍 SKU Drill"])

        with tabs[0]:
            # Trend Chart
            trend = f_data.groupby('Month').agg({'Forecast_Qty':'sum', 'PO_Qty':'sum'}).reset_index()
            fig = px.line(trend, x='Month', y=['Forecast_Qty', 'PO_Qty'], title="Demand vs Supply Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            # SANKEY CHART (FIXED Key Error)
            st.subheader("Forecast Accuracy Flow by SKU Tier")
            # Pastikan kolom SKU_Tier & Accuracy_Status tidak ada yang NaN sebelum groupby
            s_data = f_data.dropna(subset=['SKU_Tier', 'Accuracy_Status'])
            s_data = s_data.groupby(['SKU_Tier', 'Accuracy_Status']).size().reset_index(name='count')
            
            nodes = list(pd.concat([s_data['SKU_Tier'], s_data['Accuracy_Status']]).unique())
            node_map = {n: i for i, n in enumerate(nodes)}
            
            fig_s = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color="#4facfe"),
                link=dict(source=s_data['SKU_Tier'].map(node_map), target=s_data['Accuracy_Status'].map(node_map), value=s_data['count'])
            )])
            st.plotly_chart(fig_s, use_container_width=True)

        with tabs[2]:
            st.subheader("Inventory Stock Health")
            st.dataframe(res['inventory_health'], use_container_width=True)

        with tabs[3]:
            st.subheader("SKU Deep Dive")
            sku_list = all_data['product']['Product_Name'].unique()
            sel_sku = st.selectbox("Pilih Produk", sku_list)
            # Filter data khusus produk terpilih
            st.info(f"Detail analisis untuk {sel_sku} akan ditampilkan di sini.")

    # SIDEBAR
    with st.sidebar:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear(); st.rerun()
