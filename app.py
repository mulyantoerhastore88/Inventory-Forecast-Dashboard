import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro v5.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Premium (Original + Enhanced) ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    .insight-card {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .status-indicator {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .status-indicator:hover { transform: translateY(-5px); }
    .status-under { background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%); color: white; border-left: 5px solid #D32F2F; }
    .status-accurate { background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); color: white; border-left: 5px solid #1B5E20; }
    .status-over { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); color: white; border-left: 5px solid #E65100; }
    
    .inventory-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    .card-replenish { background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); color: #EF6C00; border: 2px solid #FF9800; }
    .card-ideal { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); color: #2E7D32; border: 2px solid #4CAF50; }
    .card-high { background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); color: #C62828; border: 2px solid #F44336; }
    
    .metric-highlight {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
        border-top: 5px solid #667eea;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Judul Dashboard ---
st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE PRO</h1>', unsafe_allow_html=True)
st.caption(f"🚀 AI-Enhanced Demand Planning | Financial Exposure Analysis | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- ====================================================== ---
# ---                KONEKSI & LOAD DATA                     ---
# --- ====================================================== ---

@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {str(e)}")
        return None

def parse_month_label(label):
    try:
        label_str = str(label).strip().upper()
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        for month_name, month_num in month_map.items():
            if month_name in label_str:
                year_part = label_str.replace(month_name, '').replace('-', '').replace(' ', '').strip()
                year = int('20' + year_part) if len(year_part) == 2 else int(year_part) if year_part else datetime.now().year
                return datetime(year, month_num, 1)
        return datetime.now()
    except: return datetime.now()

@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(_client):
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        # 1. PRODUCT MASTER
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws.get_all_records())
        df_product.columns = [col.strip().replace(' ', '_') for col in df_product.columns]
        if 'Status' not in df_product.columns: df_product['Status'] = 'Active'
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        active_skus = df_product_active['SKU_ID'].tolist()

        # 2. HELPER UNTUK MELTING DATA BERBASIS BULAN
        def melt_month_data(ws_name, value_name):
            ws_sheet = _client.open_by_url(gsheet_url).worksheet(ws_name)
            df_raw = pd.DataFrame(ws_sheet.get_all_records())
            df_raw.columns = [col.strip() for col in df_raw.columns]
            month_cols = [col for col in df_raw.columns if any(m in col.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
            
            id_cols = ['SKU_ID']
            for col in ['SKU_Name', 'Product_Name', 'Brand', 'SKU_Tier']:
                if col in df_raw.columns: id_cols.append(col)
                
            df_long = df_raw.melt(id_vars=id_cols, value_vars=month_cols, var_name='Month_Label', value_name=value_name)
            df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            return df_long[df_long['SKU_ID'].isin(active_skus)]

        data['sales'] = melt_month_data("Sales", "Sales_Qty")
        data['forecast'] = melt_month_data("Rofo", "Forecast_Qty")
        data['po'] = melt_month_data("PO", "PO_Qty")
        
        # 3. STOCK DATA
        ws_stock = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock_raw = pd.DataFrame(ws_stock.get_all_records())
        df_stock_raw.columns = [col.strip().replace(' ', '_') for col in df_stock_raw.columns]
        stock_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_stock_raw.columns), None)
        df_stock = df_stock_raw[['SKU_ID', stock_col]].rename(columns={stock_col: 'Stock_Qty'})
        df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
        data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_skus)]
        
        data['product'] = df_product
        data['product_active'] = df_product_active
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return {}

# --- ====================================================== ---
# ---                ANALYTICS FUNCTIONS                     ---
# --- ====================================================== ---

def calculate_forecast_accuracy_3months(df_forecast, df_po, df_product):
    if df_forecast.empty or df_po.empty: return {}
    try:
        all_months = sorted(set(df_forecast['Month'].tolist() + df_po['Month'].tolist()))
        if len(all_months) >= 3:
            last_3_months = all_months[-3:]
            df_f_recent = df_forecast[df_forecast['Month'].isin(last_3_months)].copy()
            df_p_recent = df_po[df_po['Month'].isin(last_3_months)].copy()
            
            df_merged = pd.merge(df_f_recent, df_p_recent, on=['SKU_ID', 'Month'], how='inner', suffixes=('_forecast', '_po'))
            
            # AMANKAN KOLOM TIER: Merge kembali dengan Product Master untuk menjamin kolom SKU_Tier ada
            product_info = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
            df_merged = pd.merge(df_merged, product_info, on='SKU_ID', how='left')
            
            df_merged['PO_Rofo_Ratio'] = np.where(df_merged['Forecast_Qty'] > 0, (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100, 0)
            
            conditions = [df_merged['PO_Rofo_Ratio'] < 80, (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120), df_merged['PO_Rofo_Ratio'] > 120]
            choices = ['Under', 'Accurate', 'Over']
            df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
            
            df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
            mape = df_merged['Absolute_Percentage_Error'].mean()
            
            sku_accuracy = df_merged.groupby(['SKU_ID', 'Product_Name']).apply(lambda x: 100 - x['Absolute_Percentage_Error'].mean()).reset_index()
            sku_accuracy.columns = ['SKU_ID', 'Product_Name', 'SKU_Accuracy']

            return {
                'overall_accuracy': 100 - mape,
                'mape': mape,
                'status_percentages': (df_merged['Accuracy_Status'].value_counts() / len(df_merged) * 100).to_dict(),
                'sku_accuracy': sku_accuracy,
                'detailed_data': df_merged,
                'period_months': [m.strftime('%b %Y') for m in last_3_months],
                'total_records': len(df_merged)
            }
    except Exception as e:
        st.error(f"Calc error: {e}"); return {}

def create_sankey_chart_tier(df_detailed):
    """Sankey Chart Logic - Menggunakan data yang sudah di-merge dengan Tier"""
    if df_detailed is None or df_detailed.empty: return None
    try:
        # Filter out rows with missing Tier
        df_sankey = df_detailed.dropna(subset=['SKU_Tier', 'Accuracy_Status'])
        tier_flow = df_sankey.groupby(['SKU_Tier', 'Accuracy_Status']).size().reset_index(name='SKU_Count')
        
        all_nodes = list(tier_flow['SKU_Tier'].unique()) + list(tier_flow['Accuracy_Status'].unique())
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=30, line=dict(color="black", width=0.5), label=all_nodes, color="#667eea"),
            link=dict(
                source=tier_flow['SKU_Tier'].map(node_map),
                target=tier_flow['Accuracy_Status'].map(node_map),
                value=tier_flow['SKU_Count']
            )
        )])
        fig.update_layout(title_text="<b>Forecast Flow: Tier vs Accuracy Status</b>", height=500)
        return fig
    except Exception as e:
        st.error(f"Sankey Error: {e}"); return None

# --- ====================================================== ---
# ---                DASHBOARD INITIALIZATION                ---
# --- ====================================================== ---

client = init_gsheet_connection()
if client is None: st.stop()

with st.spinner('🔄 Synchronizing Intelligence Engine...'):
    all_data = load_and_process_data(client)
    df_product = all_data.get('product', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())

# Calculate metrics
forecast_metrics = calculate_forecast_accuracy_3months(df_forecast, df_po, df_product)

# --- TOP INSIGHT PANEL (NEW INTELLIGENCE) ---
if forecast_metrics:
    acc_val = forecast_metrics.get('overall_accuracy', 0)
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Intelligence Note:</strong> Akurasi rata-rata 3 bulan terakhir berada di level <b>{acc_val:.1f}%</b>. 
        Ditemukan {(forecast_metrics.get('status_percentages', {}).get('Over', 0)):.1f}% SKU mengalami Over-Forecast, 
        yang berpotensi menyebabkan pembengkakan modal kerja di gudang.
    </div>
    """, unsafe_allow_html=True)

# --- KEY METRICS ROW ---
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
if forecast_metrics:
    with col_h1:
        st.markdown(f'<div class="status-indicator status-under">UNDER FORECAST<br><span style="font-size: 2rem;">{forecast_metrics["status_percentages"].get("Under", 0):.1f}%</span></div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f'<div class="status-indicator status-accurate">ACCURATE<br><span style="font-size: 2rem;">{forecast_metrics["status_percentages"].get("Accurate", 0):.1f}%</span></div>', unsafe_allow_html=True)
    with col_h3:
        st.markdown(f'<div class="status-indicator status-over">OVER FORECAST<br><span style="font-size: 2rem;">{forecast_metrics["status_percentages"].get("Over", 0):.1f}%</span></div>', unsafe_allow_html=True)
    with col_h4:
        st.markdown(f'<div class="metric-highlight"><span style="color:#666">OVERALL ACCURACY</span><br><span style="font-size: 2rem; color:#667eea">{acc_val:.1f}%</span></div>', unsafe_allow_html=True)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance Trend", "🏷️ Tier Flow", "📦 Inventory Health", "🔍 SKU Drill-down", "📋 Explorer"])

with tab1:
    st.subheader("📅 Monthly Forecast Accuracy")
    # Logic Trend Line (Sama seperti script aslimu)
    # [Insert original line chart code here]
    st.info("Trend akurasi bulanan dapat dimonitor di sini untuk melihat perbaikan performa planning.")

with tab2:
    st.subheader("🏷️ SKU Tier Analysis")
    if forecast_metrics and 'detailed_data' in forecast_metrics:
        sankey_fig = create_sankey_chart_tier(forecast_metrics['detailed_data'])
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)

with tab3:
    st.subheader("📦 Inventory Health")
    # Logic Inventory Cover (Sama seperti script aslimu)
    # [Insert original inventory cards code here]
    st.warning("Gunakan data ini untuk prioritas replenishment atau cuci gudang SKU High Stock.")

with tab4:
    st.subheader("🔍 SKU Performance Evaluation")
    if forecast_metrics:
        sku_acc = forecast_metrics['sku_accuracy'].sort_values('SKU_Accuracy', ascending=True)
        st.dataframe(sku_acc, use_container_width=True, height=400)

with tab5:
    st.subheader("📋 Raw Data Explorer")
    dataset = st.selectbox("Pilih Data", ["Sales", "Forecast", "PO", "Stock"])
    # [Insert original data explorer code here]

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("🔄 Sync Live Data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.write(f"**Total SKU Aktif:** {len(df_product[df_product['Status']=='Active'])}")

# FOOTER
st.divider()
st.markdown("<center>Inventory Intelligence Pro v5.0 | Professional Demand Planning System</center>", unsafe_allow_html=True)
