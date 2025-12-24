import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
import warnings
import io
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro V12",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM (FIXED FLOATING EFFECT + RESPONSIVE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .main-header {
        font-size: 2.5rem; font-weight: 800; color: #1e3799;
        text-align: center; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;
    }
    
    /* MONTH CARD (White Floating) */
    .month-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
        border-left: 5px solid #1e3799;
        transition: transform 0.2s; height: 100%; margin-bottom: 15px;
    }
    .month-card:hover { transform: translateY(-5px); }
    
    /* SUMMARY CARDS (Solid Color + Deep Floating Shadow) */
    .summary-card {
        border-radius: 15px; padding: 20px; text-align: center; color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15), 0 5px 15px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
        transition: transform 0.3s;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .summary-card:hover { transform: translateY(-5px); }

    .bg-red { background: linear-gradient(135deg, #e55039 0%, #eb2f06 100%); }
    .bg-green { background: linear-gradient(135deg, #78e08f 0%, #38ada9 100%); }
    .bg-orange { background: linear-gradient(135deg, #f6b93b 0%, #e58e26 100%); }
    .bg-gray { background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%); }
    .bg-blue { background: linear-gradient(135deg, #4a69bd 0%, #1e3799 100%); }
    .bg-purple { background: linear-gradient(135deg, #8e44ad 0%, #6c5ce7 100%); }
    .bg-teal { background: linear-gradient(135deg, #00cec9 0%, #00b894 100%); }
    
    .sum-val { font-size: 2.5rem; font-weight: 800; margin: 5px 0; line-height: 1; }
    .sum-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; letter-spacing: 1px;}
    .sum-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 8px;}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f2f6; border-radius: 8px 8px 0 0; font-weight: 600; border:none;}
    .stTabs [aria-selected="true"] { background-color: white; color: #1e3799; border-top: 3px solid #1e3799; }
    
    /* ALERT BADGE */
    .alert-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* RESPONSIVE DESIGN */
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .summary-card { margin-bottom: 10px; padding: 15px; }
        .sum-val { font-size: 1.8rem; }
        .month-card { padding: 15px; }
    }
    
    /* TOOLTIP STYLE */
    .tooltip-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        background-color: #6c757d;
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 11px;
        line-height: 16px;
        margin-left: 5px;
        cursor: help;
    }
    
    /* PROGRESS BAR STYLE */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3799 0%, #4a69bd 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; font-size: 3rem; margin-bottom: -15px;">💎</div>
<h1 class="main-header">INVENTORY INTELLIGENCE PRO V12</h1>
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 2rem;">
    🚀 Integrated Performance, Inventory & Sales Analytics
</div>
""", unsafe_allow_html=True)

# --- 1. CORE ENGINE (DATA LOADING) - IMPROVED CACHING ---
@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {str(e)}")
        return None

def parse_month_label(label):
    try:
        label_str = str(label).strip().upper()
        month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
        for m_name, m_num in month_map.items():
            if m_name in label_str:
                year_part = ''.join(filter(str.isdigit, label_str.replace(m_name, '')))
                year = int('20'+year_part) if len(year_part)==2 else int(year_part) if year_part else datetime.now().year
                return datetime(year, m_num, 1)
        return datetime.now()
    except:
        return datetime.now()

@st.cache_data(ttl=600, show_spinner=False)
def load_raw_data(_client):
    """Load raw data from Google Sheets"""
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        # Product Master
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_p = pd.DataFrame(ws.get_all_records())
        df_p.columns = [c.strip().replace(' ', '_') for c in df_p.columns]
        if 'SKU_ID' in df_p.columns: 
            df_p['SKU_ID'] = df_p['SKU_ID'].astype(str).str.strip()
        if 'Status' not in df_p.columns: 
            df_p['Status'] = 'Active'
        
        # Sales Data
        ws_sales = _client.open_by_url(gsheet_url).worksheet("Sales")
        df_sales = pd.DataFrame(ws_sales.get_all_records())
        df_sales.columns = [c.strip() for c in df_sales.columns]
        
        # Forecast Data
        ws_forecast = _client.open_by_url(gsheet_url).worksheet("Rofo")
        df_forecast = pd.DataFrame(ws_forecast.get_all_records())
        df_forecast.columns = [c.strip() for c in df_forecast.columns]
        
        # PO Data
        ws_po = _client.open_by_url(gsheet_url).worksheet("PO")
        df_po = pd.DataFrame(ws_po.get_all_records())
        df_po.columns = [c.strip() for c in df_po.columns]
        
        # Stock Data
        ws_stock = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock = pd.DataFrame(ws_stock.get_all_records())
        df_stock.columns = [c.strip().replace(' ', '_') for c in df_stock.columns]
        
        data = {
            'product_raw': df_p,
            'sales_raw': df_sales,
            'forecast_raw': df_forecast,
            'po_raw': df_po,
            'stock_raw': df_stock
        }
        return data
    except Exception as e:
        st.error(f"❌ Error Loading Raw Data: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def process_product_data(df_product_raw):
    """Process product data"""
    if df_product_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    df_p = df_product_raw.copy()
    df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
    return df_p, df_active

@st.cache_data(ttl=300)
def process_sheet_data(df_raw, val_col_name, active_ids):
    """Process sheet data (Sales, Forecast, PO)"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df_temp = df_raw.copy()
    if 'SKU_ID' in df_temp.columns: 
        df_temp['SKU_ID'] = df_temp['SKU_ID'].astype(str).str.strip()
    else: 
        return pd.DataFrame()
    
    # Identify month columns
    m_cols = [c for c in df_temp.columns if any(m in c.upper() for m in [
        'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
    ])]
    
    if not m_cols:
        return pd.DataFrame()
    
    df_long = df_temp[['SKU_ID'] + m_cols].melt(
        id_vars=['SKU_ID'], 
        value_vars=m_cols, 
        var_name='Month_Label', 
        value_name=val_col_name
    )
    
    df_long[val_col_name] = pd.to_numeric(df_long[val_col_name], errors='coerce').fillna(0)
    df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
    df_long['Month'] = pd.to_datetime(df_long['Month'])
    
    # Filter by active SKUs
    if active_ids:
        df_long = df_long[df_long['SKU_ID'].isin(active_ids)]
    
    return df_long

@st.cache_data(ttl=300)
def process_stock_data(df_stock_raw, active_ids):
    """Process stock data"""
    if df_stock_raw.empty:
        return pd.DataFrame()
    
    df_s = df_stock_raw.copy()
    if 'SKU_ID' in df_s.columns: 
        df_s['SKU_ID'] = df_s['SKU_ID'].astype(str).str.strip()
    else: 
        return pd.DataFrame()
    
    s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_s.columns), None)
    if s_col and 'SKU_ID' in df_s.columns:
        df_stock = df_s[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
        df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
        
        # Filter by active SKUs
        if active_ids:
            df_stock = df_stock[df_stock['SKU_ID'].isin(active_ids)]
            
        df_stock = df_stock.groupby('SKU_ID').max().reset_index()
        return df_stock
    return pd.DataFrame()

# --- 2. ANALYTICS ENGINE - IMPROVED ---
@st.cache_data(ttl=300)
def calculate_monthly_performance(df_forecast, df_po, df_product):
    """Calculate monthly forecast performance"""
    if df_forecast.empty or df_po.empty:
        return {}
    
    # Show progress
    progress_bar = st.progress(0)
    
    df_forecast['Month'] = pd.to_datetime(df_forecast['Month'])
    df_po['Month'] = pd.to_datetime(df_po['Month'])
    
    progress_bar.progress(20)
    
    df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner', suffixes=('_forecast', '_po'))
    
    progress_bar.progress(40)
    
    if not df_product.empty:
        meta = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']].rename(columns={'Status':'Prod_Status'})
        df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')
    
    progress_bar.progress(60)
    
    df_merged['Ratio'] = np.where(
        df_merged['Forecast_Qty'] > 0, 
        (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100, 
        0
    )
    
    conditions = [
        df_merged['Forecast_Qty'] == 0,
        df_merged['Ratio'] < 80, 
        (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), 
        df_merged['Ratio'] > 120
    ]
    choices = ['No Rofo', 'Under', 'Accurate', 'Over']
    
    df_merged['Status_Rofo'] = np.select(conditions, choices, default='Unknown')
    df_merged['APE'] = np.where(df_merged['Status_Rofo'] == 'No Rofo', np.nan, abs(df_merged['Ratio'] - 100))
    
    progress_bar.progress(80)
    
    monthly_stats = {}
    for month in sorted(df_merged['Month'].unique()):
        m_data = df_merged[df_merged['Month'] == month].copy()
        mean_ape = m_data['APE'].mean()
        monthly_stats[month] = {
            'accuracy': 100 - mean_ape if not pd.isna(mean_ape) else 0,
            'counts': m_data['Status_Rofo'].value_counts().to_dict(),
            'total': len(m_data),
            'data': m_data
        }
    
    progress_bar.progress(100)
    progress_bar.empty()
    
    return monthly_stats

@st.cache_data(ttl=300)
def calculate_inventory_metrics(df_stock, df_sales, df_product, months_range=3):
    """Calculate inventory metrics"""
    if df_stock.empty:
        return pd.DataFrame()
    
    # Show progress
    progress_bar = st.progress(0)
    
    if not df_sales.empty:
        df_sales['Month'] = pd.to_datetime(df_sales['Month'])
        months = sorted(df_sales['Month'].unique())[-months_range:]
        sales_period = df_sales[df_sales['Month'].isin(months)]
        avg_sales = sales_period.groupby('SKU_ID')['Sales_Qty'].mean().reset_index(name='Avg_Sales_3M')
        avg_sales['Avg_Sales_3M'] = avg_sales['Avg_Sales_3M'].round(0).astype(int)
    else:
        avg_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Sales_3M'])
    
    progress_bar.progress(30)
    
    inv = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
    inv['Avg_Sales_3M'] = inv['Avg_Sales_3M'].fillna(0)
    
    progress_bar.progress(50)
    
    if not df_product.empty:
        inv = pd.merge(inv, df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']], 
                      on='SKU_ID', how='left')
        inv = inv.rename(columns={'Status': 'Prod_Status'})
    
    progress_bar.progress(70)
    
    inv['Cover_Months'] = np.where(
        inv['Avg_Sales_3M'] > 0, 
        inv['Stock_Qty'] / inv['Avg_Sales_3M'], 
        999
    )
    
    inv['Status_Stock'] = np.select(
        [
            inv['Cover_Months'] < 0.8, 
            (inv['Cover_Months'] >= 0.8) & (inv['Cover_Months'] <= 1.5), 
            inv['Cover_Months'] > 1.5
        ],
        ['Need Replenishment', 'Ideal', 'High Stock'], 
        default='Unknown'
    )
    
    inv['Qty_to_Reduce'] = np.where(
        inv['Status_Stock'] == 'High Stock',
        inv['Stock_Qty'] - (1.5 * inv['Avg_Sales_3M']),
        0
    ).astype(int)
    inv['Qty_to_Reduce'] = np.where(inv['Qty_to_Reduce'] < 0, 0, inv['Qty_to_Reduce'])
    
    inv['Qty_to_Order'] = np.where(
        inv['Status_Stock'] == 'Need Replenishment',
        (0.8 * inv['Avg_Sales_3M']) - inv['Stock_Qty'],
        0
    ).astype(int)
    inv['Qty_to_Order'] = np.where(inv['Qty_to_Order'] < 0, 0, inv['Qty_to_Order'])
    
    progress_bar.progress(100)
    progress_bar.empty()
    
    return inv

@st.cache_data(ttl=300)
def get_last_3m_sales_pivot(df_sales, months_range=3):
    """Get sales pivot for last N months"""
    if df_sales.empty:
        return pd.DataFrame(), []
    
    df_sales['Month'] = pd.to_datetime(df_sales['Month'])
    last_n_months = sorted(df_sales['Month'].unique())[-months_range:]
    df_period = df_sales[df_sales['Month'].isin(last_n_months)].copy()
    
    df_pivot = df_period.pivot_table(
        index='SKU_ID', 
        columns='Month', 
        values='Sales_Qty', 
        aggfunc='sum'
    ).reset_index()
    
    new_cols = ['SKU_ID']
    month_names = []
    for col in df_pivot.columns:
        if isinstance(col, datetime):
            m_name = f"Sales {col.strftime('%b')}"
            new_cols.append(m_name)
            month_names.append(m_name)
    
    df_pivot.columns = new_cols
    df_pivot = df_pivot.fillna(0)
    
    return df_pivot, month_names

# --- 3. EXPORT FUNCTION ---
def export_to_excel(monthly_perf, inv_df, sales_df, product_df):
    """Export data to Excel file"""
    if not monthly_perf or inv_df.empty:
        st.warning("Tidak ada data untuk diexport")
        return None
    
    try:
        # Create Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Performance Summary
            last_month = sorted(monthly_perf.keys())[-1] if monthly_perf else None
            if last_month:
                perf_data = monthly_perf[last_month]['data']
                if not perf_data.empty:
                    perf_data.to_excel(writer, sheet_name='Performance', index=False)
            
            # Inventory Summary
            if not inv_df.empty:
                inv_df.to_excel(writer, sheet_name='Inventory', index=False)
            
            # Sales Analysis
            if not sales_df.empty:
                sales_df.to_excel(writer, sheet_name='Sales_Analysis', index=False)
            
            # Product Master
            if not product_df.empty:
                product_df.to_excel(writer, sheet_name='Product_Master', index=False)
            
            # Summary Stats
            summary_data = {
                'Metric': ['Total SKU', 'Active SKU', 'Need Replenishment', 'Ideal Stock', 'High Stock'],
                'Value': [
                    len(product_df),
                    len(product_df[product_df['Status'] == 'Active']),
                    len(inv_df[inv_df['Status_Stock'] == 'Need Replenishment']),
                    len(inv_df[inv_df['Status_Stock'] == 'Ideal']),
                    len(inv_df[inv_df['Status_Stock'] == 'High Stock'])
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

# --- 4. MAIN DATA LOADING WITH PROGRESS ---
def load_all_data():
    """Main function to load and process all data"""
    with st.spinner('🔄 Initializing connection...'):
        client = init_gsheet_connection()
    
    if not client:
        st.error("❌ Gagal terhubung ke Google Sheets")
        return None
    
    # Load raw data
    with st.spinner('📥 Loading raw data from Google Sheets...'):
        raw_data = load_raw_data(client)
    
    if not raw_data:
        st.error("❌ Gagal memuat data mentah")
        return None
    
    # Process product data
    with st.spinner('🔧 Processing product data...'):
        df_product, df_active = process_product_data(raw_data.get('product_raw', pd.DataFrame()))
        active_ids = df_active['SKU_ID'].tolist() if not df_active.empty else []
    
    # Process other data
    with st.spinner('📊 Processing sales, forecast, and PO data...'):
        df_sales = process_sheet_data(raw_data.get('sales_raw', pd.DataFrame()), "Sales_Qty", active_ids)
        df_forecast = process_sheet_data(raw_data.get('forecast_raw', pd.DataFrame()), "Forecast_Qty", active_ids)
        df_po = process_sheet_data(raw_data.get('po_raw', pd.DataFrame()), "PO_Qty", active_ids)
        df_stock = process_stock_data(raw_data.get('stock_raw', pd.DataFrame()), active_ids)
    
    # Show data status
    data_status = {
        "Product Data": f"{len(df_product)} SKUs ({len(df_active)} active)",
        "Sales Data": f"{len(df_sales)} records" if not df_sales.empty else "Empty",
        "Forecast Data": f"{len(df_forecast)} records" if not df_forecast.empty else "Empty",
        "PO Data": f"{len(df_po)} records" if not df_po.empty else "Empty",
        "Stock Data": f"{len(df_stock)} records" if not df_stock.empty else "Empty"
    }
    
    # Display data status in expander
    with st.expander("📊 Data Status", expanded=False):
        for key, value in data_status.items():
            st.write(f"**{key}:** {value}")
    
    return {
        'product': df_product,
        'product_active': df_active,
        'sales': df_sales,
        'forecast': df_forecast,
        'po': df_po,
        'stock': df_stock
    }

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Date Range Selector
    st.subheader("📅 Analysis Period")
    analysis_period = st.selectbox(
        "Select Period",
        ["Last 3 Months", "Last 6 Months", "Year to Date", "Last 12 Months"],
        key="analysis_period"
    )
    
    # Get months range based on selection
    if analysis_period == "Last 3 Months":
        months_range = 3
    elif analysis_period == "Last 6 Months":
        months_range = 6
    elif analysis_period == "Year to Date":
        months_range = datetime.now().month
    else:
        months_range = 12
    
    st.markdown("---")
    
    # Export Section
    st.subheader("📥 Export Data")
    export_format = st.selectbox(
        "Export Format",
        ["Excel (.xlsx)", "CSV"],
        key="export_format"
    )
    
    st.markdown("---")
    
    # Alert Section
    st.subheader("🚨 Alerts")
    
    # Placeholder for alerts - will be updated after data load
    alert_placeholder = st.empty()

# --- 6. MAIN DATA PROCESSING ---
all_data = load_all_data()

if all_data is None:
    st.error("Gagal memuat data. Silakan cek koneksi dan struktur data.")
    st.stop()

# Check for empty critical data
if all_data['forecast'].empty:
    st.warning("⚠️ Forecast data masih kosong atau tidak ditemukan")
    
if all_data['stock'].empty:
    st.warning("⚠️ Stock data masih kosong atau tidak ditemukan")

# Process analytics
with st.spinner('📈 Calculating performance metrics...'):
    monthly_perf = calculate_monthly_performance(
        all_data['forecast'], 
        all_data['po'], 
        all_data['product']
    )

with st.spinner('📦 Calculating inventory metrics...'):
    inv_df = calculate_inventory_metrics(
        all_data['stock'], 
        all_data['sales'], 
        all_data['product'],
        months_range=months_range
    )

with st.spinner('📊 Processing sales data...'):
    sales_pivot, sales_months_names = get_last_3m_sales_pivot(
        all_data['sales'],
        months_range=3  # Always last 3 months for pivot
    )

# Update sidebar alerts
with st.sidebar:
    alert_placeholder.empty()
    if not inv_df.empty:
        high_stock_count = len(inv_df[inv_df['Status_Stock'] == 'High Stock'])
        need_replenish_count = len(inv_df[inv_df['Status_Stock'] == 'Need Replenishment'])
        
        if high_stock_count > 10:
            st.warning(f"⚠️ **{high_stock_count} SKUs** memiliki High Stock!")
        
        if need_replenish_count > 5:
            st.error(f"🚨 **{need_replenish_count} SKUs** perlu replenishment!")
        
        if high_stock_count <= 10 and need_replenish_count <= 5:
            st.success("✅ Inventory status baik")

# --- 7. DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Performance Dashboard", 
    "📦 Inventory Analysis", 
    "📈 Sales Analysis"
])

# ==========================================
# TAB 1: PERFORMANCE DASHBOARD
# ==========================================
with tab1:
    if monthly_perf:
        # A. MONTHLY CARDS
        st.subheader("📅 Forecast Performance Trend")
        
        last_n_months = sorted(monthly_perf.keys())[-3:]
        cols = st.columns(len(last_n_months))
        
        for idx, month in enumerate(last_n_months):
            data = monthly_perf[month]
            cnt = data['counts']
            with cols[idx]:
                st.markdown(f"""
                <div class="month-card">
                    <div style="font-size:1.1rem; font-weight:700; color:#333; border-bottom:1px solid #eee; padding-bottom:5px;">
                        {month.strftime('%b %Y')}
                        <span class="tooltip-icon" title="Forecast accuracy for this month">i</span>
                    </div>
                    <div style="font-size:2.2rem; font-weight:800; color:#1e3799; margin:10px 0;">
                        {data['accuracy']:.1f}%
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
                        <span style="color:#eb2f06">Und: {cnt.get('Under',0)}</span>
                        <span style="color:#2ecc71">Acc: {cnt.get('Accurate',0)}</span>
                        <span style="color:#e67e22">Ovr: {cnt.get('Over',0)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # B. TOTAL METRICS
        if last_n_months:
            last_month = last_n_months[-1]
            lm_data = monthly_perf[last_month]['data']
            
            st.subheader(f"📊 Total Metrics ({last_month.strftime('%b %Y')})")
            grp = lm_data['Status_Rofo'].value_counts()
            
            # Calculate quantities
            u_qty = lm_data[lm_data['Status_Rofo']=='Under']['Forecast_Qty'].sum()
            a_qty = lm_data[lm_data['Status_Rofo']=='Accurate']['Forecast_Qty'].sum()
            o_qty = lm_data[lm_data['Status_Rofo']=='Over']['Forecast_Qty'].sum()
            nr_qty = lm_data[lm_data['Status_Rofo']=='No Rofo']['PO_Qty'].sum()
            total_rofo_qty = lm_data['Forecast_Qty'].sum()
            
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                st.markdown(f"""<div class="summary-card bg-red"><div class="sum-title">UNDER</div>
                <div class="sum-val">{grp.get("Under",0)}</div>
                <div class="sum-sub">{u_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="summary-card bg-green"><div class="sum-title">ACCURATE</div>
                <div class="sum-val">{grp.get("Accurate",0)}</div>
                <div class="sum-sub">{a_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="summary-card bg-orange"><div class="sum-title">OVER</div>
                <div class="sum-val">{grp.get("Over",0)}</div>
                <div class="sum-sub">{o_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
            with r4:
                st.markdown(f"""<div class="summary-card bg-gray"><div class="sum-title">NO ROFO</div>
                <div class="sum-val">{grp.get("No Rofo",0)}</div>
                <div class="sum-sub">{nr_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
            with r5:
                st.markdown(f"""<div class="summary-card bg-blue"><div class="sum-title">TOTAL ROFO QTY</div>
                <div class="sum-val">{total_rofo_qty:,.0f}</div>
                <div class="sum-sub">Forecast Quantity</div></div>""", unsafe_allow_html=True)
        
        # C. INVENTORY QUICK VIEW
        st.markdown("---")
        st.subheader("📦 Inventory Quick View")
        
        if not inv_df.empty:
            need_replenish = len(inv_df[inv_df['Status_Stock'] == 'Need Replenishment'])
            ideal = len(inv_df[inv_df['Status_Stock'] == 'Ideal'])
            high = len(inv_df[inv_df['Status_Stock'] == 'High Stock'])
            
            col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
            with col_inv1:
                st.markdown(f"""<div class="summary-card bg-red"><div class="sum-title">NEED REPLENISH</div>
                <div class="sum-val">{need_replenish}</div>
                <div class="sum-sub">SKUs</div></div>""", unsafe_allow_html=True)
            with col_inv2:
                st.markdown(f"""<div class="summary-card bg-green"><div class="sum-title">IDEAL STOCK</div>
                <div class="sum-val">{ideal}</div>
                <div class="sum-sub">SKUs</div></div>""", unsafe_allow_html=True)
            with col_inv3:
                st.markdown(f"""<div class="summary-card bg-orange"><div class="sum-title">HIGH STOCK</div>
                <div class="sum-val">{high}</div>
                <div class="sum-sub">SKUs</div></div>""", unsafe_allow_html=True)
            with col_inv4:
                total_stock = inv_df['Stock_Qty'].sum()
                st.markdown(f"""<div class="summary-card bg-purple"><div class="sum-title">TOTAL STOCK</div>
                <div class="sum-val">{total_stock:,.0f}</div>
                <div class="sum-sub">Units</div></div>""", unsafe_allow_html=True)
        
        # D. TIER ANALYSIS
        st.markdown("---")
        st.subheader("🏷️ Tier Analysis")
        
        if not lm_data.empty and 'SKU_Tier' in lm_data.columns:
            tier_df = lm_data.dropna(subset=['SKU_Tier'])
            
            col_t1, col_t2 = st.columns([2, 1])
            
            with col_t1:
                tier_agg = tier_df.groupby(['SKU_Tier', 'Status_Rofo']).size().reset_index(name='Count')
                fig = px.bar(tier_agg, x='SKU_Tier', y='Count', color='Status_Rofo',
                             color_discrete_map={
                                 'Under':'#e55039', 
                                 'Accurate':'#38ada9', 
                                 'Over':'#f6b93b', 
                                 'No Rofo':'#95a5a6'
                             },
                             height=300, 
                             title="SKU Distribution by Tier")
                fig.update_layout(
                    margin=dict(t=30, b=0, l=0, r=0), 
                    plot_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_t2:
                st.markdown("##### Accuracy per Tier")
                tier_sum = tier_df.groupby(['SKU_Tier', 'Status_Rofo']).size().unstack(fill_value=0)
                if not tier_sum.empty:
                    tier_sum['Total'] = tier_sum.sum(axis=1)
                    tier_sum['Acc %'] = (tier_sum.get('Accurate', 0) / tier_sum['Total'] * 100).round(1)
                    st.dataframe(
                        tier_sum[['Accurate', 'Under', 'Over', 'Acc %']].sort_values('Acc %', ascending=False), 
                        use_container_width=True,
                        height=300
                    )

        # E. EVALUASI ROFO TABLE
        st.markdown("---")
        st.subheader(f"📋 Evaluasi Rofo - {last_month.strftime('%b %Y')}")
        
        if not lm_data.empty:
            base_eval = pd.merge(
                lm_data, 
                inv_df[['SKU_ID', 'Stock_Qty', 'Avg_Sales_3M']], 
                on='SKU_ID', 
                how='left'
            )
            
            if not sales_pivot.empty:
                base_eval = pd.merge(base_eval, sales_pivot, on='SKU_ID', how='left')
                for col in sales_months_names:
                    if col in base_eval.columns:
                        base_eval[col] = base_eval[col].fillna(0).astype(int)
            
            sales_cols = [c for c in base_eval.columns if c in sales_months_names]
            final_cols = [
                'SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Prod_Status', 'Status_Rofo',
                'Forecast_Qty', 'PO_Qty', 'Ratio', 'Stock_Qty', 'Avg_Sales_3M'
            ] + sales_cols
            
            final_cols = [c for c in final_cols if c in base_eval.columns]
            
            df_display = base_eval[final_cols].rename(columns={
                'Prod_Status': 'Product Status',
                'Ratio': 'Achv %',
                'Stock_Qty': 'Stock',
                'Avg_Sales_3M': 'Avg Sales (3M)'
            })
            
            t_all, t_under, t_over, t_nr = st.tabs(["All SKU", "Under Forecast", "Over Forecast", "No Rofo"])
            
            cfg = {
                "Achv %": st.column_config.NumberColumn(format="%.0f%%"),
                "Stock": st.column_config.NumberColumn(format="%d"),
                "Avg Sales (3M)": st.column_config.NumberColumn(format="%d"),
                "Forecast_Qty": st.column_config.NumberColumn(format="%d"),
                "PO_Qty": st.column_config.NumberColumn(format="%d")
            }
            
            with t_all: 
                st.dataframe(df_display, column_config=cfg, use_container_width=True, height=500)
            with t_under: 
                st.dataframe(df_display[df_display['Status_Rofo']=='Under'], column_config=cfg, use_container_width=True, height=400)
            with t_over: 
                st.dataframe(df_display[df_display['Status_Rofo']=='Over'], column_config=cfg, use_container_width=True, height=400)
            with t_nr: 
                st.dataframe(df_display[df_display['Status_Rofo']=='No Rofo'], column_config=cfg, use_container_width=True, height=400)
    else:
        st.warning("Data performance belum tersedia. Pastikan data Forecast dan PO sudah diisi.")

# ==========================================
# TAB 2: INVENTORY ANALYSIS
# ==========================================
with tab2:
    st.subheader("📦 Inventory Overview")
    
    if not inv_df.empty:
        # 1. SALES METRICS
        st.markdown("##### Recent Sales Performance")
        
        s_months_data = {}
        if 'sales' in all_data and not all_data['sales'].empty:
            sales_df = all_data['sales'].copy()
            sales_df['Month'] = pd.to_datetime(sales_df['Month'], errors='coerce')
            months = sorted(sales_df['Month'].dropna().unique())[-3:]
            for m in months:
                qty = sales_df[sales_df['Month']==m]['Sales_Qty'].sum()
                s_months_data[m.strftime('%b')] = qty
        
        m1, m2, m3, m4 = st.columns(4)
        idx = 0
        for m_name, qty in s_months_data.items():
            if idx == 0: 
                with m1: 
                    st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            elif idx == 1: 
                with m2: 
                    st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            elif idx == 2: 
                with m3: 
                    st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            idx += 1
        
        with m4: 
            st.metric("Total Stock", f"{inv_df['Stock_Qty'].sum():,.0f}")
        
        st.markdown("---")
        
        # 2. STOCK BREAKDOWN & ACTIONABLE INSIGHTS
        c_don, c_act = st.columns([1, 2])
        
        with c_don:
            st.markdown("##### Stock Status Distribution")
            qty_stat = inv_df.groupby('Status_Stock')['Stock_Qty'].sum().reset_index()
            count_stat = inv_df['Status_Stock'].value_counts().reset_index()
            count_stat.columns = ['Status_Stock', 'SKU_Count']
            qty_stat = pd.merge(qty_stat, count_stat, on='Status_Stock')
            
            fig_don = px.pie(qty_stat, values='Stock_Qty', names='Status_Stock', hole=0.5,
                             hover_data=['SKU_Count'],
                             color='Status_Stock',
                             color_discrete_map={
                                 'Need Replenishment':'#e55039', 
                                 'Ideal':'#38ada9', 
                                 'High Stock':'#f6b93b'
                             })
            fig_don.update_layout(
                height=350, 
                margin=dict(t=0,b=0, l=0, r=0), 
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_don, use_container_width=True)
            
        with c_act:
            st.markdown("##### Actionable Insights")
            
            to_order = inv_df['Qty_to_Order'].sum()
            to_reduce = inv_df['Qty_to_Reduce'].sum()
            total_value = to_order + to_reduce
            
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                st.markdown(f"""
                <div style="background:#e8f5e9; padding:15px; border-radius:10px; text-align:center; border:1px solid #38ada9;">
                    <div style="color:#38ada9; font-weight:bold;">QTY TO ORDER</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#333;">{to_order:,.0f}</div>
                    <div style="font-size:0.8rem;">For SKU Need Replenishment</div>
                </div>
                """, unsafe_allow_html=True)
                
            with ac2:
                st.markdown(f"""
                <div style="background:#fff3e0; padding:15px; border-radius:10px; text-align:center; border:1px solid #f6b93b;">
                    <div style="color:#f6b93b; font-weight:bold;">QTY TO REDUCE</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#333;">{to_reduce:,.0f}</div>
                    <div style="font-size:0.8rem;">For High Stock SKU</div>
                </div>
                """, unsafe_allow_html=True)
            
            with ac3:
                st.markdown(f"""
                <div style="background:#f0f8ff; padding:15px; border-radius:10px; text-align:center; border:1px solid #4a69bd;">
                    <div style="color:#4a69bd; font-weight:bold;">TOTAL ACTION</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#333;">{total_value:,.0f}</div>
                    <div style="font-size:0.8rem;">Total Units to Act</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("###### Breakdown per Tier")
            if 'SKU_Tier' in inv_df.columns:
                tier_act = inv_df.groupby('SKU_Tier').agg({
                    'Qty_to_Order': 'sum',
                    'Qty_to_Reduce': 'sum',
                    'SKU_ID': 'count'
                }).reset_index()
                tier_act = tier_act.rename(columns={'SKU_ID': 'SKU_Count'})
                
                st.dataframe(
                    tier_act.sort_values('Qty_to_Order', ascending=False), 
                    column_config={
                        "Qty_to_Order": st.column_config.NumberColumn("Qty to Order", format="%d"),
                        "Qty_to_Reduce": st.column_config.NumberColumn("Qty to Reduce", format="%d"),
                        "SKU_Count": st.column_config.NumberColumn("SKU Count", format="%d")
                    }, 
                    use_container_width=True, 
                    height=200
                )

        # 3. DETAIL TABLE
        st.markdown("---")
        st.subheader("📋 Inventory Detail SKU")
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            fil_stat = st.multiselect(
                "Filter Status", 
                inv_df['Status_Stock'].unique(), 
                default=['Need Replenishment', 'High Stock']
            )
        with col_f2:
            if 'Brand' in inv_df.columns:
                brands = ['All'] + sorted(inv_df['Brand'].unique().tolist())
                fil_brand = st.selectbox("Filter Brand", brands)
            else:
                fil_brand = 'All'
        with col_f3:
            if 'SKU_Tier' in inv_df.columns:
                tiers = ['All'] + sorted(inv_df['SKU_Tier'].unique().tolist())
                fil_tier = st.selectbox("Filter Tier", tiers)
            else:
                fil_tier = 'All'
        
        # Apply filters
        inv_filtered = inv_df.copy()
        if fil_stat:
            inv_filtered = inv_filtered[inv_filtered['Status_Stock'].isin(fil_stat)]
        if fil_brand != 'All' and 'Brand' in inv_filtered.columns:
            inv_filtered = inv_filtered[inv_filtered['Brand'] == fil_brand]
        if fil_tier != 'All' and 'SKU_Tier' in inv_filtered.columns:
            inv_filtered = inv_filtered[inv_filtered['SKU_Tier'] == fil_tier]
        
        # Display columns
        view_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Prod_Status', 
                     'Stock_Qty', 'Avg_Sales_3M', 'Cover_Months', 'Status_Stock',
                     'Qty_to_Order', 'Qty_to_Reduce']
        view_cols = [c for c in view_cols if c in inv_filtered.columns]
        
        inv_show = inv_filtered[view_cols].rename(columns={
            'Prod_Status': 'Product Status',
            'Stock_Qty': 'Stock Qty',
            'Avg_Sales_3M': 'Avg Sales (3M)',
            'Cover_Months': 'Cover Month',
            'Status_Stock': 'Status Stock',
            'Qty_to_Order': 'Qty to Order',
            'Qty_to_Reduce': 'Qty to Reduce'
        })
        
        st.dataframe(
            inv_show.sort_values('Cover Month', ascending=False),
            column_config={
                "Avg Sales (3M)": st.column_config.NumberColumn(format="%d"),
                "Cover Month": st.column_config.NumberColumn(format="%.1f"),
                "Stock Qty": st.column_config.NumberColumn(format="%d"),
                "Qty to Order": st.column_config.NumberColumn(format="%d"),
                "Qty to Reduce": st.column_config.NumberColumn(format="%d")
            },
            use_container_width=True, 
            height=500
        )
    else:
        st.warning("Data Inventory belum tersedia. Pastikan data Stock sudah diisi.")

# ==========================================
# TAB 3: SALES ANALYSIS
# ==========================================
with tab3:
    st.subheader("📈 Sales vs Forecast Analysis")
    
    if 'sales' in all_data and 'forecast' in all_data:
        # A. TOTAL SALES VS FORECAST (ALL MONTHS)
        s_agg = all_data['sales'].groupby('Month')['Sales_Qty'].sum().reset_index()
        f_agg = all_data['forecast'].groupby('Month')['Forecast_Qty'].sum().reset_index()
        
        combo = pd.merge(s_agg, f_agg, on='Month', how='outer').fillna(0)
        combo_melt = combo.melt('Month', var_name='Type', value_name='Qty')
        combo_melt['Type'] = combo_melt['Type'].replace({
            'Sales_Qty': 'Actual Sales',
            'Forecast_Qty': 'Forecast'
        })
        
        st.markdown("##### 1. Total Overview (All Months)")
        fig_trend = px.bar(
            combo_melt, 
            x='Month', 
            y='Qty', 
            color='Type', 
            barmode='group',
            color_discrete_map={'Actual Sales':'#1e3799', 'Forecast':'#82ccdd'},
            labels={'Qty': 'Quantity', 'Month': 'Month'}
        )
        fig_trend.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        # B. FILTERED CHART
        st.markdown("##### 2. Filter Trend by Brand/Tier")
        
        col_filt1, col_filt2 = st.columns(2)
        with col_filt1:
            brands = sorted(all_data['product']['Brand'].unique().tolist()) if not all_data['product'].empty else []
            sel_brand = st.selectbox("Select Brand", ["All"] + brands, key="sales_brand")
        
        with col_filt2:
            tiers = sorted(all_data['product']['SKU_Tier'].unique().tolist()) if not all_data['product'].empty else []
            sel_tier = st.selectbox("Select Tier", ["All"] + tiers, key="sales_tier")
        
        s_raw = all_data['sales'].copy()
        f_raw = all_data['forecast'].copy()
        
        # Apply filters
        if sel_brand != "All" or sel_tier != "All":
            filter_skus = all_data['product'].copy()
            if sel_brand != "All":
                filter_skus = filter_skus[filter_skus['Brand'] == sel_brand]
            if sel_tier != "All":
                filter_skus = filter_skus[filter_skus['SKU_Tier'] == sel_tier]
            
            brand_tier_skus = filter_skus['SKU_ID'].tolist()
            s_raw = s_raw[s_raw['SKU_ID'].isin(brand_tier_skus)]
            f_raw = f_raw[f_raw['SKU_ID'].isin(brand_tier_skus)]
        
        s_agg_b = s_raw.groupby('Month')['Sales_Qty'].sum().reset_index()
        f_agg_b = f_raw.groupby('Month')['Forecast_Qty'].sum().reset_index()
        combo_b = pd.merge(s_agg_b, f_agg_b, on='Month', how='outer').fillna(0)
        combo_b = combo_b.rename(columns={'Sales_Qty': 'Actual Sales', 'Forecast_Qty': 'Forecast'})
        
        fig_b = px.line(
            combo_b, 
            x='Month', 
            y=['Actual Sales', 'Forecast'], 
            markers=True,
            title=f"Sales Trend: Brand={sel_brand}, Tier={sel_tier}",
            color_discrete_map={'Actual Sales':'#1e3799', 'Forecast':'#e55039'}
        )
        fig_b.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_b, use_container_width=True)
        
        # Calculate accuracy metrics
        if not combo_b.empty:
            combo_b['Accuracy %'] = np.where(
                combo_b['Forecast'] > 0,
                (combo_b['Actual Sales'] / combo_b['Forecast']) * 100,
                0
            )
            
            avg_accuracy = combo_b['Accuracy %'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        st.markdown("---")
        
        # C. DETAIL SKU ANALYSIS
        st.subheader("📋 Detail SKU Sales vs Forecast (3 Bulan Terakhir)")
        
        last_3m = sorted(s_raw['Month'].unique())[-3:] if not s_raw.empty else []
        
        if last_3m:
            # Pivot Sales
            s_3m = s_raw[s_raw['Month'].isin(last_3m)]
            s_piv = s_3m.pivot_table(
                index='SKU_ID', 
                columns='Month', 
                values='Sales_Qty', 
                aggfunc='sum'
            ).reset_index()
            s_piv.columns = ['SKU_ID'] + [f"Sales {c.strftime('%b')}" for c in s_piv.columns if isinstance(c, datetime)]
            
            # Pivot Forecast
            f_3m = f_raw[f_raw['Month'].isin(last_3m)]
            f_piv = f_3m.pivot_table(
                index='SKU_ID', 
                columns='Month', 
                values='Forecast_Qty', 
                aggfunc='sum'
            ).reset_index()
            f_piv.columns = ['SKU_ID'] + [f"Fc {c.strftime('%b')}" for c in f_piv.columns if isinstance(c, datetime)]
            
            det = pd.merge(s_piv, f_piv, on='SKU_ID', how='outer').fillna(0)
            
            if not all_data['product'].empty:
                det = pd.merge(
                    det, 
                    all_data['product'][['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status']], 
                    on='SKU_ID', 
                    how='left'
                )
                det = det.rename(columns={'Status':'Product Status'})
            
            # Apply brand/tier filter for table
            if sel_brand != "All":
                det = det[det['Brand'] == sel_brand]
            if sel_tier != "All":
                det = det[det['SKU_Tier'] == sel_tier]
            
            sales_cols = [c for c in det.columns if c.startswith('Sales ')]
            fc_cols = [c for c in det.columns if c.startswith('Fc ')]
            
            if sales_cols and fc_cols:
                det['Total Sales 3M'] = det[sales_cols].sum(axis=1)
                det['Total Fc 3M'] = det[fc_cols].sum(axis=1)
                det['Dev %'] = np.where(
                    det['Total Fc 3M'] > 0, 
                    (det['Total Sales 3M'] - det['Total Fc 3M']) / det['Total Fc 3M'] * 100, 
                    0
                )
                
                final_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Product Status'] + \
                            sales_cols + fc_cols + ['Total Sales 3M', 'Total Fc 3M', 'Dev %']
                final_cols = [c for c in final_cols if c in det.columns]
                
                st.dataframe(
                    det[final_cols].sort_values('Dev %', ascending=False), 
                    column_config={
                        "Dev %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Total Sales 3M": st.column_config.NumberColumn(format="%d"),
                        "Total Fc 3M": st.column_config.NumberColumn(format="%d")
                    }, 
                    use_container_width=True,
                    height=500
                )
    else:
        st.warning("Data Sales dan Forecast diperlukan untuk analisis ini.")

# --- 8. EXPORT FUNCTIONALITY ---
with st.sidebar:
    st.markdown("---")
    
    # Prepare data for export
    if st.button("📥 Export Full Report", use_container_width=True):
        with st.spinner("Preparing export..."):
            # Get last month's performance data
            last_month_data = None
            if monthly_perf:
                last_month = sorted(monthly_perf.keys())[-1] if monthly_perf else None
                if last_month:
                    last_month_data = monthly_perf[last_month]['data']
            
            # Get sales analysis data from tab 3
            sales_analysis_data = pd.DataFrame()
            if 'sales' in all_data and 'forecast' in all_data:
                s_raw = all_data['sales'].copy()
                f_raw = all_data['forecast'].copy()
                last_3m = sorted(s_raw['Month'].unique())[-3:] if not s_raw.empty else []
                
                if last_3m:
                    s_3m = s_raw[s_raw['Month'].isin(last_3m)]
                    s_piv = s_3m.pivot_table(index='SKU_ID', columns='Month', values='Sales_Qty', aggfunc='sum').reset_index()
                    s_piv.columns = ['SKU_ID'] + [f"Sales {c.strftime('%b')}" for c in s_piv.columns if isinstance(c, datetime)]
                    
                    f_3m = f_raw[f_raw['Month'].isin(last_3m)]
                    f_piv = f_3m.pivot_table(index='SKU_ID', columns='Month', values='Forecast_Qty', aggfunc='sum').reset_index()
                    f_piv.columns = ['SKU_ID'] + [f"Fc {c.strftime('%b')}" for c in f_piv.columns if isinstance(c, datetime)]
                    
                    sales_analysis_data = pd.merge(s_piv, f_piv, on='SKU_ID', how='outer').fillna(0)
                    
                    if not all_data['product'].empty:
                        sales_analysis_data = pd.merge(
                            sales_analysis_data, 
                            all_data['product'][['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']], 
                            on='SKU_ID', 
                            how='left'
                        )
            
            # Export to Excel
            excel_file = export_to_excel(
                monthly_perf, 
                inv_df, 
                sales_analysis_data, 
                all_data['product']
            )
            
            if excel_file:
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_file,
                    file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success("✅ Report siap di-download!")
