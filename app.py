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

# --- Custom CSS Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 1.1rem;
        color: #444;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .status-indicator {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        color: white;
    }
    .status-indicator:hover { transform: translateY(-5px); }
    .status-under { background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%); border-left: 5px solid #D32F2F; }
    .status-accurate { background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); border-left: 5px solid #1B5E20; }
    .status-over { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); border-left: 5px solid #E65100; }
    
    .inventory-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
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
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #5a67d8 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .sankey-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Judul Dashboard ---
st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE DASHBOARD</h1>', unsafe_allow_html=True)
st.caption(f"🚀 Professional Inventory Control & Demand Planning | Real-time Analytics | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- ====================================================== ---
# ---                KONEKSI & LOAD DATA                     ---
# --- ====================================================== ---

@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    """Inisialisasi koneksi ke Google Sheets"""
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
    """Parse berbagai format bulan ke datetime dengan aman"""
    try:
        label_str = str(label).strip().upper()
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        
        for month_name, month_num in month_map.items():
            if month_name in label_str:
                # Ambil angka tahun saja
                year_part = ''.join(filter(str.isdigit, label_str.replace(month_name, '')))
                if year_part:
                    year = int('20' + year_part) if len(year_part) == 2 else int(year_part)
                else:
                    year = datetime.now().year
                return datetime(year, month_num, 1)
        return datetime.now()
    except:
        return datetime.now()

@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(_client):
    """
    Load data dengan prinsip 'SKU_ID Centric'.
    Hanya mengambil SKU_ID dan Angka dari sheet transaksi untuk menghindari KeyError.
    """
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    
    try:
        # 1. PRODUCT MASTER (Metadata Utama)
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws.get_all_records())
        df_product.columns = [col.strip().replace(' ', '_') for col in df_product.columns]
        
        if 'Status' not in df_product.columns: df_product['Status'] = 'Active'
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        
        # Ambil list SKU aktif untuk filter
        active_skus = df_product_active['SKU_ID'].tolist()
        
        # Helper function untuk melt data transaksi (Sales, PO, Rofo)
        def robust_melt(sheet_name, value_name):
            ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df_temp = pd.DataFrame(ws_temp.get_all_records())
            
            # Identifikasi kolom bulan
            month_cols = [col for col in df_temp.columns if any(m in col.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
            
            # Kita HANYA butuh SKU_ID dan Bulan. Abaikan kolom nama produk dll di sheet ini agar tidak error.
            if 'SKU_ID' not in df_temp.columns:
                return pd.DataFrame()
                
            df_long = df_temp.melt(
                id_vars=['SKU_ID'], # Hanya keep SKU_ID sebagai anchor
                value_vars=month_cols,
                var_name='Month_Label',
                value_name=value_name
            )
            
            df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            
            # Filter hanya SKU aktif
            return df_long[df_long['SKU_ID'].isin(active_skus)]

        # 2. LOAD DATA TRANSAKSI
        data['sales'] = robust_melt("Sales", "Sales_Qty")
        data['forecast'] = robust_melt("Rofo", "Forecast_Qty")
        data['po'] = robust_melt("PO", "PO_Qty")
        
        # 3. STOCK DATA
        ws_stock = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock_raw = pd.DataFrame(ws_stock.get_all_records())
        df_stock_raw.columns = [col.strip().replace(' ', '_') for col in df_stock_raw.columns]
        
        # Cari kolom stok yang valid (bisa beda-beda nama)
        stock_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_stock_raw.columns), None)
        
        if stock_col and 'SKU_ID' in df_stock_raw.columns:
            df_stock = df_stock_raw[['SKU_ID', stock_col]].copy()
            df_stock.columns = ['SKU_ID', 'Stock_Qty'] # Rename standard
            df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
            df_stock = df_stock.groupby('SKU_ID')['Stock_Qty'].max().reset_index()
            data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_skus)]
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
        
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
    """Calculate forecast accuracy & Generate Insight"""
    metrics = {}
    
    if df_forecast.empty or df_po.empty:
        return metrics
    
    try:
        # Get unique months
        all_months = sorted(set(df_forecast['Month'].tolist() + df_po['Month'].tolist()))
        
        if len(all_months) >= 3:
            last_3_months = all_months[-3:]
            
            # Filter
            df_f = df_forecast[df_forecast['Month'].isin(last_3_months)].copy()
            df_p = df_po[df_po['Month'].isin(last_3_months)].copy()
            
            # Merge Forecast & PO based on SKU_ID & Month
            df_merged = pd.merge(df_f, df_p, on=['SKU_ID', 'Month'], how='inner')
            
            # MERGE METADATA (Tier, Name) DARI PRODUCT MASTER
            # Ini kunci agar tidak error KeyError 'Product_Name'
            cols_to_merge = ['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']
            # Pastikan kolom ada di product master
            available_cols = [c for c in cols_to_merge if c in df_product.columns]
            df_merged = pd.merge(df_merged, df_product[available_cols].drop_duplicates('SKU_ID'), on='SKU_ID', how='left')
            
            # Calculate Ratios
            df_merged['PO_Rofo_Ratio'] = np.where(
                df_merged['Forecast_Qty'] > 0,
                (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
                0
            )
            
            conditions = [
                df_merged['PO_Rofo_Ratio'] < 80,
                (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
                df_merged['PO_Rofo_Ratio'] > 120
            ]
            df_merged['Accuracy_Status'] = np.select(conditions, ['Under', 'Accurate', 'Over'], default='Unknown')
            
            # Metrics
            df_merged['APE'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
            mape = df_merged['APE'].mean()
            overall_accuracy = 100 - mape
            
            status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
            total = len(df_merged)
            status_pct = {k: (v/total*100) for k, v in status_counts.items()}
            
            # SKU Level Accuracy
            sku_acc = df_merged.groupby('SKU_ID')['APE'].mean().reset_index()
            sku_acc['SKU_Accuracy'] = 100 - sku_acc['APE']
            # Add names
            if 'Product_Name' in df_merged.columns:
                sku_names = df_merged[['SKU_ID', 'Product_Name']].drop_duplicates()
                sku_acc = pd.merge(sku_acc, sku_names, on='SKU_ID', how='left')
            
            # Generate AI Insight String
            period_str = ", ".join([m.strftime('%b') for m in last_3_months])
            insight = f"Analisis periode **{period_str}** menunjukkan akurasi rata-rata **{overall_accuracy:.1f}%**. "
            if status_pct.get('Under', 0) > 30:
                insight += f"Perhatian: Terdeteksi **{status_pct.get('Under', 0):.1f}%** kejadian Under Forecast yang berisiko Lost Sales."
            elif status_pct.get('Over', 0) > 30:
                insight += f"Perhatian: Terdeteksi **{status_pct.get('Over', 0):.1f}%** kejadian Over Forecast yang membebani gudang."
            else:
                insight += "Performa supply chain cukup stabil dan seimbang."

            metrics = {
                'overall_accuracy': overall_accuracy,
                'mape': mape,
                'status_percentages': status_pct,
                'sku_accuracy': sku_acc,
                'detailed_data': df_merged,
                'period_months': [m.strftime('%b %Y') for m in last_3_months],
                'total_records': total,
                'insight': insight
            }
            
    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")
        
    return metrics

def calculate_monthly_metrics(df_forecast, df_po):
    if df_forecast.empty or df_po.empty: return pd.DataFrame()
    
    df = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
    monthly_stats = []
    
    for month in sorted(df['Month'].unique()):
        sub = df[df['Month'] == month]
        ratio = np.where(sub['Forecast_Qty']>0, (sub['PO_Qty']/sub['Forecast_Qty'])*100, 0)
        acc = 100 - np.mean(abs(ratio - 100))
        
        stats = pd.Series(np.select(
            [ratio < 80, (ratio >= 80) & (ratio <= 120), ratio > 120],
            ['Under', 'Accurate', 'Over'], default='Unknown'
        )).value_counts()
        
        monthly_stats.append({
            'Month_Formatted': month.strftime('%b %Y'),
            'Accuracy': acc,
            'Under': stats.get('Under', 0),
            'Accurate': stats.get('Accurate', 0),
            'Over': stats.get('Over', 0),
            'Total_SKUs': len(sub)
        })
        
    return pd.DataFrame(monthly_stats)

def create_sankey_tier(df_detailed):
    """Sankey Chart Robust Version"""
    if df_detailed is None or df_detailed.empty: return None
    try:
        # Penting: Drop data yang tidak punya Tier atau Status agar tidak crash
        df_clean = df_detailed.dropna(subset=['SKU_Tier', 'Accuracy_Status'])
        
        flow = df_clean.groupby(['SKU_Tier', 'Accuracy_Status']).size().reset_index(name='Count')
        
        all_nodes = list(pd.concat([flow['SKU_Tier'], flow['Accuracy_Status']]).unique())
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=all_nodes, color="#667eea"
            ),
            link=dict(
                source=flow['SKU_Tier'].map(node_indices),
                target=flow['Accuracy_Status'].map(node_indices),
                value=flow['Count']
            )
        )])
        
        fig.update_layout(title_text="<b>Tier Flow Analysis</b>", height=500, font_size=12)
        return fig
    except: return None

# --- ====================================================== ---
# ---                DASHBOARD INITIALIZATION                ---
# --- ====================================================== ---

client = init_gsheet_connection()
if not client: st.stop()

with st.spinner('🔄 Synchronizing Data...'):
    all_data = load_and_process_data(client)
    df_product = all_data.get('product', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())

# Run Calculation
metrics = calculate_forecast_accuracy_3months(df_forecast, df_po, df_product)
monthly_df = calculate_monthly_metrics(df_forecast, df_po)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.metric("Total SKU Active", len(all_data.get('product_active', [])))
    st.metric("Total Stock Qty", f"{df_stock['Stock_Qty'].sum():,.0f}" if not df_stock.empty else 0)

# --- HEADER & INSIGHT ---
# AI Insight Box (New Feature)
if metrics and 'insight' in metrics:
    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Intelligence Insight:</strong> {metrics['insight']}
    </div>
    """, unsafe_allow_html=True)

# KPI Cards
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    pct = metrics['status_percentages']
    with c1:
        st.markdown(f'<div class="status-indicator status-under">UNDER FORECAST<br><span style="font-size:2rem">{pct.get("Under",0):.1f}%</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="status-indicator status-accurate">ACCURATE<br><span style="font-size:2rem">{pct.get("Accurate",0):.1f}%</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="status-indicator status-over">OVER FORECAST<br><span style="font-size:2rem">{pct.get("Over",0):.1f}%</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-highlight"><span style="color:#666">OVERALL ACCURACY</span><br><span style="font-size:2rem; color:#667eea">{metrics["overall_accuracy"]:.1f}%</span></div>', unsafe_allow_html=True)

# --- TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Monthly Performance", "📊 Tier Analysis", "📦 Inventory Health", 
    "🔍 SKU Evaluation", "📈 Sales Analytics", "📋 Data Explorer"
])

# TAB 1: Monthly
with t1:
    if not monthly_df.empty:
        col1, col2 = st.columns([2,1])
        with col1:
            c = alt.Chart(monthly_df).mark_line(point=True).encode(
                x='Month_Formatted', y='Accuracy', tooltip=['Month_Formatted', 'Accuracy']
            ).properties(title="Monthly Accuracy Trend", height=350)
            st.altair_chart(c, use_container_width=True)
        with col2:
            st.dataframe(monthly_df, use_container_width=True, height=350)

# TAB 2: Tier (Sankey)
with t2:
    if metrics and 'detailed_data' in metrics:
        fig = create_sankey_tier(metrics['detailed_data'])
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.info("Tidak cukup data untuk Sankey Chart")

# TAB 3: Inventory (Calculated on the fly to save space)
with t3:
    if not df_stock.empty and not df_sales.empty:
        avg_sales = df_sales.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
        inv = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
        # Merge product info explicitly
        inv = pd.merge(inv, df_product[['SKU_ID', 'Product_Name', 'SKU_Tier']], on='SKU_ID', how='left')
        
        inv['Cover'] = np.where(inv['Sales_Qty']>0, inv['Stock_Qty']/inv['Sales_Qty'], 99)
        inv['Status'] = np.select([inv['Cover']<0.8, (inv['Cover']>=0.8)&(inv['Cover']<=1.5), inv['Cover']>1.5],
                                  ['Replenish', 'Ideal', 'High Stock'], default='Unknown')
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="inventory-card card-replenish">REPLENISH<br><span style="font-size:1.5rem">{len(inv[inv["Status"]=="Replenish"])} SKU</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="inventory-card card-ideal">IDEAL<br><span style="font-size:1.5rem">{len(inv[inv["Status"]=="Ideal"])} SKU</span></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="inventory-card card-high">HIGH STOCK<br><span style="font-size:1.5rem">{len(inv[inv["Status"]=="High Stock"])} SKU</span></div>', unsafe_allow_html=True)
        
        st.subheader("📉 High Stock Alert")
        st.dataframe(inv[inv['Status']=="High Stock"].sort_values('Cover', ascending=False), use_container_width=True)

# TAB 4: SKU Eval
with t4:
    if metrics:
        st.dataframe(metrics['sku_accuracy'].sort_values('SKU_Accuracy'), use_container_width=True)

# TAB 5: Sales
with t5:
    if not df_sales.empty:
        sales_trend = df_sales.groupby('Month')['Sales_Qty'].sum().reset_index()
        st.altair_chart(alt.Chart(sales_trend).mark_bar().encode(x='Month', y='Sales_Qty').properties(title="Total Sales Trend"), use_container_width=True)

# TAB 6: Explorer
with t6:
    opt = st.selectbox("Dataset", ["Product", "Sales", "Forecast", "PO", "Stock"])
    d_map = {"Product": df_product, "Sales": df_sales, "Forecast": df_forecast, "PO": df_po, "Stock": df_stock}
    st.dataframe(d_map[opt], use_container_width=True)

st.divider()
st.markdown("<center>Inventory Intelligence Pro v5.0 (Robust Edition)</center>", unsafe_allow_html=True)
