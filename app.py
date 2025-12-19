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
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro V8",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM (FLOATING & SOLID CARDS) ---
st.markdown("""
<style>
    /* Import Font Keren */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #5c6bc0; /* Warna ungu kebiruan seperti di gambar */
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sub-header-caption {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* MONTHLY CARDS (TOP ROW) */
    .month-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        /* Floating Effect */
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border-left: 6px solid #4CAF50; /* Green accent on left */
        transition: transform 0.3s ease;
        margin-bottom: 20px;
        height: 100%;
    }
    .month-card:hover { transform: translateY(-5px); }
    
    .month-title { font-size: 1.5rem; font-weight: 700; color: #333; margin-bottom: 10px; }
    
    .status-badge-container {
        display: flex; gap: 5px; justify-content: flex-end; margin-bottom: 15px;
    }
    .badge {
        padding: 5px 10px; border-radius: 8px; color: white; 
        font-size: 0.75rem; font-weight: bold; text-align: center; min-width: 60px;
    }
    .badge-red { background-color: #FF5252; }
    .badge-green { background-color: #4CAF50; }
    .badge-orange { background-color: #FFA726; }
    
    .month-metric-val { font-size: 1.8rem; font-weight: 800; color: #333; }
    .month-metric-lbl { font-size: 0.8rem; color: #666; }

    /* SUMMARY CARDS (BOTTOM ROW - SOLID COLORS) */
    .summary-card {
        border-radius: 12px;
        padding: 25px 15px;
        text-align: center;
        color: white;
        /* Floating Effect Stronger */
        box-shadow: 0 14px 28px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .summary-card:hover { box-shadow: 0 19px 38px rgba(0,0,0,0.20), 0 15px 12px rgba(0,0,0,0.12); }

    /* Warna Solid sesuai gambar */
    .bg-solid-red { background: linear-gradient(135deg, #FF5252 0%, #F44336 100%); }
    .bg-solid-green { background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%); }
    .bg-solid-orange { background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%); }
    
    /* Kartu Average (White style) */
    .bg-solid-white { 
        background: white; 
        color: #333; 
        border-top: 6px solid #5c6bc0;
    }

    .sum-title { font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 10px;}
    .sum-value { font-size: 3rem; font-weight: 800; line-height: 1; margin-bottom: 5px; }
    .sum-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.9; }
    .text-blue { color: #5c6bc0; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; margin-top: 20px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4; border-radius: 8px 8px 0 0; border: none; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: white; color: #5c6bc0; border-top: 3px solid #5c6bc0;
    }
</style>
""", unsafe_allow_html=True)

# --- ICON HEADER ---
st.markdown("""
<div style="text-align: center; font-size: 4rem; margin-bottom: -20px;">🟦</div>
<h1 class="main-header">INVENTORY INTELLIGENCE DASHBOARD</h1>
<div class="sub-header-caption">🚀 Professional Inventory Control & Demand Planning | Real-time Analytics | Updated: 19 December 2025</div>
""", unsafe_allow_html=True)

# --- ====================================================== ---
# ---             1. CORE ENGINE (ROBUST DATA LOADING)       ---
# --- ====================================================== ---

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
    except: return datetime.now()

@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(_client):
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_p = pd.DataFrame(ws.get_all_records())
        df_p.columns = [c.strip().replace(' ', '_') for c in df_p.columns]
        if 'Status' not in df_p.columns: df_p['Status'] = 'Active'
        df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_active['SKU_ID'].tolist()

        def robust_melt(sheet_name, val_col):
            ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df_temp = pd.DataFrame(ws_temp.get_all_records())
            df_temp.columns = [c.strip() for c in df_temp.columns]
            m_cols = [c for c in df_temp.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            if 'SKU_ID' not in df_temp.columns: return pd.DataFrame()
            
            df_long = df_temp[['SKU_ID'] + m_cols].melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='Month_Label', value_name=val_col)
            df_long[val_col] = pd.to_numeric(df_long[val_col], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            return df_long[df_long['SKU_ID'].isin(active_ids)]

        data['sales'] = robust_melt("Sales", "Sales_Qty")
        data['forecast'] = robust_melt("Rofo", "Forecast_Qty")
        data['po'] = robust_melt("PO", "PO_Qty")
        
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_s = pd.DataFrame(ws_s.get_all_records())
        df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
        s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_s.columns), None)
        if s_col and 'SKU_ID' in df_s.columns:
            df_stock = df_s[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
            df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
            data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_ids)].groupby('SKU_ID').max().reset_index()
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
            
        data['product'] = df_p
        data['product_active'] = df_active
        return data
    except Exception as e:
        st.error(f"Error: {e}"); return {}

# --- ====================================================== ---
# ---             2. ANALYTICS ENGINE                        ---
# --- ====================================================== ---

def calculate_monthly_performance(df_forecast, df_po, df_product):
    if df_forecast.empty or df_po.empty: return {}
    
    df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
    if not df_product.empty:
        meta = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
        df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')
    
    df_merged['Ratio'] = np.where(df_merged['Forecast_Qty']>0, (df_merged['PO_Qty']/df_merged['Forecast_Qty'])*100, 0)
    df_merged['Status'] = np.select(
        [df_merged['Ratio'] < 80, (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), df_merged['Ratio'] > 120],
        ['Under', 'Accurate', 'Over'], default='Unknown'
    )
    df_merged['APE'] = abs(df_merged['Ratio'] - 100)
    
    monthly_stats = {}
    for month in sorted(df_merged['Month'].unique()):
        month_data = df_merged[df_merged['Month'] == month].copy()
        monthly_stats[month] = {
            'accuracy': 100 - month_data['APE'].mean(),
            'counts': month_data['Status'].value_counts().to_dict(),
            'total': len(month_data),
            'data': month_data
        }
    return monthly_stats

def calculate_inventory_metrics(df_stock, df_sales, df_product):
    if df_stock.empty: return pd.DataFrame()
    
    if not df_sales.empty:
        months = sorted(df_sales['Month'].unique())[-3:]
        sales_3m = df_sales[df_sales['Month'].isin(months)]
        avg_sales = sales_3m.groupby('SKU_ID')['Sales_Qty'].mean().reset_index(name='Avg_Sales_3M')
    else:
        avg_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Sales_3M'])
        
    inv = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
    inv['Avg_Sales_3M'] = inv['Avg_Sales_3M'].fillna(0)
    
    if not df_product.empty:
        inv = pd.merge(inv, df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']], on='SKU_ID', how='left')
        
    inv['Cover_Months'] = np.where(inv['Avg_Sales_3M']>0, inv['Stock_Qty']/inv['Avg_Sales_3M'], 999)
    inv['Status'] = np.select(
        [inv['Cover_Months'] < 0.8, (inv['Cover_Months'] >= 0.8) & (inv['Cover_Months'] <= 1.5), inv['Cover_Months'] > 1.5],
        ['Need Replenishment', 'Ideal/Healthy', 'High Stock'], default='Unknown'
    )
    return inv

def create_tier_chart(df_data):
    if df_data.empty: return None
    df_clean = df_data.dropna(subset=['SKU_Tier'])
    agg = df_clean.groupby(['SKU_Tier', 'Status']).size().reset_index(name='Count')
    fig = px.bar(agg, x="SKU_Tier", y="Count", color="Status", 
                 title="Accuracy Distribution by Tier",
                 color_discrete_map={'Under': '#ef5350', 'Accurate': '#66bb6a', 'Over': '#ffa726'},
                 template="plotly_white")
    fig.update_layout(height=400)
    return fig

# --- ====================================================== ---
# ---                3. MAIN DASHBOARD UI                    ---
# --- ====================================================== ---

client = init_gsheet_connection()
if not client: st.stop()

with st.spinner('🔄 Loading Data...'):
    all_data = load_and_process_data(client)
    
monthly_perf = calculate_monthly_performance(all_data['forecast'], all_data['po'], all_data['product'])
inv_df = calculate_inventory_metrics(all_data['stock'], all_data['sales'], all_data['product'])

# --- HEADER SECTION: PERFORMANCE 3 BULAN TERAKHIR ---
st.markdown("### 📊 Performance Accuracy - 3 Bulan Terakhir")

if monthly_perf:
    last_3_months = sorted(monthly_perf.keys())[-3:]
    cols = st.columns(len(last_3_months))
    
    for idx, month in enumerate(last_3_months):
        data = monthly_perf[month]
        counts = data['counts']
        
        with cols[idx]:
            # HTML Card Floating (White)
            st.markdown(f"""
            <div class="month-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="month-title">{month.strftime('%b %Y')}</div>
                    <div class="status-badge-container">
                        <div class="badge badge-red">Under: {counts.get('Under',0)}</div>
                        <div class="badge badge-green">OK: {counts.get('Accurate',0)}</div>
                        <div class="badge badge-orange">Over: {counts.get('Over',0)}</div>
                    </div>
                </div>
                <div style="margin-top:10px;">
                    <div class="month-metric-lbl">Overall Accuracy:</div>
                    <div class="month-metric-val">{data['accuracy']:.1f}%</div>
                    <div style="text-align:right; font-size:0.8rem; color:#888;">Total SKUs: {data['total']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- SUMMARY TOTAL (SOLID CARDS - BULAN TERAKHIR ONLY) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ambil Data Bulan Terakhir
    last_month = last_3_months[-1]
    last_month_data = monthly_perf[last_month]['data']
    
    # Hitung Statistik Bulan Terakhir
    # 1. Total Under
    df_under = last_month_data[last_month_data['Status'] == 'Under']
    under_count = len(df_under)
    under_qty = df_under['Forecast_Qty'].sum() # Pake Forecast Qty sebagai acuan besaran
    under_pct = (under_count / len(last_month_data) * 100) if len(last_month_data) > 0 else 0
    
    # 2. Total Accurate
    df_acc = last_month_data[last_month_data['Status'] == 'Accurate']
    acc_count = len(df_acc)
    acc_qty = df_acc['Forecast_Qty'].sum()
    acc_pct = (acc_count / len(last_month_data) * 100) if len(last_month_data) > 0 else 0
    
    # 3. Total Over
    df_over = last_month_data[last_month_data['Status'] == 'Over']
    over_count = len(df_over)
    over_qty = df_over['Forecast_Qty'].sum()
    over_pct = (over_count / len(last_month_data) * 100) if len(last_month_data) > 0 else 0
    
    # 4. Average
    avg_acc = monthly_perf[last_month]['accuracy']
    
    # Render Solid Cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="summary-card bg-solid-red">
            <div class="sum-title">TOTAL UNDER</div>
            <div class="sum-value">{under_count}</div>
            <div class="sum-sub">{under_pct:.1f}% of total</div>
            <div style="margin-top:10px; font-size:0.8rem; opacity:0.8;">Total Qty: {under_qty:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="summary-card bg-solid-green">
            <div class="sum-title">TOTAL ACCURATE</div>
            <div class="sum-value">{acc_count}</div>
            <div class="sum-sub">{acc_pct:.1f}% of total</div>
            <div style="margin-top:10px; font-size:0.8rem; opacity:0.8;">Total Qty: {acc_qty:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="summary-card bg-solid-orange">
            <div class="sum-title">TOTAL OVER</div>
            <div class="sum-value">{over_count}</div>
            <div class="sum-sub">{over_pct:.1f}% of total</div>
            <div style="margin-top:10px; font-size:0.8rem; opacity:0.8;">Total Qty: {over_qty:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="summary-card bg-solid-white">
            <div class="sum-title" style="color:#666;">AVERAGE ACCURACY</div>
            <div class="sum-value text-blue">{avg_acc:.1f}%</div>
            <div class="sum-sub" style="color:#888;">{len(last_month_data)} Total Records</div>
            <div style="margin-top:10px; font-size:0.8rem; color:#888;">Period: {last_month.strftime('%b %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Data belum tersedia untuk kalkulasi.")

st.divider()

# --- TABS ANALYSIS ---
tab_eval, tab_tier, tab_inv, tab_sales, tab_raw = st.tabs([
    "📋 Evaluasi Rofo", "📊 Tier Analysis", "📦 Inventory", "🔍 Sales vs Rofo", "📁 Raw Data"
])

# --- TAB: EVALUASI ROFO (DETAIL TABLE) ---
with tab_eval:
    if monthly_perf:
        st.subheader(f"Detail Evaluasi SKU - {last_month.strftime('%b %Y')}")
        st.info("Fokus pada perbaikan SKU dengan status Under dan Over.")
        
        # Merge Inventory Info
        eval_df = pd.merge(last_month_data, inv_df[['SKU_ID', 'Stock_Qty', 'Avg_Sales_3M']], on='SKU_ID', how='left')
        
        # Kolom Final
        cols_final = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status', 
                      'Forecast_Qty', 'PO_Qty', 'Ratio', 'Stock_Qty', 'Avg_Sales_3M']
        # Filter avail columns
        cols_final = [c for c in cols_final if c in eval_df.columns]
        
        df_show = eval_df[cols_final].rename(columns={'Ratio': 'Achv %', 'Stock_Qty': 'Stock', 'Avg_Sales_3M': 'Avg Sales'})
        
        t1, t2 = st.tabs(["📉 SKU Under Forecast", "📈 SKU Over Forecast"])
        
        with t1:
            df_u = df_show[df_show['Status']=='Under'].sort_values('Achv %')
            st.dataframe(df_u, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)
            
        with t2:
            df_o = df_show[df_show['Status']=='Over'].sort_values('Achv %', ascending=False)
            st.dataframe(df_o, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)

# --- TAB: TIER ANALYSIS ---
with tab_tier:
    if monthly_perf:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = create_tier_chart(last_month_data)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'SKU_Tier' in last_month_data.columns:
                ts = last_month_data.groupby(['SKU_Tier', 'Status']).size().unstack(fill_value=0)
                ts['Total'] = ts.sum(axis=1)
                ts['Acc %'] = (ts.get('Accurate', 0) / ts['Total'] * 100).round(1)
                st.dataframe(ts[['Accurate', 'Under', 'Over', 'Acc %']].sort_values('Acc %', ascending=False), use_container_width=True)

# --- TAB: INVENTORY ---
with tab_inv:
    if not inv_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Replenish Needed", len(inv_df[inv_df['Status']=='Need Replenishment']))
        with c2: st.metric("Healthy Stock", len(inv_df[inv_df['Status']=='Ideal/Healthy']))
        with c3: st.metric("Overstock", len(inv_df[inv_df['Status']=='High Stock']))
        
        st.write("### Detail Stock Status")
        fil = st.multiselect("Filter", inv_df['Status'].unique(), default=['Need Replenishment', 'High Stock'])
        st.dataframe(inv_df[inv_df['Status'].isin(fil)], use_container_width=True)

# --- TAB: SALES ---
with tab_sales:
    if 'sales' in all_data and 'forecast' in all_data:
        common = sorted(set(all_data['sales']['Month']) & set(all_data['forecast']['Month']))
        if common:
            lm = common[-1]
            s = all_data['sales'][all_data['sales']['Month']==lm]
            f = all_data['forecast'][all_data['forecast']['Month']==lm]
            comp = pd.merge(s, f, on='SKU_ID', suffixes=('_Sales', '_Fc'))
            
            if not all_data['product'].empty:
                comp = pd.merge(comp, all_data['product'][['SKU_ID', 'Product_Name']], on='SKU_ID', how='left')
                
            comp['Dev %'] = (comp['Sales_Qty'] - comp['Forecast_Qty']) / comp['Forecast_Qty'] * 100
            comp['Abs Dev'] = abs(comp['Dev %'])
            
            st.write(f"### High Deviation SKU (>30%) - {lm.strftime('%b %Y')}")
            st.dataframe(
                comp[comp['Abs Dev']>30].sort_values('Abs Dev', ascending=False),
                column_config={"Dev %": st.column_config.NumberColumn(format="%.1f%%")},
                use_container_width=True
            )

# --- TAB: RAW DATA ---
with tab_raw:
    opt = st.selectbox("Dataset", ["Sales", "Forecast", "PO", "Stock"])
    d_map = {"Sales": all_data.get('sales'), "Forecast": all_data.get('forecast'), "PO": all_data.get('po'), "Stock": all_data.get('stock')}
    st.dataframe(d_map[opt], use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.info("V8.0 Visual Masterpiece")
