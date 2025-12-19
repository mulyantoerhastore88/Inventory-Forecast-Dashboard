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
    page_title="Inventory Intelligence Pro V7",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Premium (Glassmorphism & Cards) ---
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #203a43;
    }

    /* Month Performance Cards (V6 Style) */
    .month-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-top: 5px solid #2c5364;
        transition: transform 0.3s ease;
    }
    .month-card:hover { transform: translateY(-5px); }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        text-align: center;
        gap: 5px;
    }
    .metric-item {
        flex: 1;
        padding: 8px 4px;
        border-radius: 8px;
    }
    .bg-under { background-color: #ffebee; color: #c62828; font-weight: bold;}
    .bg-accurate { background-color: #e8f5e9; color: #2e7d32; font-weight: bold;}
    .bg-over { background-color: #fff3e0; color: #ef6c00; font-weight: bold;}
    
    .big-number { font-size: 1.6rem; font-weight: 800; }
    .small-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #666; }

    /* Inventory Cards */
    .inv-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .inv-replenish { border-bottom: 4px solid #ff4757; }
    .inv-ideal { border-bottom: 4px solid #2ed573; }
    .inv-high { border-bottom: 4px solid #ffa502; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f2f6;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        color: #2c5364;
        border-top: 3px solid #2c5364;
        box-shadow: 0 -5px 10px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL ---
st.markdown('<h1 class="main-header">💎 INVENTORY INTELLIGENCE V7.0</h1>', unsafe_allow_html=True)
st.caption(f"🚀 AI-Powered Demand Planning | Robust Engine | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

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
    """
    DATA LOADER "BADAK" (Robust):
    Hanya mengambil SKU_ID + Angka dari sheet transaksi.
    Metadata (Nama, Brand, Tier) diambil terpusat dari Product Master.
    """
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        # 1. Product Master (Metadata)
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_p = pd.DataFrame(ws.get_all_records())
        df_p.columns = [c.strip().replace(' ', '_') for c in df_p.columns]
        if 'Status' not in df_p.columns: df_p['Status'] = 'Active'
        df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_active['SKU_ID'].tolist()

        # Helper Melt Function
        def robust_melt(sheet_name, val_col):
            ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df_temp = pd.DataFrame(ws_temp.get_all_records())
            df_temp.columns = [c.strip() for c in df_temp.columns]
            m_cols = [c for c in df_temp.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if 'SKU_ID' not in df_temp.columns or not m_cols: return pd.DataFrame()
            
            # Select ONLY SKU_ID + Months (Ignore Product Name here to prevent KeyErrors)
            df_long = df_temp[['SKU_ID'] + m_cols].melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='Month_Label', value_name=val_col)
            df_long[val_col] = pd.to_numeric(df_long[val_col], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            return df_long[df_long['SKU_ID'].isin(active_ids)]

        data['sales'] = robust_melt("Sales", "Sales_Qty")
        data['forecast'] = robust_melt("Rofo", "Forecast_Qty")
        data['po'] = robust_melt("PO", "PO_Qty")
        
        # Stock Data
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_s = pd.DataFrame(ws_s.get_all_records())
        df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
        s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_s.columns), None)
        
        if s_col and 'SKU_ID' in df_s.columns:
            df_stock = df_s[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
            df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
            # Group by just in case duplicate SKU rows
            df_stock = df_stock.groupby('SKU_ID')['Stock_Qty'].max().reset_index()
            data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_ids)]
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
            
        data['product'] = df_p
        data['product_active'] = df_active
        return data
    except Exception as e:
        st.error(f"Error Loading: {e}"); return {}

# --- ====================================================== ---
# ---             2. ANALYTICS ENGINE (CALCULATIONS)         ---
# --- ====================================================== ---

def calculate_monthly_performance(df_forecast, df_po, df_product):
    if df_forecast.empty or df_po.empty: return {}
    
    # Merge Forecast & PO
    df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
    
    # Join Metadata (Product Name, Tier, Brand) from Master
    if not df_product.empty:
        meta = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
        df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')
    
    # Calculate Metrics
    df_merged['Ratio'] = np.where(df_merged['Forecast_Qty']>0, (df_merged['PO_Qty']/df_merged['Forecast_Qty'])*100, 0)
    
    conditions = [
        df_merged['Ratio'] < 80, 
        (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), 
        df_merged['Ratio'] > 120
    ]
    df_merged['Status'] = np.select(conditions, ['Under', 'Accurate', 'Over'], default='Unknown')
    df_merged['APE'] = abs(df_merged['Ratio'] - 100)
    
    # Group by Month
    monthly_stats = {}
    for month in sorted(df_merged['Month'].unique()):
        month_data = df_merged[df_merged['Month'] == month].copy()
        
        # Brand Performance specific to this month
        brand_perf = pd.DataFrame()
        if 'Brand' in month_data.columns:
            brand_perf = month_data.groupby('Brand').agg({
                'SKU_ID': 'count',
                'Forecast_Qty': 'sum',
                'PO_Qty': 'sum',
                'APE': 'mean' # Inverse of Accuracy
            }).reset_index()
            brand_perf['Accuracy'] = 100 - brand_perf['APE']
        
        monthly_stats[month] = {
            'accuracy': 100 - month_data['APE'].mean(),
            'counts': month_data['Status'].value_counts().to_dict(),
            'total': len(month_data),
            'data': month_data,
            'brand_perf': brand_perf
        }
    return monthly_stats

def calculate_inventory_metrics(df_stock, df_sales, df_product):
    if df_stock.empty: return pd.DataFrame()
    
    # Calculate 3-Month Average Sales (Backward Looking)
    if not df_sales.empty:
        months = sorted(df_sales['Month'].unique())[-3:]
        sales_3m = df_sales[df_sales['Month'].isin(months)]
        avg_sales = sales_3m.groupby('SKU_ID')['Sales_Qty'].mean().reset_index(name='Avg_Sales_3M')
    else:
        avg_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Sales_3M'])
        
    # Merge Stock & Sales
    inv = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
    inv['Avg_Sales_3M'] = inv['Avg_Sales_3M'].fillna(0)
    
    # Merge Metadata
    if not df_product.empty:
        inv = pd.merge(inv, df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']], on='SKU_ID', how='left')
        
    # Calculate Cover
    inv['Cover_Months'] = np.where(inv['Avg_Sales_3M']>0, inv['Stock_Qty']/inv['Avg_Sales_3M'], 999)
    
    conditions = [
        inv['Cover_Months'] < 0.8, 
        (inv['Cover_Months'] >= 0.8) & (inv['Cover_Months'] <= 1.5), 
        inv['Cover_Months'] > 1.5
    ]
    inv['Status'] = np.select(conditions, ['Need Replenishment', 'Ideal/Healthy', 'High Stock'], default='Unknown')
    return inv

def create_tier_chart(df_data):
    """Stacked Bar Chart Tier vs Status"""
    if df_data.empty: return None
    # Drop rows without Tier
    df_clean = df_data.dropna(subset=['SKU_Tier'])
    agg = df_clean.groupby(['SKU_Tier', 'Status']).size().reset_index(name='Count')
    
    fig = px.bar(agg, x="SKU_Tier", y="Count", color="Status", 
                 title="Accuracy Distribution by Tier",
                 color_discrete_map={'Under': '#ef5350', 'Accurate': '#66bb6a', 'Over': '#ffa726'},
                 template="plotly_white")
    return fig

# --- ====================================================== ---
# ---                3. MAIN DASHBOARD UI                    ---
# --- ====================================================== ---

client = init_gsheet_connection()
if not client: st.stop()

with st.spinner('🔄 Synchronizing Intelligence Engine...'):
    all_data = load_and_process_data(client)
    
monthly_perf = calculate_monthly_performance(all_data['forecast'], all_data['po'], all_data['product'])
inv_df = calculate_inventory_metrics(all_data['stock'], all_data['sales'], all_data['product'])

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance Dashboard", 
    "📊 Tier & Brand Analysis", 
    "📦 Inventory Analysis", 
    "🔍 Sales Analysis", 
    "📋 Data Explorer"
])

# --- TAB 1: DASHBOARD UTAMA (Requested: Month Cards & Evaluasi Rofo) ---
with tab1:
    if monthly_perf:
        # A. MONTH CARDS (3 Bulan Terakhir)
        st.subheader("📅 Forecast Performance - 3 Bulan Terakhir")
        last_3_months = sorted(monthly_perf.keys())[-3:]
        cols = st.columns(len(last_3_months))
        
        for idx, month in enumerate(last_3_months):
            data = monthly_perf[month]
            counts = data['counts']
            
            with cols[idx]:
                st.markdown(f"""
                <div class="month-card">
                    <div style="text-align:center; margin-bottom:10px;">
                        <h3 style="margin:0; color:#333;">{month.strftime('%b %Y')}</h3>
                        <div style="font-size:2.2rem; font-weight:900; color:#2c5364;">{data['accuracy']:.1f}%</div>
                        <div style="color:#666; font-size:0.8rem;">Overall Accuracy</div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-item bg-under">
                            <div class="big-number">{counts.get('Under', 0)}</div>
                            <div class="small-label">Under</div>
                        </div>
                        <div class="metric-item bg-accurate">
                            <div class="big-number">{counts.get('Accurate', 0)}</div>
                            <div class="small-label">Akurat</div>
                        </div>
                        <div class="metric-item bg-over">
                            <div class="big-number">{counts.get('Over', 0)}</div>
                            <div class="small-label">Over</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # B. EVALUASI ROFO ONWARDS (DETAIL TABLE)
        st.divider()
        last_month = last_3_months[-1]
        last_month_name = last_month.strftime('%b %Y')
        last_month_df = monthly_perf[last_month]['data']
        
        st.subheader(f"📋 Evaluasi Rofo - {last_month_name}")
        st.info(f"Detail SKU yang **Under** dan **Over** di bulan {last_month_name}. Gunakan data ini untuk penyesuaian forecast bulan depan.")
        
        # Join Inventory Info
        eval_df = pd.merge(last_month_df, inv_df[['SKU_ID', 'Stock_Qty', 'Avg_Sales_3M']], on='SKU_ID', how='left')
        eval_df = eval_df[eval_df['Status'].isin(['Under', 'Over'])] # Filter only deviations
        
        # Display Cols
        d_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status', 'Forecast_Qty', 'PO_Qty', 'Ratio', 'Stock_Qty', 'Avg_Sales_3M']
        # Filter available columns
        d_cols = [c for c in d_cols if c in eval_df.columns]
        
        final_view = eval_df[d_cols].rename(columns={'Ratio':'Achv %', 'Stock_Qty':'Stock', 'Avg_Sales_3M':'Avg Sales'})
        
        t_under, t_over = st.tabs(["📉 Detail UNDER", "📈 Detail OVER"])
        
        with t_under:
            df_u = final_view[final_view['Status']=='Under'].sort_values('Achv %')
            st.dataframe(df_u, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)
            
        with t_over:
            df_o = final_view[final_view['Status']=='Over'].sort_values('Achv %', ascending=False)
            st.dataframe(df_o, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)

# --- TAB 2: TIER & BRAND ANALYSIS (Stacked Bar + Brand Table) ---
with tab2:
    if monthly_perf:
        last_month = sorted(monthly_perf.keys())[-1]
        data_last = monthly_perf[last_month]['data']
        brand_last = monthly_perf[last_month]['brand_perf']
        
        st.subheader(f"📊 Analysis by Tier & Brand ({last_month.strftime('%b %Y')})")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Accuracy by Tier")
            fig_tier = create_tier_chart(data_last)
            if fig_tier: st.plotly_chart(fig_tier, use_container_width=True)
            
        with col2:
            st.markdown("##### Performance by Brand")
            if not brand_last.empty:
                st.dataframe(
                    brand_last.sort_values('Accuracy', ascending=False),
                    column_config={
                        "Accuracy": st.column_config.ProgressColumn("Accuracy %", format="%.1f%%", min_value=0, max_value=100),
                        "Forecast_Qty": st.column_config.NumberColumn("Total Rofo"),
                        "PO_Qty": st.column_config.NumberColumn("Total PO")
                    },
                    use_container_width=True, height=350
                )

# --- TAB 3: INVENTORY ANALYSIS (Cards + Filter Table) ---
with tab3:
    st.subheader("📦 Inventory Health Status")
    
    if not inv_df.empty:
        c1, c2, c3 = st.columns(3)
        n_rep = len(inv_df[inv_df['Status']=='Need Replenishment'])
        n_ideal = len(inv_df[inv_df['Status']=='Ideal/Healthy'])
        n_high = len(inv_df[inv_df['Status']=='High Stock'])
        
        with c1: st.markdown(f'<div class="inv-card inv-replenish"><h3>Need Replenishment</h3><h1 style="color:#ff4757">{n_rep}</h1><p>Cover < 0.8 Mo</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="inv-card inv-ideal"><h3>Ideal Inventory</h3><h1 style="color:#2ed573">{n_ideal}</h1><p>0.8 - 1.5 Mo</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="inv-card inv-high"><h3>High Stock</h3><h1 style="color:#ffa502">{n_high}</h1><p>Cover > 1.5 Mo</p></div>', unsafe_allow_html=True)
        
        st.divider()
        st.write("### 🔍 Detail Inventory SKU")
        
        fil_stat = st.multiselect("Filter Status", inv_df['Status'].unique(), default=['Need Replenishment', 'High Stock'])
        df_show = inv_df[inv_df['Status'].isin(fil_stat)].copy()
        
        st.dataframe(
            df_show.sort_values('Cover_Months', ascending=False),
            column_config={
                "Cover_Months": st.column_config.NumberColumn("Cover (Mo)", format="%.1f"),
                "Stock_Qty": st.column_config.NumberColumn("Stock", format="%d"),
                "Avg_Sales_3M": st.column_config.NumberColumn("Avg Sales", format="%d")
            },
            use_container_width=True
        )

# --- TAB 4: SALES ANALYSIS (Overview & High Deviation) ---
with tab4:
    st.subheader("📈 Sales Analysis vs Plan")
    
    if 'sales' in all_data and 'forecast' in all_data and 'po' in all_data:
        # Logic: Compare Last Month Sales vs Rofo vs PO
        common_m = sorted(set(all_data['sales']['Month']) & set(all_data['forecast']['Month']))
        
        if common_m:
            last_m = common_m[-1]
            st.caption(f"Analysis Period: {last_m.strftime('%B %Y')}")
            
            s_df = all_data['sales'][all_data['sales']['Month']==last_m]
            f_df = all_data['forecast'][all_data['forecast']['Month']==last_m]
            p_df = all_data['po'][all_data['po']['Month']==last_m]
            
            # Merge 3 datasets
            comp = pd.merge(s_df, f_df, on='SKU_ID', suffixes=('_Sales', '_Fc'))
            comp = pd.merge(comp, p_df, on='SKU_ID', how='left') # PO might be missing
            
            # Add Metadata
            if not all_data['product'].empty:
                comp = pd.merge(comp, all_data['product'][['SKU_ID','Product_Name','Brand','SKU_Tier']], on='SKU_ID', how='left')
            
            # KPI
            tot_sales = comp['Sales_Qty'].sum()
            tot_fc = comp['Forecast_Qty'].sum()
            tot_po = comp['PO_Qty'].sum()
            
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Total Sales", f"{tot_sales:,.0f}")
            with k2: st.metric("Total Rofo", f"{tot_fc:,.0f}", f"{(tot_sales-tot_fc)/tot_fc*100:.1f}% vs Sales")
            with k3: st.metric("Total PO", f"{tot_po:,.0f}", f"{(tot_po-tot_fc)/tot_fc*100:.1f}% vs Rofo")
            
            st.divider()
            
            # High Deviation Logic
            comp['Dev_Rofo'] = np.where(comp['Forecast_Qty']>0, (comp['Sales_Qty']-comp['Forecast_Qty'])/comp['Forecast_Qty']*100, 0)
            comp['Abs_Dev'] = abs(comp['Dev_Rofo'])
            
            st.subheader("⚠️ High Deviation SKUs (>30% vs Rofo)")
            high_dev = comp[comp['Abs_Dev'] > 30].sort_values('Abs_Dev', ascending=False)
            
            st.dataframe(
                high_dev[['SKU_ID', 'Product_Name', 'Brand', 'Sales_Qty', 'Forecast_Qty', 'PO_Qty', 'Dev_Rofo']],
                column_config={
                    "Dev_Rofo": st.column_config.NumberColumn("Dev %", format="%.1f%%"),
                    "Sales_Qty": st.column_config.NumberColumn("Sales", format="%d"),
                    "Forecast_Qty": st.column_config.NumberColumn("Rofo", format="%d")
                },
                use_container_width=True
            )
        else:
            st.warning("Data Sales & Forecast tidak memiliki bulan yang sama.")

# --- TAB 5: EXPLORER ---
with tab5:
    st.subheader("📁 Raw Data Explorer")
    ds = st.selectbox("Select Dataset", ["Sales", "Forecast", "PO", "Stock", "Product"])
    d_map = {
        "Sales": all_data.get('sales'), "Forecast": all_data.get('forecast'),
        "PO": all_data.get('po'), "Stock": all_data.get('stock'),
        "Product": all_data.get('product')
    }
    st.dataframe(d_map[ds], use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.info("V7.0 Ultimate | SKU_ID Centric Engine")
