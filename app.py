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
    page_title="Inventory Intelligence Pro V9.0",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM (FLOATING & SOLID CARDS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .main-header {
        font-size: 2.5rem; font-weight: 800; color: #5c6bc0;
        text-align: center; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .sub-header-caption { text-align: center; color: #888; font-size: 0.9rem; margin-bottom: 2rem; }

    /* MONTH CARD */
    .month-card {
        background: white; border-radius: 15px; padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); border-left: 6px solid #5c6bc0;
        transition: transform 0.3s ease; margin-bottom: 20px; height: 100%;
    }
    .month-card:hover { transform: translateY(-5px); }
    .month-title { font-size: 1.4rem; font-weight: 700; color: #333; margin-bottom: 10px; }
    
    .status-badge-container { display: flex; gap: 4px; justify-content: center; margin-bottom: 15px; flex-wrap: wrap; }
    .badge { padding: 4px 6px; border-radius: 6px; color: white; font-size: 0.65rem; font-weight: bold; min-width: 40px; text-align: center; }
    .badge-red { background-color: #ef5350; }
    .badge-green { background-color: #66bb6a; }
    .badge-orange { background-color: #ffa726; }
    .badge-gray { background-color: #78909c; } /* Warna No Rofo */
    
    .month-metric-val { font-size: 1.8rem; font-weight: 800; color: #2c3e50; }
    .month-metric-lbl { font-size: 0.8rem; color: #7f8c8d; }

    /* SUMMARY CARDS */
    .summary-card {
        border-radius: 15px; padding: 25px 15px; text-align: center;
        color: white; box-shadow: 0 14px 28px rgba(0,0,0,0.10); margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .summary-card:hover { transform: scale(1.02); }

    .bg-solid-red { background: linear-gradient(135deg, #FF5252 0%, #D32F2F 100%); }
    .bg-solid-green { background: linear-gradient(135deg, #66BB6A 0%, #2E7D32 100%); }
    .bg-solid-orange { background: linear-gradient(135deg, #FFA726 0%, #EF6C00 100%); }
    .bg-solid-gray { background: linear-gradient(135deg, #90A4AE 0%, #607D8B 100%); } /* No Rofo */
    .bg-solid-white { background: white; color: #333; border-top: 5px solid #5c6bc0; }

    .sum-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 10px; }
    .sum-value { font-size: 2.5rem; font-weight: 800; line-height: 1; margin-bottom: 5px; }
    .sum-pct { font-size: 0.9rem; font-weight: 600; margin-bottom: 10px; opacity: 0.9; }
    .sum-footer { border-top: 1px solid rgba(255,255,255,0.3); padding-top: 10px; font-size: 0.75rem; font-weight: 500; opacity: 0.9; }
    
    .bg-solid-white .sum-title { color: #666; }
    .bg-solid-white .sum-value { color: #5c6bc0; }
    .bg-solid-white .sum-pct { color: #333; }
    .bg-solid-white .sum-footer { border-top: 1px solid #eee; color: #666; }

    .stTabs [data-baseweb="tab-list"] { gap: 10px; margin-top: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: #f8f9fa; border-radius: 8px 8px 0 0; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: white; color: #5c6bc0; border-top: 3px solid #5c6bc0; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; font-size: 3.5rem; margin-bottom: -15px;">🟦</div>
<h1 class="main-header">INVENTORY INTELLIGENCE DASHBOARD</h1>
<div class="sub-header-caption">🚀 Professional Inventory Control & Demand Planning | Real-time Analytics</div>
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
        
        # FIX: Paksa SKU_ID jadi String
        if 'SKU_ID' in df_p.columns:
            df_p['SKU_ID'] = df_p['SKU_ID'].astype(str).str.strip()
            
        if 'Status' not in df_p.columns: df_p['Status'] = 'Active'
        df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_active['SKU_ID'].tolist()

        def robust_melt(sheet_name, val_col):
            ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df_temp = pd.DataFrame(ws_temp.get_all_records())
            df_temp.columns = [c.strip() for c in df_temp.columns]
            
            # FIX: Paksa SKU_ID jadi String
            if 'SKU_ID' in df_temp.columns:
                df_temp['SKU_ID'] = df_temp['SKU_ID'].astype(str).str.strip()
            else:
                return pd.DataFrame()
                
            m_cols = [c for c in df_temp.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            df_long = df_temp[['SKU_ID'] + m_cols].melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='Month_Label', value_name=val_col)
            df_long[val_col] = pd.to_numeric(df_long[val_col], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            # FIX: Force to datetime
            df_long['Month'] = pd.to_datetime(df_long['Month'])
            
            return df_long[df_long['SKU_ID'].isin(active_ids)]

        data['sales'] = robust_melt("Sales", "Sales_Qty")
        data['forecast'] = robust_melt("Rofo", "Forecast_Qty")
        data['po'] = robust_melt("PO", "PO_Qty")
        
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_s = pd.DataFrame(ws_s.get_all_records())
        df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
        
        if 'SKU_ID' in df_s.columns:
            df_s['SKU_ID'] = df_s['SKU_ID'].astype(str).str.strip()
            
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
        st.error(f"Error Loading Data: {e}"); return {}

# --- ====================================================== ---
# ---             2. ANALYTICS ENGINE (UPDATED V9.0)         ---
# --- ====================================================== ---

def calculate_monthly_performance(df_forecast, df_po, df_product):
    if df_forecast.empty or df_po.empty: return {}
    
    # Ensure Datetime
    df_forecast['Month'] = pd.to_datetime(df_forecast['Month'])
    df_po['Month'] = pd.to_datetime(df_po['Month'])
    
    # Merge Forecast & PO
    df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
    
    if not df_product.empty:
        meta = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
        df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')
    
    # Calculate Ratio (Handle div by zero later)
    df_merged['Ratio'] = np.where(df_merged['Forecast_Qty']>0, (df_merged['PO_Qty']/df_merged['Forecast_Qty'])*100, 0)
    
    # --- LOGIC STATUS BARU (V9.0) ---
    conditions = [
        df_merged['Forecast_Qty'] == 0,  # Prioritas 1: No Rofo (Flush Out / Disc)
        df_merged['Ratio'] < 80, 
        (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), 
        df_merged['Ratio'] > 120
    ]
    choices = ['No Rofo', 'Under', 'Accurate', 'Over']
    
    df_merged['Status'] = np.select(conditions, choices, default='Unknown')
    
    # --- LOGIC AKURASI BARU: Exclude 'No Rofo' ---
    # Jika Status 'No Rofo', APE dianggap NaN agar tidak dihitung di rata-rata
    df_merged['APE'] = np.where(df_merged['Status'] == 'No Rofo', np.nan, abs(df_merged['Ratio'] - 100))
    
    monthly_stats = {}
    for month in sorted(df_merged['Month'].unique()):
        month_data = df_merged[df_merged['Month'] == month].copy()
        
        # Calculate Accuracy excluding No Rofo
        mean_ape = month_data['APE'].mean() # Pandas secara default ignore NaN
        accuracy = 100 - mean_ape if not pd.isna(mean_ape) else 0
        
        # Total records (untuk display)
        total_records = len(month_data)
        
        monthly_stats[month] = {
            'accuracy': accuracy,
            'counts': month_data['Status'].value_counts().to_dict(),
            'total': total_records,
            'data': month_data
        }
    return monthly_stats

def calculate_inventory_metrics(df_stock, df_sales, df_product):
    if df_stock.empty: return pd.DataFrame()
    
    if not df_sales.empty:
        df_sales['Month'] = pd.to_datetime(df_sales['Month'])
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
    
    # Filter out No Rofo for Tier Chart Accuracy view (Optional, but cleaner)
    # Or keep it to show distribution. Let's keep it but color it Gray.
    agg = df_clean.groupby(['SKU_Tier', 'Status']).size().reset_index(name='Count')
    
    fig = px.bar(agg, x="SKU_Tier", y="Count", color="Status", 
                 title="Accuracy Distribution by Tier",
                 color_discrete_map={
                     'Under': '#ef5350', 
                     'Accurate': '#66bb6a', 
                     'Over': '#ffa726',
                     'No Rofo': '#78909c' # Gray
                 },
                 template="plotly_white")
    fig.update_layout(height=400)
    return fig

# --- ====================================================== ---
# ---                3. MAIN DASHBOARD UI                    ---
# --- ====================================================== ---

client = init_gsheet_connection()
if not client: st.stop()

with st.spinner('🔄 Loading Intelligence Engine...'):
    all_data = load_and_process_data(client)
    
monthly_perf = calculate_monthly_performance(all_data['forecast'], all_data['po'], all_data['product'])
inv_df = calculate_inventory_metrics(all_data['stock'], all_data['sales'], all_data['product'])

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance Dashboard", "📊 Tier Analysis", "📦 Inventory Analysis", "🔍 Sales Analysis", "📋 Data Explorer"
])

# --- TAB 1: DASHBOARD UTAMA ---
with tab1:
    if monthly_perf:
        st.subheader("Forecast Accuracy - 3 Bulan Terakhir")
        st.caption("*Accuracy score exclude 'No Rofo' (Flush Out/Disc items)")
        
        last_3_months = sorted(monthly_perf.keys())[-3:]
        cols = st.columns(len(last_3_months))
        
        for idx, month in enumerate(last_3_months):
            data = monthly_perf[month]
            counts = data['counts']
            with cols[idx]:
                html_code = f"""
<div class="month-card">
    <div class="month-title">{month.strftime('%b %Y')}</div>
    <div style="text-align:center; margin-bottom:15px;">
        <span style="font-size:2.5rem; font-weight:800; color:#5c6bc0;">{data['accuracy']:.1f}%</span>
        <br><span style="color:#888; font-size:0.8rem;">Performance Accuracy</span>
    </div>
    <div class="status-badge-container" style="justify-content: center; gap: 8px;">
        <div class="badge badge-red">Und: {counts.get('Under',0)}</div>
        <div class="badge badge-green">Acc: {counts.get('Accurate',0)}</div>
        <div class="badge badge-orange">Ovr: {counts.get('Over',0)}</div>
        <div class="badge badge-gray">None: {counts.get('No Rofo',0)}</div>
    </div>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)
        
        # --- TOTALAN BULAN TERAKHIR ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Total Metrics (Bulan Terakhir)")
        
        last_month = last_3_months[-1]
        last_month_data = monthly_perf[last_month]['data']
        total_skus = len(last_month_data)
        
        # Helper Stat
        grp = last_month_data.groupby('Status').agg({'SKU_ID':'count', 'Forecast_Qty':'sum', 'PO_Qty':'sum'}).to_dict('index')
        
        def get_stat(status):
            row = grp.get(status, {'SKU_ID': 0, 'Forecast_Qty': 0, 'PO_Qty': 0})
            count = row['SKU_ID']
            pct = (count / total_skus * 100) if total_skus > 0 else 0
            # Jika No Rofo, tampilkan PO Qty (karena Forecast 0)
            qty = row['PO_Qty'] if status == 'No Rofo' else row['Forecast_Qty']
            return count, pct, qty

        u_cnt, u_pct, u_qty = get_stat('Under')
        a_cnt, a_pct, a_qty = get_stat('Accurate')
        o_cnt, o_pct, o_qty = get_stat('Over')
        nr_cnt, nr_pct, nr_qty = get_stat('No Rofo')
        
        avg_acc = monthly_perf[last_month]['accuracy']
        
        # 5 Columns Layout
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
            <div class="summary-card bg-solid-red">
                <div class="sum-title">UNDER</div>
                <div class="sum-value">{u_cnt}</div>
                <div class="sum-pct">{u_pct:.1f}%</div>
                <div class="sum-footer">Rofo: {u_qty:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="summary-card bg-solid-green">
                <div class="sum-title">ACCURATE</div>
                <div class="sum-value">{a_cnt}</div>
                <div class="sum-pct">{a_pct:.1f}%</div>
                <div class="sum-footer">Rofo: {a_qty:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="summary-card bg-solid-orange">
                <div class="sum-title">OVER</div>
                <div class="sum-value">{o_cnt}</div>
                <div class="sum-pct">{o_pct:.1f}%</div>
                <div class="sum-footer">Rofo: {o_qty:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c4:
            st.markdown(f"""
            <div class="summary-card bg-solid-gray">
                <div class="sum-title">NO ROFO</div>
                <div class="sum-value">{nr_cnt}</div>
                <div class="sum-pct">{nr_pct:.1f}%</div>
                <div class="sum-footer">PO: {nr_qty:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c5:
            st.markdown(f"""
            <div class="summary-card bg-solid-white">
                <div class="sum-title">PERFORMANCE</div>
                <div class="sum-value">{avg_acc:.1f}%</div>
                <div class="sum-pct">{total_skus} Total SKUs</div>
                <div class="sum-footer">{last_month.strftime('%b')} Score</div>
            </div>
            """, unsafe_allow_html=True)

        # C. EVALUASI ROFO
        st.divider()
        st.subheader(f"📋 Evaluasi Rofo - {last_month.strftime('%b %Y')}")
        
        eval_df = pd.merge(last_month_data, inv_df[['SKU_ID', 'Stock_Qty', 'Avg_Sales_3M']], on='SKU_ID', how='left')
        cols_final = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status', 
                      'Forecast_Qty', 'PO_Qty', 'Ratio', 'Stock_Qty', 'Avg_Sales_3M']
        cols_final = [c for c in cols_final if c in eval_df.columns]
        
        df_show = eval_df[cols_final].rename(columns={'Ratio': 'Achv %', 'Stock_Qty': 'Stock', 'Avg_Sales_3M': 'Avg Sales'})
        
        t1, t2, t3 = st.tabs(["📉 Detail UNDER", "📈 Detail OVER", "⚪ Detail NO ROFO"])
        
        with t1:
            df_u = df_show[df_show['Status']=='Under'].sort_values('Achv %')
            st.dataframe(df_u, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)
            
        with t2:
            df_o = df_show[df_show['Status']=='Over'].sort_values('Achv %', ascending=False)
            st.dataframe(df_o, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)
            
        with t3:
            df_nr = df_show[df_show['Status']=='No Rofo'].sort_values('PO_Qty', ascending=False)
            st.dataframe(df_nr, column_config={"Achv %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)

    else:
        st.warning("Data Forecast/PO tidak cukup.")

# --- TAB 2: TIER ANALYSIS ---
with tab2:
    st.subheader("📊 Tier Analysis")
    if monthly_perf:
        last_month = sorted(monthly_perf.keys())[-1]
        last_month_df = monthly_perf[last_month]['data']
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = create_tier_chart(last_month_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'SKU_Tier' in last_month_df.columns:
                # Exclude No Rofo from Accuracy % calc
                valid_tier = last_month_df[last_month_df['Status'] != 'No Rofo']
                ts = valid_tier.groupby(['SKU_Tier', 'Status']).size().unstack(fill_value=0)
                ts['Total'] = ts.sum(axis=1)
                ts['Acc %'] = (ts.get('Accurate', 0) / ts['Total'] * 100).round(1)
                st.dataframe(ts.sort_values('Acc %', ascending=False), use_container_width=True)

# --- TAB 3: INVENTORY ANALYSIS ---
with tab3:
    st.subheader("📦 Inventory Health")
    if not inv_df.empty:
        fil = st.multiselect("Filter Status", inv_df['Status'].unique(), default=['Need Replenishment', 'High Stock'])
        show_cols = ['SKU_ID', 'Product_Name', 'Stock_Qty', 'Avg_Sales_3M', 'Cover_Months', 'Status', 'Brand', 'SKU_Tier']
        show_cols = [c for c in show_cols if c in inv_df.columns]
        
        st.dataframe(
            inv_df[inv_df['Status'].isin(fil)][show_cols].sort_values('Cover_Months', ascending=False),
            column_config={"Cover_Months": st.column_config.NumberColumn(format="%.1f")},
            use_container_width=True
        )

# --- TAB 4: SALES ---
with tab4:
    st.subheader("🔍 Sales vs Forecast Deviation")
    if 'sales' in all_data and 'forecast' in all_data:
        # FIX: Ensure datetime matching for Sales & Forecast
        sales_df = all_data['sales'].copy()
        sales_df['Month'] = pd.to_datetime(sales_df['Month'])
        fc_df = all_data['forecast'].copy()
        fc_df['Month'] = pd.to_datetime(fc_df['Month'])
        
        common = sorted(set(sales_df['Month']) & set(fc_df['Month']))
        if common:
            lm = common[-1]
            s = sales_df[sales_df['Month']==lm]
            f = fc_df[fc_df['Month']==lm]
            comp = pd.merge(s, f, on='SKU_ID', suffixes=('_Sales', '_Fc'))
            
            if not all_data['product'].empty:
                comp = pd.merge(comp, all_data['product'][['SKU_ID', 'Product_Name']], on='SKU_ID', how='left')
                
            comp['Dev %'] = np.where(comp['Forecast_Qty']>0, (comp['Sales_Qty']-comp['Forecast_Qty'])/comp['Forecast_Qty']*100, 0)
            comp['Abs Dev'] = abs(comp['Dev %'])
            
            st.dataframe(
                comp[comp['Abs Dev']>30].sort_values('Abs Dev', ascending=False),
                column_config={"Dev %": st.column_config.NumberColumn(format="%.1f%%")},
                use_container_width=True
            )

# --- TAB 5: RAW DATA ---
with tab5:
    opt = st.selectbox("Dataset", ["Sales", "Forecast", "PO", "Stock"])
    d_map = {"Sales": all_data.get('sales'), "Forecast": all_data.get('forecast'), "PO": all_data.get('po'), "Stock": all_data.get('stock')}
    st.dataframe(d_map[opt], use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
