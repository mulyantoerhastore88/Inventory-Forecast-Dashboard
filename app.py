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
    page_title="Inventory Intelligence Pro V12",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM (FIXED FLOATING EFFECT) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
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
        /* Efek Floating Lebih Tebal */
        box-shadow: 0 10px 30px rgba(0,0,0,0.15), 0 5px 15px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .summary-card:hover { transform: translateY(-5px); }

    .bg-red { background: linear-gradient(135deg, #e55039 0%, #eb2f06 100%); }
    .bg-green { background: linear-gradient(135deg, #78e08f 0%, #38ada9 100%); }
    .bg-orange { background: linear-gradient(135deg, #f6b93b 0%, #e58e26 100%); }
    .bg-gray { background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%); }
    
    .sum-val { font-size: 2.5rem; font-weight: 800; margin: 5px 0; line-height: 1; }
    .sum-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; letter-spacing: 1px;}
    .sum-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 8px;}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f2f6; border-radius: 8px 8px 0 0; font-weight: 600; border:none;}
    .stTabs [aria-selected="true"] { background-color: white; color: #1e3799; border-top: 3px solid #1e3799; }
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

# --- 1. CORE ENGINE (DATA LOADING) ---
@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {str(e)}"); return None

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
        if 'SKU_ID' in df_p.columns: df_p['SKU_ID'] = df_p['SKU_ID'].astype(str).str.strip()
        if 'Status' not in df_p.columns: df_p['Status'] = 'Active'
        df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_active['SKU_ID'].tolist()

        def robust_melt(sheet_name, val_col):
            ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df_temp = pd.DataFrame(ws_temp.get_all_records())
            df_temp.columns = [c.strip() for c in df_temp.columns]
            if 'SKU_ID' in df_temp.columns: df_temp['SKU_ID'] = df_temp['SKU_ID'].astype(str).str.strip()
            else: return pd.DataFrame()
            m_cols = [c for c in df_temp.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            df_long = df_temp[['SKU_ID'] + m_cols].melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='Month_Label', value_name=val_col)
            df_long[val_col] = pd.to_numeric(df_long[val_col], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            df_long['Month'] = pd.to_datetime(df_long['Month'])
            return df_long[df_long['SKU_ID'].isin(active_ids)]

        data['sales'] = robust_melt("Sales", "Sales_Qty")
        data['forecast'] = robust_melt("Rofo", "Forecast_Qty")
        data['po'] = robust_melt("PO", "PO_Qty")
        
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_s = pd.DataFrame(ws_s.get_all_records())
        df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
        if 'SKU_ID' in df_s.columns: df_s['SKU_ID'] = df_s['SKU_ID'].astype(str).str.strip()
        s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_s.columns), None)
        if s_col and 'SKU_ID' in df_s.columns:
            df_stock = df_s[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
            df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
            data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_ids)].groupby('SKU_ID').max().reset_index()
        else: data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
            
        data['product'] = df_p
        data['product_active'] = df_active
        return data
    except Exception as e: st.error(f"Error Loading: {e}"); return {}

# --- 2. ANALYTICS ENGINE ---
def calculate_monthly_performance(df_forecast, df_po, df_product):
    if df_forecast.empty or df_po.empty: return {}
    df_forecast['Month'] = pd.to_datetime(df_forecast['Month'])
    df_po['Month'] = pd.to_datetime(df_po['Month'])
    df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
    
    if not df_product.empty:
        meta = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']].rename(columns={'Status':'Prod_Status'})
        df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')

    df_merged['Ratio'] = np.where(df_merged['Forecast_Qty']>0, (df_merged['PO_Qty']/df_merged['Forecast_Qty'])*100, 0)
    
    conditions = [
        df_merged['Forecast_Qty'] == 0,
        df_merged['Ratio'] < 80, 
        (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), 
        df_merged['Ratio'] > 120
    ]
    df_merged['Status_Rofo'] = np.select(conditions, ['No Rofo', 'Under', 'Accurate', 'Over'], default='Unknown')
    df_merged['APE'] = np.where(df_merged['Status_Rofo'] == 'No Rofo', np.nan, abs(df_merged['Ratio'] - 100))
    
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
    return monthly_stats

def calculate_inventory_metrics(df_stock, df_sales, df_product):
    if df_stock.empty: return pd.DataFrame()
    if not df_sales.empty:
        df_sales['Month'] = pd.to_datetime(df_sales['Month'])
        months = sorted(df_sales['Month'].unique())[-3:]
        sales_3m = df_sales[df_sales['Month'].isin(months)]
        avg_sales = sales_3m.groupby('SKU_ID')['Sales_Qty'].mean().reset_index(name='Avg_Sales_3M')
        # Round Avg Sales
        avg_sales['Avg_Sales_3M'] = avg_sales['Avg_Sales_3M'].round(0).astype(int)
    else:
        avg_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Sales_3M'])
        
    inv = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
    inv['Avg_Sales_3M'] = inv['Avg_Sales_3M'].fillna(0)
    
    if not df_product.empty:
        inv = pd.merge(inv, df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']], on='SKU_ID', how='left')
        inv = inv.rename(columns={'Status': 'Prod_Status'})

    inv['Cover_Months'] = np.where(inv['Avg_Sales_3M']>0, inv['Stock_Qty']/inv['Avg_Sales_3M'], 999)
    inv['Status_Stock'] = np.select(
        [inv['Cover_Months'] < 0.8, (inv['Cover_Months'] >= 0.8) & (inv['Cover_Months'] <= 1.5), inv['Cover_Months'] > 1.5],
        ['Need Replenishment', 'Ideal', 'High Stock'], default='Unknown'
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
    return inv

def get_last_3m_sales_pivot(df_sales):
    if df_sales.empty: return pd.DataFrame(), []
    df_sales['Month'] = pd.to_datetime(df_sales['Month'])
    last_3_months = sorted(df_sales['Month'].unique())[-3:]
    df_3m = df_sales[df_sales['Month'].isin(last_3_months)].copy()
    
    df_pivot = df_3m.pivot_table(index='SKU_ID', columns='Month', values='Sales_Qty', aggfunc='sum').reset_index()
    
    new_cols = ['SKU_ID']
    month_names = []
    for col in df_pivot.columns:
        if isinstance(col, datetime):
            m_name = f"Sales {col.strftime('%b')}"
            new_cols.append(m_name)
            month_names.append(m_name)
    
    df_pivot.columns = ['SKU_ID'] + month_names
    df_pivot = df_pivot.fillna(0)
    return df_pivot, month_names

# --- 3. UI DASHBOARD ---
client = init_gsheet_connection()
if not client: st.stop()

with st.spinner('🔄 Synchronizing Engine...'):
    all_data = load_and_process_data(client)
    
monthly_perf = calculate_monthly_performance(all_data['forecast'], all_data['po'], all_data['product'])
inv_df = calculate_inventory_metrics(all_data['stock'], all_data['sales'], all_data['product'])
sales_pivot, sales_months_names = get_last_3m_sales_pivot(all_data['sales'])

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "📦 Inventory Analysis", "📈 Sales Analysis"])

# ==========================================
# TAB 1: PERFORMANCE DASHBOARD (FIXED LAYOUT)
# ==========================================
with tab1:
    if monthly_perf:
        # A. MONTHLY CARDS
        st.subheader("📅 Forecast Performance Trend")
        
        last_3_months = sorted(monthly_perf.keys())[-3:]
        cols = st.columns(len(last_3_months))
        
        for idx, month in enumerate(last_3_months):
            data = monthly_perf[month]
            cnt = data['counts']
            with cols[idx]:
                st.markdown(f"""
                <div class="month-card">
                    <div style="font-size:1.1rem; font-weight:700; color:#333; border-bottom:1px solid #eee; padding-bottom:5px;">{month.strftime('%b %Y')}</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#1e3799; margin:10px 0;">{data['accuracy']:.1f}%</div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
                        <span style="color:#eb2f06">Und: {cnt.get('Under',0)}</span>
                        <span style="color:#2ecc71">Acc: {cnt.get('Accurate',0)}</span>
                        <span style="color:#e67e22">Ovr: {cnt.get('Over',0)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # B. TOTAL METRICS (FULL WIDTH ROW)
        last_month = last_3_months[-1]
        lm_data = monthly_perf[last_month]['data']
        
        st.subheader(f"📊 Total Metrics ({last_month.strftime('%b %Y')})")
        grp = lm_data['Status_Rofo'].value_counts()
        total_skus = len(lm_data)
        
        # Hitung Qty
        u_qty = lm_data[lm_data['Status_Rofo']=='Under']['Forecast_Qty'].sum()
        a_qty = lm_data[lm_data['Status_Rofo']=='Accurate']['Forecast_Qty'].sum()
        o_qty = lm_data[lm_data['Status_Rofo']=='Over']['Forecast_Qty'].sum()
        nr_qty = lm_data[lm_data['Status_Rofo']=='No Rofo']['PO_Qty'].sum() # PO for No Rofo
        
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class="summary-card bg-red"><div class="sum-title">UNDER</div><div class="sum-val">{grp.get("Under",0)}</div>
            <div class="sum-sub">{u_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="summary-card bg-green"><div class="sum-title">ACCURATE</div><div class="sum-val">{grp.get("Accurate",0)}</div>
            <div class="sum-sub">{a_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="summary-card bg-orange"><div class="sum-title">OVER</div><div class="sum-val">{grp.get("Over",0)}</div>
            <div class="sum-sub">{o_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class="summary-card bg-gray"><div class="sum-title">NO ROFO</div><div class="sum-val">{grp.get("No Rofo",0)}</div>
            <div class="sum-sub">{nr_qty:,.0f} Qty</div></div>""", unsafe_allow_html=True)
            
        # C. TIER ANALYSIS (BELOW TOTAL - PROPER CHART)
        st.markdown("---")
        st.subheader("🏷️ Tier Analysis")
        
        tier_df = lm_data.dropna(subset=['SKU_Tier'])
        
        col_t1, col_t2 = st.columns([2, 1])
        
        with col_t1:
            # Proper Stacked Bar
            tier_agg = tier_df.groupby(['SKU_Tier', 'Status_Rofo']).size().reset_index(name='Count')
            fig = px.bar(tier_agg, x='SKU_Tier', y='Count', color='Status_Rofo',
                         color_discrete_map={'Under':'#e55039', 'Accurate':'#38ada9', 'Over':'#f6b93b', 'No Rofo':'#95a5a6'},
                         height=300, title="SKU Distribution by Tier")
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_t2:
            st.markdown("##### Accuracy per Tier")
            tier_sum = tier_df.groupby(['SKU_Tier', 'Status_Rofo']).size().unstack(fill_value=0)
            if not tier_sum.empty:
                tier_sum['Total'] = tier_sum.sum(axis=1)
                tier_sum['Acc %'] = (tier_sum.get('Accurate', 0) / tier_sum['Total'] * 100).round(1)
                st.dataframe(tier_sum[['Accurate', 'Under', 'Over', 'Acc %']].sort_values('Acc %', ascending=False), use_container_width=True)

        # D. EVALUASI ROFO TABLE (BOTTOM)
        st.markdown("---")
        st.subheader(f"📋 Evaluasi Rofo - {last_month.strftime('%b %Y')}")
        
        base_eval = pd.merge(lm_data, inv_df[['SKU_ID', 'Stock_Qty', 'Avg_Sales_3M']], on='SKU_ID', how='left')
        
        if not sales_pivot.empty:
            base_eval = pd.merge(base_eval, sales_pivot, on='SKU_ID', how='left')
            for col in sales_months_names:
                if col in base_eval.columns:
                    base_eval[col] = base_eval[col].fillna(0).astype(int)
        
        sales_cols = [c for c in base_eval.columns if c in sales_months_names]
        final_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Prod_Status', 'Status_Rofo', 
                      'Forecast_Qty', 'PO_Qty', 'Ratio', 'Stock_Qty', 'Avg_Sales_3M'] + sales_cols
        
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
            "Avg Sales (3M)": st.column_config.NumberColumn(format="%d")
        }
        
        with t_all: st.dataframe(df_display, column_config=cfg, use_container_width=True, height=500)
        with t_under: st.dataframe(df_display[df_display['Status_Rofo']=='Under'], column_config=cfg, use_container_width=True)
        with t_over: st.dataframe(df_display[df_display['Status_Rofo']=='Over'], column_config=cfg, use_container_width=True)
        with t_nr: st.dataframe(df_display[df_display['Status_Rofo']=='No Rofo'], column_config=cfg, use_container_width=True)
    else:
        st.warning("Data belum tersedia.")

# ==========================================
# TAB 2: INVENTORY ANALYSIS (ACTIONABLE)
# ==========================================
with tab2:
    st.subheader("📦 Inventory Overview")
    
    if not inv_df.empty:
        # 1. INDIVIDUAL MONTH SALES
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
            if idx == 0: with m1: st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            elif idx == 1: with m2: st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            elif idx == 2: with m3: st.metric(f"Sales {m_name}", f"{qty:,.0f}")
            idx += 1
            
        with m4: st.metric("Total Stock", f"{inv_df['Stock_Qty'].sum():,.0f}")
        
        st.markdown("---")
        
        # 2. STOCK BREAKDOWN & ACTIONABLE
        c_don, c_act = st.columns([1, 2])
        
        with c_don:
            st.markdown("##### Stock Status Qty")
            # Hitung Qty per Status
            qty_stat = inv_df.groupby('Status_Stock')['Stock_Qty'].sum().reset_index()
            fig_don = px.pie(qty_stat, values='Stock_Qty', names='Status_Stock', hole=0.5, 
                             color='Status_Stock',
                             color_discrete_map={'Need Replenishment':'#e55039', 'Ideal':'#38ada9', 'High Stock':'#f6b93b'})
            fig_don.update_layout(height=300, margin=dict(t=0,b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_don, use_container_width=True)
            
        with c_act:
            st.markdown("##### Actionable Insights (Qty)")
            
            # Action Cards
            to_order = inv_df['Qty_to_Order'].sum()
            to_reduce = inv_df['Qty_to_Reduce'].sum()
            
            ac1, ac2 = st.columns(2)
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
                
            st.markdown("###### Breakdown per Tier")
            if 'SKU_Tier' in inv_df.columns:
                tier_act = inv_df.groupby('SKU_Tier').agg({
                    'Qty_to_Order': 'sum',
                    'Qty_to_Reduce': 'sum'
                }).reset_index()
                st.dataframe(tier_act, column_config={
                    "Qty_to_Order": st.column_config.NumberColumn("Qty to Order", format="%d"),
                    "Qty_to_Reduce": st.column_config.NumberColumn("Qty to Reduce", format="%d")
                }, use_container_width=True, height=150)

        # 3. DETAIL TABLE
        st.markdown("---")
        st.subheader("📋 Inventory Detail SKU")
        fil_stat = st.multiselect("Filter Status", inv_df['Status_Stock'].unique(), default=['Need Replenishment', 'High Stock'])
        
        view_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Prod_Status', 'Stock_Qty', 'Avg_Sales_3M', 'Cover_Months', 'Status_Stock']
        view_cols = [c for c in view_cols if c in inv_df.columns]
        
        inv_show = inv_df[inv_df['Status_Stock'].isin(fil_stat)][view_cols].rename(columns={
            'Prod_Status': 'Product Status',
            'Stock_Qty': 'Stock Qty',
            'Avg_Sales_3M': 'Avg Sales (3M)',
            'Cover_Months': 'Cover Month',
            'Status_Stock': 'Status Stock'
        })
        
        st.dataframe(inv_show.sort_values('Cover Month', ascending=False),
                     column_config={"Avg Sales (3M)": st.column_config.NumberColumn(format="%d"),
                                    "Cover Month": st.column_config.NumberColumn(format="%.1f"),
                                    "Stock Qty": st.column_config.NumberColumn(format="%d")},
                     use_container_width=True, height=600)
    else: st.warning("Data Inventory belum lengkap.")

# ==========================================
# TAB 3: SALES ANALYSIS (FOCUSED)
# ==========================================
with tab3:
    st.subheader("📈 Sales vs Forecast Analysis")
    
    if 'sales' in all_data and 'forecast' in all_data:
        # A. TOTAL SALES VS FORECAST (ALL MONTHS)
        s_agg = all_data['sales'].groupby('Month')['Sales_Qty'].sum().reset_index()
        f_agg = all_data['forecast'].groupby('Month')['Forecast_Qty'].sum().reset_index()
        
        combo = pd.merge(s_agg, f_agg, on='Month', how='outer').fillna(0)
        combo_melt = combo.melt('Month', var_name='Type', value_name='Qty')
        
        st.markdown("##### 1. Total Overview (All Months)")
        fig_trend = px.bar(combo_melt, x='Month', y='Qty', color='Type', barmode='group',
                           color_discrete_map={'Sales_Qty':'#1e3799', 'Forecast_Qty':'#82ccdd'})
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        # B. FILTERED CHART
        st.markdown("##### 2. Filter Trend by Brand")
        brands = sorted(all_data['product']['Brand'].unique().tolist()) if not all_data['product'].empty else []
        sel_brand = st.selectbox("Select Brand", ["All"] + brands)
        
        s_raw = all_data['sales'].copy()
        f_raw = all_data['forecast'].copy()
        
        if sel_brand != "All" and not all_data['product'].empty:
            brand_skus = all_data['product'][all_data['product']['Brand'] == sel_brand]['SKU_ID'].tolist()
            s_raw = s_raw[s_raw['SKU_ID'].isin(brand_skus)]
            f_raw = f_raw[f_raw['SKU_ID'].isin(brand_skus)]
            
        s_agg_b = s_raw.groupby('Month')['Sales_Qty'].sum().reset_index()
        f_agg_b = f_raw.groupby('Month')['Forecast_Qty'].sum().reset_index()
        combo_b = pd.merge(s_agg_b, f_agg_b, on='Month', how='outer').fillna(0)
        
        fig_b = px.line(combo_b, x='Month', y=['Sales_Qty', 'Forecast_Qty'], markers=True,
                        title=f"Trend for Brand: {sel_brand}", 
                        color_discrete_map={'Sales_Qty':'#1e3799', 'Forecast_Qty':'#e55039'})
        fig_b.update_layout(height=350)
        st.plotly_chart(fig_b, use_container_width=True)
        
        st.markdown("---")
        
        # C. DETAIL SKU
        st.subheader("📋 Detail SKU Sales vs Forecast (3 Bulan Terakhir)")
        
        # Helper Pivot
        last_3m = sorted(s_raw['Month'].unique())[-3:]
        
        # Pivot Sales
        s_3m = s_raw[s_raw['Month'].isin(last_3m)]
        s_piv = s_3m.pivot_table(index='SKU_ID', columns='Month', values='Sales_Qty', aggfunc='sum').reset_index()
        s_piv.columns = ['SKU_ID'] + [f"Sales {c.strftime('%b')}" for c in s_piv.columns if isinstance(c, datetime)]
        
        # Pivot Forecast
        f_3m = f_raw[f_raw['Month'].isin(last_3m)]
        f_piv = f_3m.pivot_table(index='SKU_ID', columns='Month', values='Forecast_Qty', aggfunc='sum').reset_index()
        f_piv.columns = ['SKU_ID'] + [f"Fc {c.strftime('%b')}" for c in f_piv.columns if isinstance(c, datetime)]
        
        det = pd.merge(s_piv, f_piv, on='SKU_ID', how='outer').fillna(0)
        
        if not all_data['product'].empty:
            det = pd.merge(det, all_data['product'][['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status']], on='SKU_ID', how='left')
            det = det.rename(columns={'Status':'Product Status'})
        
        # Filter by Brand for Table
        if sel_brand != "All":
            det = det[det['Brand'] == sel_brand]
            
        sales_cols = [c for c in det.columns if c.startswith('Sales ')]
        fc_cols = [c for c in det.columns if c.startswith('Fc ')]
        
        det['Total Sales 3M'] = det[sales_cols].sum(axis=1)
        det['Total Fc 3M'] = det[fc_cols].sum(axis=1)
        det['Dev %'] = np.where(det['Total Fc 3M']>0, (det['Total Sales 3M']-det['Total Fc 3M'])/det['Total Fc 3M']*100, 0)
        
        final_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Product Status'] + sales_cols + fc_cols + ['Dev %']
        final_cols = [c for c in final_cols if c in det.columns]
        
        st.dataframe(det[final_cols].sort_values('Dev %', ascending=True), column_config={"Dev %": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control")
    if st.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()
