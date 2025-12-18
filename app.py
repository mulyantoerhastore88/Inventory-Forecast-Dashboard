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

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Inventory Intelligence Pro", page_icon="📊", layout="wide")

# --- Custom CSS Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem; border-bottom: 3px solid #667eea;
    }
    .insight-card {
        background: #f0f2f6; border-left: 5px solid #764ba2;
        border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;
    }
    .status-indicator {
        border-radius: 10px; padding: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); color: white;
    }
    .status-under { background: linear-gradient(135deg, #FF5252, #FF1744); }
    .status-accurate { background: linear-gradient(135deg, #4CAF50, #2E7D32); }
    .status-over { background: linear-gradient(135deg, #FF9800, #F57C00); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE DASHBOARD</h1>', unsafe_allow_html=True)

# --- KONEKSI & LOAD DATA ---
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
    url = st.secrets["gsheet_url"]
    try:
        # 1. Product Master (Sumber utama info SKU)
        ws_p = _client.open_by_url(url).worksheet("Product_Master")
        df_p = pd.DataFrame(ws_p.get_all_records())
        df_p.columns = [c.strip().replace(' ', '_') for c in df_p.columns]
        active_skus = df_p[df_p['Status'].get(df_p.index, 'Active').astype(str).str.upper() == 'ACTIVE']['SKU_ID'].tolist()

        def melt_data(sheet_name, val_name):
            ws = _client.open_by_url(url).worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            df.columns = [c.strip() for c in df.columns]
            m_cols = [c for c in df.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            # Hanya ambil SKU_ID untuk melting
            df_l = df.melt(id_vars=['SKU_ID'], value_vars=m_cols, var_name='M_Label', value_name=val_name)
            df_l[val_name] = pd.to_numeric(df_l[val_name], errors='coerce').fillna(0)
            df_l['Month'] = df_l['M_Label'].apply(parse_month_label)
            return df_l[df_l['SKU_ID'].isin(active_skus)]

        return {
            'product': df_p,
            'sales': melt_data("Sales", "Sales_Qty"),
            'forecast': melt_data("Rofo", "Forecast_Qty"),
            'po': melt_data("PO", "PO_Qty"),
            'stock': pd.DataFrame(_client.open_by_url(url).worksheet("Stock_Onhand").get_all_records())
        }
    except Exception as e:
        st.error(f"❌ Error Load: {str(e)}"); return {}

# --- ANALYTICS ENGINE (SKU_ID CENTRIC) ---
def run_analytics(all_data):
    df_f = all_data['forecast']
    df_po = all_data['po']
    df_p = all_data['product']
    
    # 1. Forecast Accuracy (Last 3 Months)
    months = sorted(df_f['Month'].unique())
    last_3m = months[-3:] if len(months) >= 3 else months
    
    # Inner join Forecast & PO berdasarkan SKU_ID & Month
    merged = pd.merge(df_f[df_f['Month'].isin(last_3m)], 
                      df_po[df_po['Month'].isin(last_3m)], 
                      on=['SKU_ID', 'Month'], how='inner')
    
    # Ambil info Tier & Nama dari Product Master di sini
    merged = pd.merge(merged, df_p[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']], on='SKU_ID', how='left')
    
    merged['Ratio'] = np.where(merged['Forecast_Qty'] > 0, (merged['PO_Qty']/merged['Forecast_Qty'])*100, 0)
    merged['Status'] = np.select(
        [merged['Ratio'] < 80, (merged['Ratio'] >= 80) & (merged['Ratio'] <= 120), merged['Ratio'] > 120],
        ['Under', 'Accurate', 'Over'], default='Unknown'
    )
    merged['APE'] = abs(merged['Ratio'] - 100)
    
    # 2. Inventory Health
    df_s = all_data['stock']
    df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
    s_col = next((c for c in ['Stock_Qty', 'Quantity_Available', 'STOCK_SAP'] if c in df_s.columns), df_s.columns[1])
    
    stock_df = df_s[['SKU_ID', s_col]].rename(columns={s_col: 'Stock_Qty'})
    avg_sales = all_data['sales'].groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
    
    inv = pd.merge(stock_df, avg_sales, on='SKU_ID', how='left').merge(df_p[['SKU_ID', 'Product_Name', 'SKU_Tier']], on='SKU_ID', how='left')
    inv['Cover'] = np.where(inv['Sales_Qty'] > 0, inv['Stock_Qty']/inv['Sales_Qty'], 99)
    inv['Inv_Status'] = np.select([inv['Cover'] < 0.8, (inv['Cover'] <= 1.5), inv['Cover'] > 1.5], 
                                  ['Replenish', 'Ideal', 'High Stock'], default='Unknown')
    
    return merged, inv, [m.strftime('%b %Y') for m in last_3m]

# --- UI LOGIC ---
client = init_gsheet_connection()
if client:
    data = load_and_process_data(client)
    if data:
        f_merged, inv_health, period = run_analytics(data)
        
        # AI INSIGHT BOX
        accuracy = 100 - f_merged['APE'].mean()
        st.markdown(f"""
        <div class="insight-card">
            <strong>🚀 AI Intelligence Insight:</strong> Analisis periode <b>{", ".join(period)}</b> menunjukkan akurasi 
            sebesar <b>{accuracy:.1f}%</b>. Fokus pada SKU Tier A yang masuk kategori Under-Forecast untuk mencegah Loss Sales.
        </div>
        """, unsafe_allow_html=True)

        # KEY METRIC CARDS
        c1, c2, c3, c4 = st.columns(4)
        stats = f_merged['Status'].value_counts(normalize=True) * 100
        with c1: st.markdown(f'<div class="status-indicator status-under">UNDER FORECAST<br><span style="font-size:2rem">{stats.get("Under",0):.1f}%</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="status-indicator status-accurate">ACCURATE<br><span style="font-size:2rem">{stats.get("Accurate",0):.1f}%</span></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="status-indicator status-over">OVER FORECAST<br><span style="font-size:2rem">{stats.get("Over",0):.1f}%</span></div>', unsafe_allow_html=True)
        with c4: st.metric("Overall Accuracy", f"{accuracy:.1f}%", f"{f_merged['APE'].mean():.1f}% MAPE", delta_color="inverse")

        # TABS
        t1, t2, t3, t4 = st.tabs(["📊 Performance Flow", "📦 Inventory Health", "🔍 SKU Drill-down", "📋 Raw Data"])
        
        with t1:
            # SANKEY (Centric to SKU_ID & Tier)
            s_data = f_merged.dropna(subset=['SKU_Tier', 'Status'])
            flow = s_data.groupby(['SKU_Tier', 'Status']).size().reset_index(name='val')
            nodes = list(pd.concat([flow['SKU_Tier'], flow['Status']]).unique())
            n_map = {n: i for i, n in enumerate(nodes)}
            fig_s = go.Figure(go.Sankey(
                node=dict(pad=15, thickness=20, label=nodes, color="#667eea"),
                link=dict(source=flow['SKU_Tier'].map(n_map), target=flow['Status'].map(n_map), value=flow['val'])
            ))
            
            st.plotly_chart(fig_s, use_container_width=True)

        with t2:
            st.subheader("Inventory Stock Cover (Months)")
            st.dataframe(inv_health.sort_values('Cover', ascending=False), use_container_width=True)

        with t3:
            st.subheader("SKU Performance Deep Dive")
            sel_sku = st.selectbox("Select SKU to Analyze", options=f_merged['SKU_ID'].unique())
            sku_detail = f_merged[f_merged['SKU_ID'] == sel_sku]
            st.table(sku_detail[['Month', 'Forecast_Qty', 'PO_Qty', 'Ratio', 'Status']])

        with t4:
            st.write(data['product'])

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Control Panel")
    if st.button("🔄 Sync Data"):
        st.cache_data.clear(); st.rerun()
