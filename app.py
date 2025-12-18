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

# --- CONFIG ---
st.set_page_config(
    page_title="Inventory Intelligence Pro AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PREMIUM (Glassmorphism & Advanced UI) ---
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
        transition: 0.3s; color: white;
    }
    
    .status-under { background: linear-gradient(135deg, #FF5252, #B71C1C); }
    .status-accurate { background: linear-gradient(135deg, #4CAF50, #1B5E20); }
    .status-over { background: linear-gradient(135deg, #FF9800, #E65100); }
    
    .metric-highlight {
        background: white; border-radius: 15px;
        padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border-top: 5px solid #4facfe; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- ====================================================== ---
# ---             KONEKSI & LOAD DATA (STABLE)               ---
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
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        for m_name, m_num in month_map.items():
            if m_name in label_str:
                year_part = label_str.replace(m_name, '').replace('-', '').replace(' ', '').strip()
                year = int('20' + year_part) if len(year_part) == 2 else int(year_part) if year_part else datetime.now().year
                return datetime(year, m_num, 1)
        return datetime.now()
    except: return datetime.now()

@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(_client):
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    try:
        # 1. Product Master
        ws_p = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws_p.get_all_records())
        df_product.columns = [c.strip().replace(' ', '_') for c in df_product.columns]
        if 'Status' not in df_product.columns: df_product['Status'] = 'Active'
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_product_active['SKU_ID'].tolist()

        # 2. Sales & Forecast & PO (Melt Logic)
        def process_sheet(sheet_name, val_name):
            ws = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            m_cols = [c for c in df.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            id_vars = ['SKU_ID']
            for extra in ['Product_Name', 'Brand', 'SKU_Tier']:
                if extra in df.columns: id_vars.append(extra)
            df_long = df.melt(id_vars=id_vars, value_vars=m_cols, var_name='Month_Label', value_name=val_name)
            df_long[val_name] = pd.to_numeric(df_long[val_name], errors='coerce').fillna(0)
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            return df_long[df_long['SKU_ID'].isin(active_ids)]

        data['sales'] = process_sheet("Sales", "Sales_Qty")
        data['forecast'] = process_sheet("Rofo", "Forecast_Qty")
        data['po'] = process_sheet("PO", "PO_Qty")
        
        # 3. Stock
        ws_s = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_s = pd.DataFrame(ws_s.get_all_records())
        df_s.columns = [c.strip().replace(' ', '_') for c in df_s.columns]
        s_col = next((c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] if c in df_s.columns), None)
        df_stock = df_s[['SKU_ID', s_col]].copy().rename(columns={s_col: 'Stock_Qty'})
        df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
        data['stock'] = df_stock[df_stock['SKU_ID'].isin(active_ids)]
        data['product'] = df_product
        data['product_active'] = df_product_active
        return data
    except Exception as e:
        st.error(f"Error Processing Data: {e}")
        return {}

# --- ====================================================== ---
# ---             ANALYTICS ENGINE (THE INTELLIGENCE)        ---
# --- ====================================================== ---

def calculate_metrics(data):
    # Forecast Metrics (Last 3M)
    df_f, df_po, df_p = data['forecast'], data['po'], data['product_active']
    all_months = sorted(df_f['Month'].unique())
    last_3m = all_months[-3:] if len(all_months) >= 3 else all_months
    
    merged = pd.merge(df_f[df_f['Month'].isin(last_3m)], df_po[df_po['Month'].isin(last_3m)], 
                      on=['SKU_ID', 'Month'], how='inner')
    
    merged['Ratio'] = np.where(merged['Forecast_Qty'] > 0, (merged['PO_Qty']/merged['Forecast_Qty'])*100, 0)
    conditions = [merged['Ratio'] < 80, (merged['Ratio'] >= 80) & (merged['Ratio'] <= 120), merged['Ratio'] > 120]
    merged['Accuracy_Status'] = np.select(conditions, ['Under', 'Accurate', 'Over'], default='Unknown')
    merged['APE'] = abs(merged['Ratio'] - 100)
    
    # Inventory Health
    df_s, df_sl = data['stock'], data['sales']
    avg_sales = df_sl.groupby('SKU_ID')['Sales_Qty'].mean().reset_index().rename(columns={'Sales_Qty': 'Avg_Sales'})
    inv = pd.merge(df_s, avg_sales, on='SKU_ID', how='left').merge(df_p[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']], on='SKU_ID')
    inv['Cover'] = np.where(inv['Avg_Sales'] > 0, inv['Stock_Qty']/inv['Avg_Sales'], 99)
    inv['Status'] = np.select([inv['Cover'] < 0.8, (inv['Cover'] >= 0.8) & (inv['Cover'] <= 1.5), inv['Cover'] > 1.5],
                              ['Need Replenishment', 'Ideal', 'High Stock'], default='Unknown')
    
    return {
        'forecast_merged': merged,
        'inventory_health': inv,
        'mape': merged['APE'].mean(),
        'period': [m.strftime('%b %Y') for m in last_3m]
    }

# --- ====================================================== ---
# ---             UI RENDERING (THE POWERFUL VIEW)           ---
# --- ====================================================== ---

client = init_gsheet_connection()
if client:
    with st.spinner('🧠 AI is analyzing your inventory...'):
        all_data = load_and_process_data(client)
        if all_data:
            res = calculate_metrics(all_data)
            
            # --- HEADER ---
            st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE PRO</h1>', unsafe_allow_html=True)
            
            # AI INSIGHT OVERVIEW
            st.markdown(f"""
            <div class="insight-card">
                <strong>💡 AI Insight Advisor:</strong> 
                Performance Forecast 3 bulan terakhir ({", ".join(res['period'])}) mencapai akurasi <strong>{(100-res['mape']):.1f}%</strong>. 
                Ditemukan {len(res['inventory_health'][res['inventory_health']['Status']=='High Stock'])} SKU dengan penumpukan stok tinggi 
                yang beresiko mengurangi cash flow.
            </div>
            """, unsafe_allow_html=True)

            # --- TOP KPI CARDS ---
            c1, c2, c3, c4 = st.columns(4)
            f_data = res['forecast_merged']
            with c1:
                val = (len(f_data[f_data['Accuracy_Status']=='Under'])/len(f_data))*100
                st.markdown(f'<div class="status-indicator status-under">UNDER FORECAST<br><span style="font-size:2rem">{val:.1f}%</span></div>', unsafe_allow_html=True)
            with c2:
                val = (len(f_data[f_data['Accuracy_Status']=='Accurate'])/len(f_data))*100
                st.markdown(f'<div class="status-indicator status-accurate">ACCURATE<br><span style="font-size:2rem">{val:.1f}%</span></div>', unsafe_allow_html=True)
            with c3:
                val = (len(f_data[f_data['Accuracy_Status']=='Over'])/len(f_data))*100
                st.markdown(f'<div class="status-indicator status-over">OVER FORECAST<br><span style="font-size:2rem">{val:.1f}%</span></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-highlight"><span style="color:#666">OVERALL ACCURACY</span><br><span style="font-size:2rem; color:#4facfe">{(100-res["mape"]):.1f}%</span><br>MAPE: {res["mape"]:.1f}%</div>', unsafe_allow_html=True)

            # --- TABS ---
            st.write("")
            tabs = st.tabs(["📈 Monthly Performance", "🏷️ Tier & Flow", "📦 Inventory Health", "🔍 SKU Drill-down", "📋 Raw Data"])

            # TAB 1: MONTHLY
            with tabs[0]:
                m_df = f_data.groupby('Month').agg({'APE': lambda x: 100-x.mean()}).reset_index().rename(columns={'APE':'Accuracy'})
                fig = px.line(m_df, x='Month', y='Accuracy', title="Historical Forecast Accuracy Trend", markers=True)
                fig.update_layout(yaxis_range=[0,105])
                st.plotly_chart(fig, use_container_width=True)

            # TAB 2: TIER & SANKEY (POWERFUL)
            with tabs[1]:
                col_t1, col_t2 = st.columns([2,1])
                with col_t1:
                    # Sankey Logic
                    s_data = f_data.groupby(['SKU_Tier', 'Accuracy_Status']).size().reset_index(name='count')
                    nodes = list(pd.concat([s_data['SKU_Tier'], s_data['Accuracy_Status']]).unique())
                    node_map = {n: i for i, n in enumerate(nodes)}
                    
                    fig_s = go.Figure(data=[go.Sankey(
                        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color="blue"),
                        link=dict(source=s_data['SKU_Tier'].map(node_map), target=s_data['Accuracy_Status'].map(node_map), value=s_data['count'])
                    )])
                    fig_s.update_layout(title_text="SKU Flow: Tier to Accuracy Status", height=500)
                    st.plotly_chart(fig_s, use_container_width=True)
                
                with col_t2:
                    st.subheader("Accuracy by Tier")
                    t_acc = f_data.groupby('SKU_Tier')['APE'].apply(lambda x: 100-x.mean()).reset_index()
                    st.dataframe(t_acc, column_config={"APE": st.column_config.ProgressColumn("Acc %", min_value=0, max_value=100)}, use_container_width=True)

            # TAB 3: INVENTORY HEALTH (POWERFUL)
            with tabs[2]:
                inv = res['inventory_health']
                c_inv1, c_inv2, c_inv3 = st.columns(3)
                with c_inv1: st.metric("Replenishment Needed", len(inv[inv['Status']=='Need Replenishment']))
                with c_inv2: st.metric("Healthy SKUs", len(inv[inv['Status']=='Ideal']))
                with c_inv3: st.metric("Overstock Risk", len(inv[inv['Status']=='High Stock']))
                
                st.subheader("⚠️ High Stock Items (Dead Capital Risk)")
                st.dataframe(inv[inv['Status']=='High Stock'].sort_values('Cover', ascending=False), 
                             column_config={"Cover": st.column_config.NumberColumn("Month Cover", format="%.2f")},
                             use_container_width=True)

            # TAB 4: SKU EVALUATION (ADVANCED SEARCH)
            with tabs[3]:
                search = st.text_input("🔍 Search SKU or Product Name")
                eval_df = res['inventory_health']
                if search:
                    eval_df = eval_df[eval_df['Product_Name'].str.contains(search, case=False) | eval_df['SKU_ID'].str.contains(search, case=False)]
                
                st.dataframe(eval_df, use_container_width=True)
                
                # Plotly Scatter for SKU Position
                fig_scatter = px.scatter(eval_df, x='Avg_Sales', y='Stock_Qty', color='Status', size='Cover',
                                         hover_name='Product_Name', title="SKU Position: Sales vs Stock")
                st.plotly_chart(fig_scatter, use_container_width=True)

            # TAB 5: RAW DATA
            with tabs[4]:
                st.selectbox("Select Sheet", ["Product", "Sales", "Forecast", "PO"], key="sheet_sel")
                st.write("Full data access available in the sidebar download.")

            # FOOTER
            st.divider()
            st.markdown("<p style='text-align: center; color: grey;'>Inventory Intelligence Pro v5.0 | AI-Powered Demand Planning</p>", unsafe_allow_html=True)

# --- SIDEBAR REFRESH ---
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🔄 Sync Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.info("Dashboard ini menggunakan data 3 bulan terakhir untuk menghitung akurasi secara dinamis.")
