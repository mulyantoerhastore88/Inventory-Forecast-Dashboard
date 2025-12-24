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
import io
from openpyxl import Workbook

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
        box-shadow: 0 10px 30px rgba(0,0,0,0.15), 0 5px 15px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .summary-card:hover { transform: translateY(-5px); }

    .bg-red { background: linear-gradient(135deg, #e55039 0%, #eb2f06 100%); }
    .bg-green { background: linear-gradient(135deg, #78e08f 0%, #38ada9 100%); }
    .bg-orange { background: linear-gradient(135deg, #f6b93b 0%, #e58e26 100%); }
    .bg-gray { background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%); }
    .bg-blue { background: linear-gradient(135deg, #1e3799 0%, #4a69bd 100%); }
    
    .sum-val { font-size: 2.5rem; font-weight: 800; margin: 5px 0; line-height: 1; }
    .sum-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; letter-spacing: 1px;}
    .sum-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 8px;}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f2f6; border-radius: 8px 8px 0 0; font-weight: 600; border:none;}
    .stTabs [aria-selected="true"] { background-color: white; color: #1e3799; border-top: 3px solid #1e3799; }
    
    /* ALERT BANNER */
    .alert-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .summary-card { margin-bottom: 10px; }
        .sum-val { font-size: 1.8rem; }
        .month-card { padding: 15px; }
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; font-size: 3rem; margin-bottom: -15px;">💎</div>
<h1 class="main-header">INVENTORY INTELLIGENCE PRO V12</h1>
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 2rem;">
    🚀 Integrated Performance, Inventory & Sales Analytics | Real-time Dashboard
</div>
""", unsafe_allow_html=True)

# --- 1. CORE ENGINE (DATA LOADING) ---
@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    """Initialize Google Sheets connection"""
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Connection Failed: {str(e)}")
        return None

def parse_month_label(label):
    """Parse month label to datetime"""
    try:
        label_str = str(label).strip().upper()
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        for m_name, m_num in month_map.items():
            if m_name in label_str:
                # Extract year
                year_part = ''.join(filter(str.isdigit, label_str.replace(m_name, '')))
                if len(year_part) == 2:
                    year = int('20' + year_part)
                elif year_part:
                    year = int(year_part)
                else:
                    year = datetime.now().year
                return datetime(year, m_num, 1)
        return datetime.now()
    except:
        return datetime.now()

@st.cache_data(ttl=300, show_spinner=False)
def load_raw_data(_client):
    """Load raw data from Google Sheets"""
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        
        # 1. Load Product Master
        progress_bar.progress(10)
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_p = pd.DataFrame(ws.get_all_records())
        df_p.columns = [c.strip().replace(' ', '_') for c in df_p.columns]
        
        if 'SKU_ID' in df_p.columns:
            df_p['SKU_ID'] = df_p['SKU_ID'].astype(str).str.strip()
        
        if 'Status' not in df_p.columns:
            df_p['Status'] = 'Active'
        
        df_active = df_p[df_p['Status'].str.upper() == 'ACTIVE'].copy()
        active_ids = df_active['SKU_ID'].tolist()
        data['product'] = df_p
        data['product_active'] = df_active
        
        # Helper function to load and melt data
        def load_and_melt(sheet_name, value_name):
            try:
                ws_temp = _client.open_by_url(gsheet_url).worksheet(sheet_name)
                df_temp = pd.DataFrame(ws_temp.get_all_records())
                df_temp.columns = [c.strip() for c in df_temp.columns]
                
                if 'SKU_ID' in df_temp.columns:
                    df_temp['SKU_ID'] = df_temp['SKU_ID'].astype(str).str.strip()
                else:
                    return pd.DataFrame()
                
                # Identify month columns
                month_cols = [c for c in df_temp.columns if any(
                    m in c.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                )]
                
                if not month_cols:
                    return pd.DataFrame()
                
                # Melt to long format
                df_long = df_temp[['SKU_ID'] + month_cols].melt(
                    id_vars=['SKU_ID'],
                    value_vars=month_cols,
                    var_name='Month_Label',
                    value_name=value_name
                )
                
                df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').fillna(0)
                df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
                df_long['Month'] = pd.to_datetime(df_long['Month'])
                
                # Filter active SKUs
                return df_long[df_long['SKU_ID'].isin(active_ids)]
                
            except Exception as e:
                st.warning(f"⚠️ Warning loading {sheet_name}: {str(e)}")
                return pd.DataFrame()
        
        # 2. Load Sales, Forecast, and PO data
        progress_bar.progress(30)
        data['sales'] = load_and_melt("Sales", "Sales_Qty")
        
        progress_bar.progress(50)
        data['forecast'] = load_and_melt("Rofo", "Forecast_Qty")
        
        progress_bar.progress(70)
        data['po'] = load_and_melt("PO", "PO_Qty")
        
        # 3. Load Stock Data
        progress_bar.progress(90)
        ws_stock = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock_raw = pd.DataFrame(ws_stock.get_all_records())
        df_stock_raw.columns = [c.strip().replace(' ', '_') for c in df_stock_raw.columns]
        
        if 'SKU_ID' in df_stock_raw.columns:
            df_stock_raw['SKU_ID'] = df_stock_raw['SKU_ID'].astype(str).str.strip()
            
            # Identify stock quantity column
            stock_col = next(
                (c for c in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP'] 
                 if c in df_stock_raw.columns),
                None
            )
            
            if stock_col:
                df_stock = df_stock_raw[['SKU_ID', stock_col]].rename(
                    columns={stock_col: 'Stock_Qty'}
                )
                df_stock['Stock_Qty'] = pd.to_numeric(
                    df_stock['Stock_Qty'], errors='coerce'
                ).fillna(0)
                
                # Group by SKU_ID to get max stock
                data['stock'] = df_stock[
                    df_stock['SKU_ID'].isin(active_ids)
                ].groupby('SKU_ID')['Stock_Qty'].max().reset_index()
            else:
                data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
        
        progress_bar.progress(100)
        return data
        
    except Exception as e:
        st.error(f"❌ Error Loading Data: {str(e)}")
        return {}

# --- 2. ANALYTICS ENGINE ---
@st.cache_data(ttl=300)
def calculate_monthly_performance(df_forecast, df_po, df_product):
    """Calculate monthly forecast performance"""
    if df_forecast.empty or df_po.empty:
        return {}
    
    # Convert to datetime
    df_forecast['Month'] = pd.to_datetime(df_forecast['Month'], errors='coerce')
    df_po['Month'] = pd.to_datetime(df_po['Month'], errors='coerce')
    
    # Merge forecast and PO data
    df_merged = pd.merge(
        df_forecast, 
        df_po, 
        on=['SKU_ID', 'Month'], 
        how='inner',
        suffixes=('_forecast', '_po')
    )
    
    # Add product metadata
    if not df_product.empty:
        meta_cols = ['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']
        meta_cols = [c for c in meta_cols if c in df_product.columns]
        if meta_cols:
            meta = df_product[meta_cols].rename(columns={'Status': 'Prod_Status'})
            df_merged = pd.merge(df_merged, meta, on='SKU_ID', how='left')
    
    # Calculate ratio and accuracy
    df_merged['Ratio'] = np.where(
        df_merged['Forecast_Qty'] > 0,
        (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
        0
    )
    
    # Categorize status
    conditions = [
        df_merged['Forecast_Qty'] == 0,
        df_merged['Ratio'] < 80, 
        (df_merged['Ratio'] >= 80) & (df_merged['Ratio'] <= 120), 
        df_merged['Ratio'] > 120
    ]
    choices = ['No Rofo', 'Under', 'Accurate', 'Over']
    df_merged['Status_Rofo'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate Absolute Percentage Error
    df_merged['APE'] = np.where(
        df_merged['Status_Rofo'] == 'No Rofo',
        np.nan,
        abs(df_merged['Ratio'] - 100)
    )
    
    # Calculate monthly statistics
    monthly_stats = {}
    for month in sorted(df_merged['Month'].unique()):
        month_data = df_merged[df_merged['Month'] == month].copy()
        mean_ape = month_data['APE'].mean()
        
        monthly_stats[month] = {
            'accuracy': 100 - mean_ape if not pd.isna(mean_ape) else 0,
            'counts': month_data['Status_Rofo'].value_counts().to_dict(),
            'total': len(month_data),
            'data': month_data,
            'under_qty': month_data[month_data['Status_Rofo'] == 'Under']['Forecast_Qty'].sum(),
            'accurate_qty': month_data[month_data['Status_Rofo'] == 'Accurate']['Forecast_Qty'].sum(),
            'over_qty': month_data[month_data['Status_Rofo'] == 'Over']['Forecast_Qty'].sum(),
            'no_rofo_qty': month_data[month_data['Status_Rofo'] == 'No Rofo']['PO_Qty'].sum()
        }
    
    return monthly_stats

@st.cache_data(ttl=300)
def calculate_inventory_metrics(df_stock, df_sales, df_product):
    """Calculate comprehensive inventory metrics"""
    if df_stock.empty:
        return pd.DataFrame()
    
    # Calculate 3-month average sales
    avg_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Sales_3M'])
    if not df_sales.empty:
        df_sales['Month'] = pd.to_datetime(df_sales['Month'], errors='coerce')
        recent_months = sorted(df_sales['Month'].dropna().unique())[-3:]
        
        if recent_months:
            recent_sales = df_sales[df_sales['Month'].isin(recent_months)]
            avg_sales = recent_sales.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M']
            avg_sales['Avg_Sales_3M'] = avg_sales['Avg_Sales_3M'].round(0).astype(int)
    
    # Merge stock with average sales
    inventory_df = pd.merge(df_stock, avg_sales, on='SKU_ID', how='left')
    inventory_df['Avg_Sales_3M'] = inventory_df['Avg_Sales_3M'].fillna(0)
    
    # Add product metadata
    if not df_product.empty:
        product_cols = ['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']
        product_cols = [c for c in product_cols if c in df_product.columns]
        if product_cols:
            inventory_df = pd.merge(
                inventory_df, 
                df_product[product_cols], 
                on='SKU_ID', 
                how='left'
            )
            inventory_df = inventory_df.rename(columns={'Status': 'Prod_Status'})
    
    # Calculate cover months
    inventory_df['Cover_Months'] = np.where(
        inventory_df['Avg_Sales_3M'] > 0,
        inventory_df['Stock_Qty'] / inventory_df['Avg_Sales_3M'],
        999  # For SKUs with no sales
    )
    
    # Categorize inventory status
    conditions = [
        inventory_df['Cover_Months'] < 0.8,
        (inventory_df['Cover_Months'] >= 0.8) & (inventory_df['Cover_Months'] <= 1.5),
        inventory_df['Cover_Months'] > 1.5
    ]
    choices = ['Need Replenishment', 'Ideal', 'High Stock']
    inventory_df['Status_Stock'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate actionable quantities
    # Quantity to reduce for high stock
    inventory_df['Qty_to_Reduce'] = np.where(
        inventory_df['Status_Stock'] == 'High Stock',
        inventory_df['Stock_Qty'] - (1.5 * inventory_df['Avg_Sales_3M']),
        0
    ).astype(int)
    inventory_df['Qty_to_Reduce'] = np.where(
        inventory_df['Qty_to_Reduce'] < 0, 0, inventory_df['Qty_to_Reduce']
    )
    
    # Quantity to order for low stock
    inventory_df['Qty_to_Order'] = np.where(
        inventory_df['Status_Stock'] == 'Need Replenishment',
        (0.8 * inventory_df['Avg_Sales_3M']) - inventory_df['Stock_Qty'],
        0
    ).astype(int)
    inventory_df['Qty_to_Order'] = np.where(
        inventory_df['Qty_to_Order'] < 0, 0, inventory_df['Qty_to_Order']
    )
    
    return inventory_df

@st.cache_data(ttl=300)
def get_last_3m_sales_pivot(df_sales):
    """Create pivot table for last 3 months sales"""
    if df_sales.empty:
        return pd.DataFrame(), []
    
    df_sales['Month'] = pd.to_datetime(df_sales['Month'], errors='coerce')
    last_3_months = sorted(df_sales['Month'].dropna().unique())[-3:]
    
    if not last_3_months:
        return pd.DataFrame(), []
    
    recent_sales = df_sales[df_sales['Month'].isin(last_3_months)].copy()
    
    # Create pivot table
    pivot_table = recent_sales.pivot_table(
        index='SKU_ID',
        columns='Month',
        values='Sales_Qty',
        aggfunc='sum'
    ).reset_index()
    
    # Format column names
    new_columns = ['SKU_ID']
    month_names = []
    
    for col in pivot_table.columns:
        if isinstance(col, datetime):
            month_name = f"Sales {col.strftime('%b')}"
            new_columns.append(month_name)
            month_names.append(month_name)
    
    pivot_table.columns = new_columns
    pivot_table = pivot_table.fillna(0)
    
    return pivot_table, month_names

# --- 3. EXPORT FUNCTIONALITY ---
def create_excel_report(monthly_perf, inventory_df, sales_analysis_df, last_month):
    """Create Excel report with multiple sheets"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Performance Summary
            if last_month in monthly_perf:
                perf_data = monthly_perf[last_month]['data']
                if not perf_data.empty:
                    perf_data.to_excel(
                        writer, 
                        sheet_name='Performance_Summary', 
                        index=False
                    )
            
            # Sheet 2: Inventory Analysis
            if not inventory_df.empty:
                inv_cols = [
                    'SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Prod_Status',
                    'Stock_Qty', 'Avg_Sales_3M', 'Cover_Months', 'Status_Stock',
                    'Qty_to_Order', 'Qty_to_Reduce'
                ]
                inv_cols = [c for c in inv_cols if c in inventory_df.columns]
                inventory_df[inv_cols].to_excel(
                    writer, 
                    sheet_name='Inventory_Analysis', 
                    index=False
                )
            
            # Sheet 3: Sales Analysis
            if not sales_analysis_df.empty:
                sales_analysis_df.to_excel(
                    writer, 
                    sheet_name='Sales_Analysis', 
                    index=False
                )
            
            # Sheet 4: Action Items
            action_items = []
            if not inventory_df.empty:
                # High Stock items
                high_stock = inventory_df[
                    inventory_df['Status_Stock'] == 'High Stock'
                ].sort_values('Cover_Months', ascending=False).head(20)
                
                # Need Replenishment items
                need_replenish = inventory_df[
                    inventory_df['Status_Stock'] == 'Need Replenishment'
                ].sort_values('Cover_Months').head(20)
                
                action_items = pd.concat([high_stock, need_replenish])
            
            if action_items:
                action_items.to_excel(
                    writer, 
                    sheet_name='Action_Items', 
                    index=False
                )
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        return None

# --- 4. INITIALIZE DATA ---
client = init_gsheet_connection()
if not client:
    st.stop()

# Show loading progress
with st.spinner('🔄 Loading and processing data...'):
    progress_bar = st.progress(0)
    
    # Load raw data
    progress_bar.progress(30)
    all_data = load_raw_data(client)
    
    # Calculate metrics
    progress_bar.progress(60)
    monthly_perf = calculate_monthly_performance(
        all_data.get('forecast', pd.DataFrame()),
        all_data.get('po', pd.DataFrame()),
        all_data.get('product', pd.DataFrame())
    )
    
    progress_bar.progress(80)
    inventory_df = calculate_inventory_metrics(
        all_data.get('stock', pd.DataFrame()),
        all_data.get('sales', pd.DataFrame()),
        all_data.get('product', pd.DataFrame())
    )
    
    progress_bar.progress(90)
    sales_pivot, sales_months_names = get_last_3m_sales_pivot(
        all_data.get('sales', pd.DataFrame())
    )
    
    progress_bar.progress(100)

# --- ALERTS & NOTIFICATIONS ---
if not inventory_df.empty:
    high_stock_count = len(inventory_df[inventory_df['Status_Stock'] == 'High Stock'])
    need_replenish_count = len(inventory_df[inventory_df['Status_Stock'] == 'Need Replenishment'])
    
    if high_stock_count > 10:
        st.markdown(f"""
        <div class="alert-banner">
            ⚠️ HIGH STOCK ALERT: {high_stock_count} SKUs have High Stock levels (Cover > 1.5 months)
        </div>
        """, unsafe_allow_html=True)
    
    if need_replenish_count > 10:
        st.markdown(f"""
        <div class="alert-banner" style="background:linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
            🔄 REPLENISHMENT NEEDED: {need_replenish_count} SKUs need replenishment (Cover < 0.8 months)
        </div>
        """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Date Range Selector
    st.subheader("📅 Date Range")
    analysis_period = st.selectbox(
        "Analysis Period",
        ["Last 3 Months", "Last 6 Months", "Year to Date", "Full History"],
        index=0
    )
    
    # Threshold Settings
    st.subheader("⚙️ Threshold Settings")
    
    with st.expander("Forecast Accuracy"):
        under_threshold = st.slider("Under Forecast Threshold (%)", 0, 100, 80)
        over_threshold = st.slider("Over Forecast Threshold (%)", 100, 200, 120)
    
    with st.expander("Inventory Settings"):
        low_stock_threshold = st.slider("Low Stock (months)", 0.0, 2.0, 0.8, 0.1)
        high_stock_threshold = st.slider("High Stock (months)", 1.0, 6.0, 1.5, 0.1)
    
    # Actions
    st.subheader("📊 Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Data Check", use_container_width=True):
            with st.spinner("Checking data quality..."):
                issues = []
                
                if all_data.get('forecast', pd.DataFrame()).empty:
                    issues.append("Forecast data is empty")
                if all_data.get('po', pd.DataFrame()).empty:
                    issues.append("PO data is empty")
                if all_data.get('stock', pd.DataFrame()).empty:
                    issues.append("Stock data is empty")
                
                if issues:
                    st.warning(f"Found {len(issues)} issues:")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ Data quality check passed!")
    
    # Export Functionality
    st.subheader("📥 Export Reports")
    
    if st.button("📈 Export Excel Report", use_container_width=True):
        if monthly_perf:
            last_month = sorted(monthly_perf.keys())[-1]
            
            # Prepare sales analysis data
            sales_analysis_df = pd.DataFrame()
            if 'sales' in all_data and 'forecast' in all_data:
                last_3_months = sorted(all_data['sales']['Month'].unique())[-3:]
                sales_3m = all_data['sales'][all_data['sales']['Month'].isin(last_3_months)]
                forecast_3m = all_data['forecast'][all_data['forecast']['Month'].isin(last_3_months)]
                
                sales_pivot = sales_3m.pivot_table(
                    index='SKU_ID', 
                    columns='Month', 
                    values='Sales_Qty', 
                    aggfunc='sum'
                ).reset_index()
                sales_pivot.columns = ['SKU_ID'] + [
                    f"Sales {c.strftime('%b')}" for c in sales_pivot.columns 
                    if isinstance(c, datetime)
                ]
                
                forecast_pivot = forecast_3m.pivot_table(
                    index='SKU_ID', 
                    columns='Month', 
                    values='Forecast_Qty', 
                    aggfunc='sum'
                ).reset_index()
                forecast_pivot.columns = ['SKU_ID'] + [
                    f"Fc {c.strftime('%b')}" for c in forecast_pivot.columns 
                    if isinstance(c, datetime)
                ]
                
                sales_analysis_df = pd.merge(sales_pivot, forecast_pivot, on='SKU_ID', how='outer').fillna(0)
                
                if not all_data['product'].empty:
                    sales_analysis_df = pd.merge(
                        sales_analysis_df,
                        all_data['product'][['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status']],
                        on='SKU_ID',
                        how='left'
                    )
            
            excel_file = create_excel_report(monthly_perf, inventory_df, sales_analysis_df, last_month)
            
            if excel_file:
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_file,
                    file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    
    if not all_data['product_active'].empty:
        st.metric("Active SKUs", len(all_data['product_active']))
    
    if not inventory_df.empty:
        total_stock = inventory_df['Stock_Qty'].sum()
        st.metric("Total Stock", f"{total_stock:,.0f}")
    
    if monthly_perf:
        last_month = sorted(monthly_perf.keys())[-1]
        accuracy = monthly_perf[last_month]['accuracy']
        st.metric("Latest Accuracy", f"{accuracy:.1f}%")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "📦 Inventory Analysis", "📈 Sales Analysis"])

# ==========================================
# TAB 1: PERFORMANCE DASHBOARD
# ==========================================
with tab1:
    if monthly_perf:
        # A. MONTHLY PERFORMANCE CARDS
        st.subheader("📅 Forecast Performance Trend")
        
        last_3_months = sorted(monthly_perf.keys())[-3:]
        cols = st.columns(len(last_3_months))
        
        for idx, month in enumerate(last_3_months):
            data = monthly_perf[month]
            counts = data['counts']
            
            with cols[idx]:
                html_content = (
                    f'<div class="month-card">'
                    f'<div style="font-size:1.1rem; font-weight:700; color:#333; border-bottom:1px solid #eee; padding-bottom:5px;">'
                    f'{month.strftime("%b %Y")}'
                    f'</div>'
                    f'<div style="font-size:2.2rem; font-weight:800; color:#1e3799; margin:10px 0;">'
                    f'{data["accuracy"]:.1f}%'
                    f'</div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.75rem;">'
                    f'<span style="color:#eb2f06">Und: {counts.get("Under",0)}</span>'
                    f'<span style="color:#2ecc71">Acc: {counts.get("Accurate",0)}</span>'
                    f'<span style="color:#e67e22">Ovr: {counts.get("Over",0)}</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(html_content, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # B. QUICK INVENTORY OVERVIEW
        if not inventory_df.empty:
            st.subheader("📦 Quick Inventory Overview")
            
            col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
            
            with col_inv1:
                need_replenish = len(inventory_df[inventory_df['Status_Stock'] == 'Need Replenishment'])
                st.metric(
                    "Need Replenishment", 
                    need_replenish,
                    help="SKUs with stock cover less than 0.8 months"
                )
            
            with col_inv2:
                ideal = len(inventory_df[inventory_df['Status_Stock'] == 'Ideal'])
                st.metric(
                    "Ideal Stock", 
                    ideal,
                    help="SKUs with stock cover between 0.8-1.5 months"
                )
            
            with col_inv3:
                high = len(inventory_df[inventory_df['Status_Stock'] == 'High Stock'])
                st.metric(
                    "High Stock", 
                    high,
                    help="SKUs with stock cover more than 1.5 months"
                )
            
            with col_inv4:
                total_stock = inventory_df['Stock_Qty'].sum()
                st.metric(
                    "Total Stock", 
                    f"{total_stock:,.0f}",
                    help="Total stock quantity across all SKUs"
                )
        
        st.markdown("---")
        
        # C. TOTAL METRICS FOR LAST MONTH
        last_month = last_3_months[-1]
        lm_data = monthly_perf[last_month]['data']
        
        st.subheader(f"📊 Total Metrics ({last_month.strftime('%b %Y')})")
        
        status_counts = lm_data['Status_Rofo'].value_counts()
        total_skus = len(lm_data)
        
        # Calculate quantities
        under_qty = lm_data[lm_data['Status_Rofo'] == 'Under']['Forecast_Qty'].sum()
        accurate_qty = lm_data[lm_data['Status_Rofo'] == 'Accurate']['Forecast_Qty'].sum()
        over_qty = lm_data[lm_data['Status_Rofo'] == 'Over']['Forecast_Qty'].sum()
        no_rofo_qty = lm_data[lm_data['Status_Rofo'] == 'No Rofo']['PO_Qty'].sum()
        
        # Summary cards
        r1, r2, r3, r4 = st.columns(4)
        
        with r1:
            st.markdown(f"""
            <div class="summary-card bg-red">
                <div class="sum-title">UNDER FORECAST</div>
                <div class="sum-val">{status_counts.get("Under", 0)}</div>
                <div class="sum-sub">{under_qty:,.0f} Qty</div>
            </div>
            """, unsafe_allow_html=True)
        
        with r2:
            st.markdown(f"""
            <div class="summary-card bg-green">
                <div class="sum-title">ACCURATE FORECAST</div>
                <div class="sum-val">{status_counts.get("Accurate", 0)}</div>
                <div class="sum-sub">{accurate_qty:,.0f} Qty</div>
            </div>
            """, unsafe_allow_html=True)
        
        with r3:
            st.markdown(f"""
            <div class="summary-card bg-orange">
                <div class="sum-title">OVER FORECAST</div>
                <div class="sum-val">{status_counts.get("Over", 0)}</div>
                <div class="sum-sub">{over_qty:,.0f} Qty</div>
            </div>
            """, unsafe_allow_html=True)
        
        with r4:
            st.markdown(f"""
            <div
