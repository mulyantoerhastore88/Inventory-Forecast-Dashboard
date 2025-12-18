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
    page_title="Inventory Intelligence Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS untuk UI Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .status-box {
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-weight: 600;
        text-align: center;
    }
    .status-under { background-color: #FFEBEE; color: #C62828; border-left: 4px solid #F44336; }
    .status-accurate { background-color: #E8F5E9; color: #2E7D32; border-left: 4px solid #4CAF50; }
    .status-over { background-color: #FFF3E0; color: #EF6C00; border-left: 4px solid #FF9800; }
    
    .inventory-status {
        border-radius: 6px;
        padding: 0.5rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-need-replenishment { background-color: #FFF3E0; color: #EF6C00; }
    .status-ideal { background-color: #E8F5E9; color: #2E7D32; }
    .status-high-stock { background-color: #FFEBEE; color: #C62828; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        border-top: 4px solid;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-blue { border-top-color: #667eea; }
    .card-green { border-top-color: #4CAF50; }
    .card-red { border-top-color: #F44336; }
    .card-orange { border-top-color: #FF9800; }
    .card-purple { border-top-color: #9C27B0; }
</style>
""", unsafe_allow_html=True)

# --- Judul Dashboard ---
st.markdown('<h1 class="main-header">📊 INVENTORY INTELLIGENCE DASHBOARD</h1>', unsafe_allow_html=True)
st.caption(f"🚀 Professional Inventory Control & Demand Planning | Real-time Data | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- ====================================================== ---
# ---               KONEKSI & LOAD DATA                     ---
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

@st.cache_data(ttl=300, show_spinner=False)
def load_all_sheets(_client):
    """Load semua sheet dengan preprocessing khusus"""
    
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    
    # 1. PRODUCT MASTER (Core Reference)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'Product_Name': ['Product_Name', 'Product Name', 'SKU Name'],
            'Status': ['Status', 'SKU_Status']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns and standard_name not in df.columns:
                    df[standard_name] = df[possible_name]
        
        # Ensure required columns exist
        if 'Status' not in df.columns:
            df['Status'] = 'Active'
        
        # Filter only Active SKUs
        df_active = df[df['Status'].str.upper() == 'ACTIVE'].copy()
        
        data['product'] = df
        data['product_active'] = df_active
        
    except Exception as e:
        data['product'] = pd.DataFrame()
        data['product_active'] = pd.DataFrame()
        st.warning(f"Product Master error: {str(e)}")
    
    # 2. SALES DATA (Customer Sales - Monthly)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Sales")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Standardize SKU_ID column
        if 'SKU_ID' not in df.columns and 'Current_SKU' in df.columns:
            df['SKU_ID'] = df['Current_SKU']
        elif 'SKU_ID' not in df.columns and 'SKU_ID' in df.columns:
            df['SKU_ID'] = df['SKU_ID']
        
        # Transform from wide to long format
        month_columns = [col for col in df.columns if any(x in col.lower() for x in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
        
        if month_columns and 'SKU_ID' in df.columns:
            # Keep essential columns
            id_columns = ['SKU_ID', 'SKU_Name', 'Product_Name', 'Brand', 'SKU_Tier']
            available_id_cols = []
            for col in id_columns:
                if col in df.columns:
                    available_id_cols.append(col)
            
            # Melt to long format
            df_long = df.melt(
                id_vars=available_id_cols,
                value_vars=month_columns,
                var_name='Month_Label',
                value_name='Sales_Qty'
            )
            
            # Convert sales quantity to numeric
            df_long['Sales_Qty'] = pd.to_numeric(df_long['Sales_Qty'], errors='coerce').fillna(0)
            
            # Parse month from label
            df_long['Month'] = pd.to_datetime(df_long['Month_Label'], errors='coerce')
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['sales'] = df_long
            
    except Exception as e:
        data['sales'] = pd.DataFrame()
        st.warning(f"Sales data error: {str(e)}")
    
    # 3. ROFO DATA (Forecast - Monthly)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Rofo")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Identify month columns
        month_columns = [col for col in df.columns if any(x in col.lower() for x in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
        
        if month_columns:
            # Keep essential columns
            id_columns = ['SKU_ID', 'Product_Name', 'Brand']
            available_id_cols = []
            for col in id_columns:
                if col in df.columns:
                    available_id_cols.append(col)
            
            # Melt to long format
            df_long = df.melt(
                id_vars=available_id_cols,
                value_vars=month_columns,
                var_name='Month_Label',
                value_name='Forecast_Qty'
            )
            
            # Convert forecast quantity to numeric
            df_long['Forecast_Qty'] = pd.to_numeric(df_long['Forecast_Qty'], errors='coerce').fillna(0)
            
            # Parse month
            df_long['Month'] = pd.to_datetime(df_long['Month_Label'], errors='coerce')
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['forecast'] = df_long
            
    except Exception as e:
        data['forecast'] = pd.DataFrame()
        st.warning(f"Forecast data error: {str(e)}")
    
    # 4. PO DATA (Purchase Orders - Monthly)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("PO")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Identify month columns
        month_columns = [col for col in df.columns if any(x in col.lower() for x in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
        
        if month_columns and 'SKU_ID' in df.columns:
            # Melt to long format
            df_long = df.melt(
                id_vars=['SKU_ID'],
                value_vars=month_columns,
                var_name='Month_Label',
                value_name='PO_Qty'
            )
            
            # Convert to numeric
            df_long['PO_Qty'] = pd.to_numeric(df_long['PO_Qty'], errors='coerce').fillna(0)
            df_long['Month'] = pd.to_datetime(df_long['Month_Label'], errors='coerce')
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['po'] = df_long
            
    except Exception as e:
        data['po'] = pd.DataFrame()
        st.warning(f"PO data error: {str(e)}")
    
    # 5. STOCK ON HAND
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Use Quantity_Available as primary
        stock_col = None
        for col in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP']:
            if col in df.columns:
                stock_col = col
                break
        
        if stock_col and 'SKU_ID' in df.columns:
            # Create clean stock dataframe
            stock_data = pd.DataFrame({
                'SKU_ID': df['SKU_ID'],
                'Stock_Qty': pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
            })
            
            # Remove duplicates by SKU_ID (keep max stock)
            stock_data = stock_data.groupby('SKU_ID')['Stock_Qty'].max().reset_index()
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                stock_data = stock_data[stock_data['SKU_ID'].isin(active_skus)]
            
            data['stock'] = stock_data
            
    except Exception as e:
        data['stock'] = pd.DataFrame()
        st.warning(f"Stock data error: {str(e)}")
    
    return data

# --- ====================================================== ---
# ---               CORE ANALYTICS FUNCTIONS                ---
# --- ====================================================== ---

def calculate_forecast_accuracy_3months(df_forecast, df_po, df_product):
    """Calculate forecast accuracy for LAST 3 MONTHS only"""
    
    metrics = {}
    
    if df_forecast.empty or df_po.empty:
        return metrics
    
    try:
        # Get last 3 months data only
        df_forecast['Month'] = pd.to_datetime(df_forecast['Month'])
        df_po['Month'] = pd.to_datetime(df_po['Month'])
        
        # Find latest month
        latest_month = max(df_forecast['Month'].max(), df_po['Month'].max())
        three_months_ago = latest_month - pd.DateOffset(months=2)  # Get 3 months period
        
        # Filter data for last 3 months
        df_forecast_recent = df_forecast[df_forecast['Month'] >= three_months_ago].copy()
        df_po_recent = df_po[df_po['Month'] >= three_months_ago].copy()
        
        if df_forecast_recent.empty or df_po_recent.empty:
            return metrics
        
        # Merge forecast and PO data for last 3 months
        df_merged = pd.merge(
            df_forecast_recent,
            df_po_recent,
            on=['SKU_ID', 'Month'],
            how='inner',
            suffixes=('_forecast', '_po')
        )
        
        if df_merged.empty:
            return metrics
        
        # Add Product_Name if available
        if not df_product.empty and 'SKU_ID' in df_product.columns and 'Product_Name' in df_product.columns:
            product_names = df_product[['SKU_ID', 'Product_Name']].drop_duplicates()
            df_merged = pd.merge(df_merged, product_names, on='SKU_ID', how='left')
        
        # Calculate PO/Rofo ratio
        df_merged['PO_Rofo_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        # Categorize based on ratio
        conditions = [
            df_merged['PO_Rofo_Ratio'] < 80,
            (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
            df_merged['PO_Rofo_Ratio'] > 120
        ]
        choices = ['Under', 'Accurate', 'Over']
        df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate MAPE
        df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
        mape = df_merged['Absolute_Percentage_Error'].mean()
        overall_accuracy = 100 - mape
        
        # Count by status
        status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
        total_records = len(df_merged)
        status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
        
        # SKU-level accuracy for last 3 months
        sku_accuracy = df_merged.groupby('SKU_ID').apply(
            lambda x: 100 - x['Absolute_Percentage_Error'].mean()
        ).reset_index()
        sku_accuracy.columns = ['SKU_ID', 'SKU_Accuracy']
        
        # Add Product_Name to SKU accuracy
        if 'Product_Name' in df_merged.columns:
            sku_names = df_merged[['SKU_ID', 'Product_Name']].drop_duplicates()
            sku_accuracy = pd.merge(sku_accuracy, sku_names, on='SKU_ID', how='left')
        
        # Add sales data for comparison
        if 'sales' in globals() and not df_sales.empty:
            df_sales_recent = df_sales[df_sales['Month'] >= three_months_ago].copy()
            sales_by_sku = df_sales_recent.groupby('SKU_ID')['Sales_Qty'].sum().reset_index()
            sales_by_sku.columns = ['SKU_ID', 'Total_Sales_3Months']
            sku_accuracy = pd.merge(sku_accuracy, sales_by_sku, on='SKU_ID', how='left')
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'mape': mape,
            'status_counts': status_counts,
            'status_percentages': status_percentages,
            'sku_accuracy': sku_accuracy,
            'detailed_data': df_merged,
            'period': f"{three_months_ago.strftime('%b %Y')} - {latest_month.strftime('%b %Y')}"
        }
        
    except Exception as e:
        st.error(f"Accuracy calculation error: {str(e)}")
    
    return metrics

def calculate_monthly_forecast_accuracy(df_forecast, df_po):
    """Calculate forecast accuracy PER MONTH (not aggregated)"""
    
    monthly_metrics = {}
    
    if df_forecast.empty or df_po.empty:
        return monthly_metrics
    
    try:
        # Merge forecast and PO data
        df_merged = pd.merge(
            df_forecast,
            df_po,
            on=['SKU_ID', 'Month'],
            how='inner',
            suffixes=('_forecast', '_po')
        )
        
        if df_merged.empty:
            return monthly_metrics
        
        # Calculate PO/Rofo ratio per record
        df_merged['PO_Rofo_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        # Calculate accuracy per month
        monthly_accuracy = df_merged.groupby('Month').apply(
            lambda x: 100 - abs(x['PO_Rofo_Ratio'] - 100).mean()
        ).reset_index()
        monthly_accuracy.columns = ['Month', 'Monthly_Accuracy']
        
        # Count status per month
        monthly_status = df_merged.groupby(['Month', 'Accuracy_Status']).size().unstack(fill_value=0)
        
        monthly_metrics = {
            'monthly_accuracy': monthly_accuracy,
            'monthly_status': monthly_status,
            'detailed_data': df_merged
        }
        
    except Exception as e:
        st.error(f"Monthly accuracy calculation error: {str(e)}")
    
    return monthly_metrics

def calculate_inventory_metrics_by_cover(df_stock, df_sales, df_product):
    """Calculate inventory metrics based on cover months"""
    
    metrics = {}
    
    if df_stock.empty:
        return metrics
    
    try:
        # Start with stock data
        df_inventory = df_stock.copy()
        
        # Add product info
        if not df_product.empty:
            product_cols = ['SKU_ID', 'SKU_Tier', 'Brand', 'Product_Name', 'Status']
            available_cols = [col for col in product_cols if col in df_product.columns]
            df_inventory = pd.merge(df_inventory, df_product[available_cols], on='SKU_ID', how='left')
        
        # Calculate cover months if sales data available
        if df_sales is not None and not df_sales.empty:
            # Calculate average monthly sales per SKU
            avg_monthly_sales = df_sales.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales']
            
            # Merge with inventory
            df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
            df_inventory['Avg_Monthly_Sales'] = df_inventory['Avg_Monthly_Sales'].fillna(0)
            
            # Calculate cover months
            df_inventory['Cover_Months'] = np.where(
                df_inventory['Avg_Monthly_Sales'] > 0,
                df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales'],
                999
            )
            
            # Categorize based on cover months
            conditions = [
                df_inventory['Cover_Months'] < 0.8,
                (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 1.5),
                df_inventory['Cover_Months'] > 1.5
            ]
            choices = ['Need Replenishment', 'Ideal/Healthy', 'High Stock']
            df_inventory['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
            
            # Count by status
            status_counts = df_inventory['Inventory_Status'].value_counts().to_dict()
            total_skus = len(df_inventory)
            status_percentages = {k: (v/total_skus*100) for k, v in status_counts.items()}
            
            metrics['cover_months_avg'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
            metrics['status_counts'] = status_counts
            metrics['status_percentages'] = status_percentages
        
        metrics['total_stock_qty'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        metrics['inventory_df'] = df_inventory
        
        # High Stock SKUs only
        if 'Inventory_Status' in df_inventory.columns:
            high_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'High Stock'].copy()
            metrics['high_stock_df'] = high_stock_df
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
    
    return metrics

# --- ====================================================== ---
# ---               VISUALIZATION FUNCTIONS                 ---
# --- ====================================================== ---

def create_simple_bar_chart(data_dict, title):
    """Create simple bar chart without complex encoding"""
    if not data_dict:
        return None
    
    df = pd.DataFrame(list(data_dict.items()), columns=['Status', 'Count'])
    
    # Create simple chart
    chart = alt.Chart(df).mark_bar().encode(
        x='Status',
        y='Count',
        color=alt.Color('Status', scale=alt.Scale(
            domain=['Under', 'Accurate', 'Over'],
            range=['#FF9800', '#4CAF50', '#F44336']
        ))
    ).properties(
        title=title,
        height=300
    )
    
    return chart

def create_monthly_line_chart(monthly_data, title):
    """Create simple line chart for monthly data"""
    if monthly_data.empty:
        return None
    
    # Ensure Month is datetime
    monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
    
    # Create line chart
    line = alt.Chart(monthly_data).mark_line(point=True).encode(
        x=alt.X('Month:T', title='Month'),
        y=alt.Y('Monthly_Accuracy:Q', title='Accuracy (%)'),
        tooltip=['Month:T', 'Monthly_Accuracy']
    ).properties(
        title=title,
        height=350
    )
    
    return line

# --- ====================================================== ---
# ---               MAIN DASHBOARD LAYOUT                   ---
# --- ====================================================== ---

# Initialize connection and load data
client = init_gsheet_connection()

if client is None:
    st.error("❌ Tidak dapat terhubung ke Google Sheets. Periksa koneksi dan kredensial.")
    st.stop()

# Load all data
with st.spinner('🔄 Memuat dan memproses data dari Google Sheets...'):
    all_data = load_all_sheets(client)
    
    df_product = all_data.get('product', pd.DataFrame())
    df_product_active = all_data.get('product_active', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())

# Calculate metrics
forecast_metrics_3months = calculate_forecast_accuracy_3months(df_forecast, df_po, df_product)
monthly_metrics = calculate_monthly_forecast_accuracy(df_forecast, df_po)
inventory_metrics = calculate_inventory_metrics_by_cover(df_stock, df_sales, df_product)

# --- SIDEBAR FILTERS & CONTROLS ---
with st.sidebar:
    st.markdown("### 🔍 Dashboard Controls")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Data Summary")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if not df_product_active.empty:
            st.metric("Active SKUs", len(df_product_active))
    with col_s2:
        if not df_stock.empty:
            st.metric("In Stock", len(df_stock))

# --- MAIN DASHBOARD CONTENT ---

# Header Metrics - LAST 3 MONTHS ONLY
st.subheader("🎯 Forecast Accuracy Metrics - LAST 3 MONTHS (PO vs Rofo)")

if forecast_metrics_3months:
    # Display period
    st.caption(f"**Period:** {forecast_metrics_3months.get('period', 'N/A')}")
    
    # Get status metrics
    status_counts = forecast_metrics_3months.get('status_counts', {})
    status_percentages = forecast_metrics_3months.get('status_percentages', {})
    
    # Create 3 columns for each status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        under_count = status_counts.get('Under', 0)
        under_percentage = status_percentages.get('Under', 0)
        st.markdown(f"""
        <div class="status-box status-under">
            <div style="font-size: 1.2rem; font-weight: 800;">PO/SO UNDER Rofo</div>
            <div style="font-size: 2rem; font-weight: 900;">{under_percentage:.1f}%</div>
            <div style="font-size: 0.9rem;">{under_count} records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accurate_count = status_counts.get('Accurate', 0)
        accurate_percentage = status_percentages.get('Accurate', 0)
        st.markdown(f"""
        <div class="status-box status-accurate">
            <div style="font-size: 1.2rem; font-weight: 800;">AKURAT</div>
            <div style="font-size: 2rem; font-weight: 900;">{accurate_percentage:.1f}%</div>
            <div style="font-size: 0.9rem;">{accurate_count} records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        over_count = status_counts.get('Over', 0)
        over_percentage = status_percentages.get('Over', 0)
        st.markdown(f"""
        <div class="status-box status-over">
            <div style="font-size: 1.2rem; font-weight: 800;">PO/SO OVER Rofo</div>
            <div style="font-size: 2rem; font-weight: 900;">{over_percentage:.1f}%</div>
            <div style="font-size: 0.9rem;">{over_count} records</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall MAPE for last 3 months
    col_mape1, col_mape2 = st.columns(2)
    with col_mape1:
        mape = forecast_metrics_3months.get('mape', 0)
        st.metric("Mean Absolute % Error (MAPE)", f"{mape:.1f}%", "Last 3 months")
    
    with col_mape2:
        accuracy = forecast_metrics_3months.get('overall_accuracy', 0)
        st.metric("Overall Accuracy", f"{accuracy:.1f}%", "Last 3 months")

else:
    st.warning("⚠️ Tidak cukup data untuk 3 bulan terakhir")

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Performance",
    "📦 Inventory Health",
    "🤖 SKU Evaluation",
    "📊 Sales Analytics",
    "📋 Data Explorer"
])

# --- TAB 1: FORECAST PERFORMANCE ---
with tab1:
    st.subheader("📊 Monthly Forecast Performance (PO vs Rofo)")
    
    if not monthly_metrics:
        st.warning("⚠️ Forecast atau PO data tidak tersedia")
    else:
        # Monthly Accuracy Trend
        monthly_acc = monthly_metrics.get('monthly_accuracy')
        if monthly_acc is not None and not monthly_acc.empty:
            # Simple table display instead of chart
            st.subheader("📅 Monthly Accuracy Table")
            
            # Format table
            monthly_display = monthly_acc.copy()
            monthly_display['Month'] = monthly_display['Month'].dt.strftime('%b %Y')
            monthly_display['Monthly_Accuracy'] = monthly_display['Monthly_Accuracy'].round(1)
            
            st.dataframe(
                monthly_display,
                column_config={
                    "Month": "Month",
                    "Monthly_Accuracy": st.column_config.ProgressColumn(
                        "Accuracy %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
                },
                use_container_width=True,
                height=300
            )
        else:
            st.info("Tidak ada data accuracy bulanan")
        
        # Status Distribution by Month
        monthly_status = monthly_metrics.get('monthly_status')
        if monthly_status is not None and not monthly_status.empty:
            st.subheader("📊 Status Distribution by Month")
            
            # Reset index for display
            monthly_status_display = monthly_status.reset_index()
            monthly_status_display['Month'] = monthly_status_display['Month'].dt.strftime('%b %Y')
            
            st.dataframe(
                monthly_status_display,
                use_container_width=True,
                height=300
            )

# --- TAB 2: INVENTORY HEALTH ---
with tab2:
    st.subheader("📦 Inventory Status Dashboard")
    
    if not inventory_metrics:
        st.warning("⚠️ Stock data tidak tersedia")
    else:
        # Inventory Status Metrics
        status_counts = inventory_metrics.get('status_counts', {})
        status_percentages = inventory_metrics.get('status_percentages', {})
        
        if status_counts:
            # Create 3 columns for each status
            col_inv1, col_inv2, col_inv3 = st.columns(3)
            
            with col_inv1:
                need_replenish = status_counts.get('Need Replenishment', 0)
                need_percentage = status_percentages.get('Need Replenishment', 0)
                st.markdown(f"""
                <div class="inventory-status status-need-replenishment">
                    <div style="font-size: 1.1rem; font-weight: 800;">NEED REPLENISHMENT</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{need_percentage:.1f}%</div>
                    <div style="font-size: 0.9rem;">{need_replenish} SKUs (Cover < 0.8 months)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_inv2:
                ideal = status_counts.get('Ideal/Healthy', 0)
                ideal_percentage = status_percentages.get('Ideal/Healthy', 0)
                st.markdown(f"""
                <div class="inventory-status status-ideal">
                    <div style="font-size: 1.1rem; font-weight: 800;">IDEAL/HEALTHY</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{ideal_percentage:.1f}%</div>
                    <div style="font-size: 0.9rem;">{ideal} SKUs (0.8-1.5 months cover)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_inv3:
                high_stock = status_counts.get('High Stock', 0)
                high_percentage = status_percentages.get('High Stock', 0)
                st.markdown(f"""
                <div class="inventory-status status-high-stock">
                    <div style="font-size: 1.1rem; font-weight: 800;">HIGH STOCK</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{high_percentage:.1f}%</div>
                    <div style="font-size: 0.9rem;">{high_stock} SKUs (Cover > 1.5 months)</div>
                </div>
                """, unsafe_allow_html=True)
        
        # HIGH STOCK SKUs Only (for evaluation)
        st.subheader("📋 High Stock SKUs (Need Reduction)")
        
        high_stock_df = inventory_metrics.get('high_stock_df', pd.DataFrame())
        if not high_stock_df.empty:
            # Select columns to display
            display_cols = ['SKU_ID']
            if 'Product_Name' in high_stock_df.columns:
                display_cols.append('Product_Name')
            display_cols.extend(['Stock_Qty', 'Cover_Months'])
            if 'Brand' in high_stock_df.columns:
                display_cols.append('Brand')
            if 'SKU_Tier' in high_stock_df.columns:
                display_cols.append('SKU_Tier')
            
            # Sort by Cover_Months descending (highest stock first)
            high_stock_display = high_stock_df[display_cols].copy()
            high_stock_display = high_stock_display.sort_values('Cover_Months', ascending=False)
            
            st.dataframe(
                high_stock_display,
                column_config={
                    "SKU_ID": "SKU ID",
                    "Product_Name": "Product Name",
                    "Stock_Qty": st.column_config.NumberColumn("Stock Qty", format="%d"),
                    "Cover_Months": st.column_config.NumberColumn("Cover (Months)", format="%.2f"),
                    "Brand": "Brand",
                    "SKU_Tier": "Tier"
                },
                use_container_width=True,
                height=400
            )
            
            # Summary metrics
            col_hs1, col_hs2, col_hs3 = st.columns(3)
            with col_hs1:
                st.metric("Total High Stock SKUs", len(high_stock_df))
            with col_hs2:
                avg_cover = high_stock_df['Cover_Months'].mean()
                st.metric("Avg Cover Months", f"{avg_cover:.1f}")
            with col_hs3:
                total_stock = high_stock_df['Stock_Qty'].sum()
                st.metric("Total Stock Units", f"{total_stock:,.0f}")
        else:
            st.info("✅ Tidak ada SKU dengan High Stock status")

# --- TAB 3: SKU EVALUATION ---
with tab3:
    st.subheader("📋 SKU Accuracy Evaluation (Last 3 Months)")
    
    if not forecast_metrics_3months:
        st.warning("⚠️ Tidak ada data accuracy untuk evaluasi")
    else:
        sku_accuracy = forecast_metrics_3months.get('sku_accuracy', pd.DataFrame())
        detailed_data = forecast_metrics_3months.get('detailed_data', pd.DataFrame())
        
        if not sku_accuracy.empty and not detailed_data.empty:
            # Separate Under and Over SKUs
            under_skus = detailed_data[detailed_data['Accuracy_Status'] == 'Under']['SKU_ID'].unique()
            over_skus = detailed_data[detailed_data['Accuracy_Status'] == 'Over']['SKU_ID'].unique()
            
            # Create two columns for separate tables
            col_eval1, col_eval2 = st.columns(2)
            
            with col_eval1:
                st.subheader("🔻 UNDER FORECAST SKUs")
                
                if len(under_skus) > 0:
                    under_df = sku_accuracy[sku_accuracy['SKU_ID'].isin(under_skus)].copy()
                    
                    # Add sales data if available
                    if 'Total_Sales_3Months' in under_df.columns:
                        under_df = under_df.sort_values('Total_Sales_3Months', ascending=False)
                    else:
                        under_df = under_df.sort_values('SKU_Accuracy', ascending=True)  # Worst accuracy first
                    
                    st.dataframe(
                        under_df,
                        column_config={
                            "SKU_ID": "SKU ID",
                            "Product_Name": "Product Name",
                            "SKU_Accuracy": st.column_config.ProgressColumn(
                                "Accuracy %",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "Total_Sales_3Months": st.column_config.NumberColumn(
                                "Sales (3 Months)",
                                format="%d"
                            )
                        },
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("✅ Tidak ada SKU dengan UNDER forecast")
            
            with col_eval2:
                st.subheader("🔺 OVER FORECAST SKUs")
                
                if len(over_skus) > 0:
                    over_df = sku_accuracy[sku_accuracy['SKU_ID'].isin(over_skus)].copy()
                    
                    # Add sales data if available
                    if 'Total_Sales_3Months' in over_df.columns:
                        over_df = over_df.sort_values('Total_Sales_3Months', ascending=False)
                    else:
                        over_df = over_df.sort_values('SKU_Accuracy', ascending=True)  # Worst accuracy first
                    
                    st.dataframe(
                        over_df,
                        column_config={
                            "SKU_ID": "SKU ID",
                            "Product_Name": "Product Name",
                            "SKU_Accuracy": st.column_config.ProgressColumn(
                                "Accuracy %",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "Total_Sales_3Months": st.column_config.NumberColumn(
                                "Sales (3 Months)",
                                format="%d"
                            )
                        },
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("✅ Tidak ada SKU dengan OVER forecast")
            
            # Summary
            st.subheader("📊 Evaluation Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Total UNDER SKUs", len(under_skus))
            with col_sum2:
                st.metric("Total OVER SKUs", len(over_skus))
            with col_sum3:
                accurate_skus = detailed_data[detailed_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].unique()
                st.metric("ACCURATE SKUs", len(accurate_skus))

# --- TAB 4: SALES ANALYTICS ---
with tab4:
    st.subheader("📈 Sales Performance Analysis (Active SKUs Only)")
    
    if df_sales.empty:
        st.warning("⚠️ Sales data tidak tersedia")
    else:
        # Filter only Active SKUs
        if not df_product_active.empty:
            active_skus = df_product_active['SKU_ID'].tolist()
            df_sales_active = df_sales[df_sales['SKU_ID'].isin(active_skus)].copy()
        else:
            df_sales_active = df_sales.copy()
        
        if df_sales_active.empty:
            st.info("⚠️ Tidak ada sales data untuk SKU aktif")
        else:
            # Sales summary table
            st.subheader("📊 Monthly Sales Summary")
            
            # Aggregate sales by month
            monthly_sales = df_sales_active.groupby('Month')['Sales_Qty'].sum().reset_index()
            monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%b %Y')
            monthly_sales = monthly_sales.sort_values('Month')
            
            st.dataframe(
                monthly_sales,
                column_config={
                    "Month": "Month",
                    "Sales_Qty": st.column_config.NumberColumn("Sales Qty", format="%d")
                },
                use_container_width=True,
                height=300
            )
            
            # Top performing Active SKUs
            st.subheader("🏆 Top 10 Active SKUs by Sales")
            
            # Total sales by SKU with Product_Name
            sku_sales_total = df_sales_active.groupby('SKU_ID').agg({
                'Sales_Qty': 'sum'
            }).reset_index()
            
            # Add Product_Name if available
            if 'Product_Name' in df_sales_active.columns:
                product_names = df_sales_active[['SKU_ID', 'Product_Name']].drop_duplicates()
                sku_sales_total = pd.merge(sku_sales_total, product_names, on='SKU_ID', how='left')
            
            sku_sales_total = sku_sales_total.sort_values('Sales_Qty', ascending=False).head(10)
            
            if not sku_sales_total.empty:
                st.dataframe(
                    sku_sales_total,
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Product_Name": "Product Name",
                        "Sales_Qty": st.column_config.NumberColumn("Total Sales", format="%d")
                    },
                    use_container_width=True,
                    height=400
                )

# --- TAB 5: DATA EXPLORER ---
with tab5:
    st.subheader("📋 Raw Data Explorer")
    
    # Dataset selection
    dataset_options = {}
    
    if not df_product.empty:
        dataset_options["Product Master (All)"] = df_product
    if not df_product_active.empty:
        dataset_options["Product Master (Active Only)"] = df_product_active
    
    if not df_sales.empty:
        dataset_options["Sales Data"] = df_sales
    
    if not df_forecast.empty:
        dataset_options["Forecast Data"] = df_forecast
    
    if not df_po.empty:
        dataset_options["PO Data"] = df_po
    
    if not df_stock.empty:
        dataset_options["Stock Data"] = df_stock
    
    if dataset_options:
        selected_data = st.selectbox("Select Dataset", list(dataset_options.keys()))
        df_selected = dataset_options[selected_data]
        
        if not df_selected.empty:
            st.write(f"**Rows:** {df_selected.shape[0]}, **Columns:** {df_selected.shape[1]}")
            
            # Data preview
            st.dataframe(
                df_selected,
                use_container_width=True,
                height=400
            )
    else:
        st.warning("⚠️ Tidak ada data yang tersedia")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Inventory Intelligence Dashboard v3.1 | Professional Inventory Control System</p>
    <p>✅ Last 3 Months Analysis | ✅ Monthly Performance | ✅ High Stock Evaluation</p>
</div>
""", unsafe_allow_html=True)
