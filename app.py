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

# --- Custom CSS Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .status-indicator {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .status-indicator:hover {
        transform: translateY(-5px);
    }
    .status-under { 
        background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%);
        color: white;
        border-left: 5px solid #D32F2F;
    }
    .status-accurate { 
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border-left: 5px solid #1B5E20;
    }
    .status-over { 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        border-left: 5px solid #E65100;
    }
    
    .inventory-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .inventory-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .card-replenish { 
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #EF6C00;
        border: 2px solid #FF9800;
    }
    .card-ideal { 
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        border: 2px solid #4CAF50;
    }
    .card-high { 
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        color: #C62828;
        border: 2px solid #F44336;
    }
    
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 10px 0;
    }
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
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
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

def parse_month_label(label):
    """Parse berbagai format bulan ke datetime"""
    try:
        label_str = str(label).strip().upper()
        
        # Mapping bulan
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        for month_name, month_num in month_map.items():
            if month_name in label_str:
                # Cari tahun
                year_part = label_str.replace(month_name, '').replace('-', '').replace(' ', '').strip()
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
    """Load dan proses semua data sekaligus"""
    
    gsheet_url = st.secrets["gsheet_url"]
    data = {}
    
    try:
        # 1. PRODUCT MASTER
        ws = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws.get_all_records())
        df_product.columns = [col.strip().replace(' ', '_') for col in df_product.columns]
        
        # Ensure Status column
        if 'Status' not in df_product.columns:
            df_product['Status'] = 'Active'
        
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        
        # 2. SALES DATA
        ws_sales = _client.open_by_url(gsheet_url).worksheet("Sales")
        df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
        df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
        
        # Process Sales data
        month_cols = [col for col in df_sales_raw.columns if any(m in col.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
        
        if month_cols and 'SKU_ID' in df_sales_raw.columns:
            # Get ID columns
            id_cols = ['SKU_ID']
            for col in ['SKU_Name', 'Product_Name', 'Brand', 'SKU_Tier']:
                if col in df_sales_raw.columns:
                    id_cols.append(col)
            
            # Melt to long format
            df_sales_long = df_sales_raw.melt(
                id_vars=id_cols,
                value_vars=month_cols,
                var_name='Month_Label',
                value_name='Sales_Qty'
            )
            
            df_sales_long['Sales_Qty'] = pd.to_numeric(df_sales_long['Sales_Qty'], errors='coerce').fillna(0)
            df_sales_long['Month'] = df_sales_long['Month_Label'].apply(parse_month_label)
            
            # Filter active SKUs
            active_skus = df_product_active['SKU_ID'].tolist()
            df_sales_long = df_sales_long[df_sales_long['SKU_ID'].isin(active_skus)]
            
            data['sales'] = df_sales_long
        
        # 3. ROFO DATA
        ws_rofo = _client.open_by_url(gsheet_url).worksheet("Rofo")
        df_rofo_raw = pd.DataFrame(ws_rofo.get_all_records())
        df_rofo_raw.columns = [col.strip() for col in df_rofo_raw.columns]
        
        month_cols_rofo = [col for col in df_rofo_raw.columns if any(m in col.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
        
        if month_cols_rofo:
            id_cols_rofo = ['SKU_ID']
            for col in ['Product_Name', 'Brand']:
                if col in df_rofo_raw.columns:
                    id_cols_rofo.append(col)
            
            df_rofo_long = df_rofo_raw.melt(
                id_vars=id_cols_rofo,
                value_vars=month_cols_rofo,
                var_name='Month_Label',
                value_name='Forecast_Qty'
            )
            
            df_rofo_long['Forecast_Qty'] = pd.to_numeric(df_rofo_long['Forecast_Qty'], errors='coerce').fillna(0)
            df_rofo_long['Month'] = df_rofo_long['Month_Label'].apply(parse_month_label)
            df_rofo_long = df_rofo_long[df_rofo_long['SKU_ID'].isin(active_skus)]
            
            data['forecast'] = df_rofo_long
        
        # 4. PO DATA
        ws_po = _client.open_by_url(gsheet_url).worksheet("PO")
        df_po_raw = pd.DataFrame(ws_po.get_all_records())
        df_po_raw.columns = [col.strip() for col in df_po_raw.columns]
        
        month_cols_po = [col for col in df_po_raw.columns if any(m in col.upper() for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
        
        if month_cols_po and 'SKU_ID' in df_po_raw.columns:
            df_po_long = df_po_raw.melt(
                id_vars=['SKU_ID'],
                value_vars=month_cols_po,
                var_name='Month_Label',
                value_name='PO_Qty'
            )
            
            df_po_long['PO_Qty'] = pd.to_numeric(df_po_long['PO_Qty'], errors='coerce').fillna(0)
            df_po_long['Month'] = df_po_long['Month_Label'].apply(parse_month_label)
            df_po_long = df_po_long[df_po_long['SKU_ID'].isin(active_skus)]
            
            data['po'] = df_po_long
        
        # 5. STOCK DATA
        ws_stock = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df_stock_raw = pd.DataFrame(ws_stock.get_all_records())
        df_stock_raw.columns = [col.strip().replace(' ', '_') for col in df_stock_raw.columns]
        
        stock_col = None
        for col in ['Quantity_Available', 'Stock_Qty', 'STOCK_SAP']:
            if col in df_stock_raw.columns:
                stock_col = col
                break
        
        if stock_col and 'SKU_ID' in df_stock_raw.columns:
            df_stock = pd.DataFrame({
                'SKU_ID': df_stock_raw['SKU_ID'],
                'Stock_Qty': pd.to_numeric(df_stock_raw[stock_col], errors='coerce').fillna(0)
            })
            
            df_stock = df_stock.groupby('SKU_ID')['Stock_Qty'].max().reset_index()
            df_stock = df_stock[df_stock['SKU_ID'].isin(active_skus)]
            
            data['stock'] = df_stock
        
        data['product'] = df_product
        data['product_active'] = df_product_active
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return {}

# --- ====================================================== ---
# ---               ANALYTICS FUNCTIONS                    ---
# --- ====================================================== ---

def calculate_forecast_accuracy_3months(df_forecast, df_po, df_product):
    """Calculate forecast accuracy for LAST 3 MONTHS"""
    
    metrics = {}
    
    if df_forecast.empty or df_po.empty:
        return metrics
    
    try:
        # Get unique months from both datasets
        all_months = sorted(set(df_forecast['Month'].tolist() + df_po['Month'].tolist()))
        
        if len(all_months) >= 3:
            # Take last 3 months
            last_3_months = all_months[-3:]
            
            # Filter data for last 3 months
            df_forecast_recent = df_forecast[df_forecast['Month'].isin(last_3_months)].copy()
            df_po_recent = df_po[df_po['Month'].isin(last_3_months)].copy()
            
            # Merge forecast and PO
            df_merged = pd.merge(
                df_forecast_recent,
                df_po_recent,
                on=['SKU_ID', 'Month'],
                how='inner',
                suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                # Add product info
                if not df_product.empty:
                    product_info = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
                    df_merged = pd.merge(df_merged, product_info, on='SKU_ID', how='left')
                
                # Calculate ratios
                df_merged['PO_Rofo_Ratio'] = np.where(
                    df_merged['Forecast_Qty'] > 0,
                    (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
                    0
                )
                
                # Categorize
                conditions = [
                    df_merged['PO_Rofo_Ratio'] < 80,
                    (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
                    df_merged['PO_Rofo_Ratio'] > 120
                ]
                choices = ['Under', 'Accurate', 'Over']
                df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
                
                # Calculate metrics
                df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
                mape = df_merged['Absolute_Percentage_Error'].mean()
                overall_accuracy = 100 - mape
                
                # Status counts
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                # SKU-level accuracy
                sku_accuracy = df_merged.groupby('SKU_ID').apply(
                    lambda x: 100 - x['Absolute_Percentage_Error'].mean()
                ).reset_index()
                sku_accuracy.columns = ['SKU_ID', 'SKU_Accuracy']
                
                # Add product info
                if 'Product_Name' in df_merged.columns:
                    sku_names = df_merged[['SKU_ID', 'Product_Name']].drop_duplicates()
                    sku_accuracy = pd.merge(sku_accuracy, sku_names, on='SKU_ID', how='left')
                
                # Add sales data if available
                if 'sales' in globals():
                    df_sales_filtered = df_sales[df_sales['Month'].isin(last_3_months)]
                    if not df_sales_filtered.empty:
                        sales_by_sku = df_sales_filtered.groupby('SKU_ID')['Sales_Qty'].sum().reset_index()
                        sales_by_sku.columns = ['SKU_ID', 'Sales_Last_3M']
                        sku_accuracy = pd.merge(sku_accuracy, sales_by_sku, on='SKU_ID', how='left')
                
                metrics = {
                    'overall_accuracy': overall_accuracy,
                    'mape': mape,
                    'status_counts': status_counts,
                    'status_percentages': status_percentages,
                    'sku_accuracy': sku_accuracy,
                    'detailed_data': df_merged,
                    'period_months': [m.strftime('%b %Y') for m in last_3_months],
                    'total_records': total_records
                }
        
    except Exception as e:
        st.error(f"Forecast calculation error: {str(e)}")
    
    return metrics

def calculate_monthly_forecast_metrics(df_forecast, df_po):
    """Calculate monthly forecast metrics"""
    
    monthly_data = []
    
    if df_forecast.empty or df_po.empty:
        return pd.DataFrame()
    
    try:
        # Merge data
        df_merged = pd.merge(
            df_forecast,
            df_po,
            on=['SKU_ID', 'Month'],
            how='inner'
        )
        
        if not df_merged.empty:
            # Calculate monthly metrics
            for month in sorted(df_merged['Month'].unique()):
                month_data = df_merged[df_merged['Month'] == month]
                
                # Calculate ratio and accuracy
                month_data['PO_Rofo_Ratio'] = np.where(
                    month_data['Forecast_Qty'] > 0,
                    (month_data['PO_Qty'] / month_data['Forecast_Qty']) * 100,
                    0
                )
                
                month_data['Absolute_Error'] = abs(month_data['PO_Rofo_Ratio'] - 100)
                monthly_accuracy = 100 - month_data['Absolute_Error'].mean()
                
                # Count status
                conditions = [
                    month_data['PO_Rofo_Ratio'] < 80,
                    (month_data['PO_Rofo_Ratio'] >= 80) & (month_data['PO_Rofo_Ratio'] <= 120),
                    month_data['PO_Rofo_Ratio'] > 120
                ]
                choices = ['Under', 'Accurate', 'Over']
                month_data['Status'] = np.select(conditions, choices, default='Unknown')
                
                status_counts = month_data['Status'].value_counts().to_dict()
                
                monthly_data.append({
                    'Month': month,
                    'Month_Formatted': month.strftime('%b %Y'),
                    'Accuracy': monthly_accuracy,
                    'Under': status_counts.get('Under', 0),
                    'Accurate': status_counts.get('Accurate', 0),
                    'Over': status_counts.get('Over', 0),
                    'Total_SKUs': len(month_data)
                })
        
        return pd.DataFrame(monthly_data)
        
    except Exception as e:
        st.error(f"Monthly metrics error: {str(e)}")
        return pd.DataFrame()

def calculate_inventory_metrics(df_stock, df_sales, df_product):
    """Calculate comprehensive inventory metrics"""
    
    metrics = {}
    
    if df_stock.empty:
        return metrics
    
    try:
        # Merge with product info
        df_inventory = pd.merge(
            df_stock,
            df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']],
            on='SKU_ID',
            how='left'
        )
        
        # Calculate cover months if sales data available
        if not df_sales.empty:
            # Calculate average monthly sales
            monthly_sales = df_sales.groupby(['SKU_ID', 'Month']).agg({'Sales_Qty': 'sum'}).reset_index()
            avg_monthly_sales = monthly_sales.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales']
            
            df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
            df_inventory['Avg_Monthly_Sales'] = df_inventory['Avg_Monthly_Sales'].fillna(0)
            
            # Calculate cover months
            df_inventory['Cover_Months'] = np.where(
                df_inventory['Avg_Monthly_Sales'] > 0,
                df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales'],
                999
            )
            
            # Categorize inventory status
            conditions = [
                df_inventory['Cover_Months'] < 0.8,
                (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 1.5),
                df_inventory['Cover_Months'] > 1.5
            ]
            choices = ['Need Replenishment', 'Ideal/Healthy', 'High Stock']
            df_inventory['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
            
            # Get high stock items for reduction
            high_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'High Stock'].copy()
            high_stock_df = high_stock_df.sort_values('Cover_Months', ascending=False)
            
            metrics['high_stock'] = high_stock_df
            metrics['avg_cover'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
        
        # Tier analysis
        if 'SKU_Tier' in df_inventory.columns:
            tier_analysis = df_inventory.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'Stock_Qty': 'sum'
            }).reset_index()
            tier_analysis.columns = ['Tier', 'SKU_Count', 'Total_Stock']
            metrics['tier_analysis'] = tier_analysis
        
        metrics['inventory_df'] = df_inventory
        metrics['total_stock'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        
        return metrics
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
        return metrics

def create_sankey_chart_tier(df_forecast, df_po, df_product):
    """Create Sankey chart for Tier-based analysis"""
    
    if df_forecast.empty or df_po.empty or df_product.empty:
        return None
    
    try:
        # Merge all data
        df_merged = pd.merge(df_forecast, df_po, on=['SKU_ID', 'Month'], how='inner')
        df_merged = pd.merge(df_merged, df_product[['SKU_ID', 'SKU_Tier']], on='SKU_ID', how='left')
        
        # Calculate ratio and categorize
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
        choices = ['Under Forecast', 'Accurate', 'Over Forecast']
        df_merged['Accuracy_Category'] = np.select(conditions, choices, default='Unknown')
        
        # Group by Tier and Category
        tier_flow = df_merged.groupby(['SKU_Tier', 'Accuracy_Category']).agg({
            'SKU_ID': 'count',
            'Forecast_Qty': 'sum',
            'PO_Qty': 'sum'
        }).reset_index()
        
        # Prepare Sankey data
        all_tiers = tier_flow['SKU_Tier'].unique().tolist()
        all_categories = tier_flow['Accuracy_Category'].unique().tolist()
        
        # Nodes: Tiers + Categories
        nodes = all_tiers + all_categories
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        labels = []
        
        for _, row in tier_flow.iterrows():
            source_idx = nodes.index(row['SKU_Tier'])
            target_idx = nodes.index(row['Accuracy_Category'])
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['SKU_ID'])  # Use count of SKUs as flow value
            
            # Create label with details
            label = f"{row['SKU_Tier']} → {row['Accuracy_Category']}<br>SKUs: {row['SKU_ID']}<br>Forecast: {row['Forecast_Qty']:,.0f}<br>PO: {row['PO_Qty']:,.0f}"
            labels.append(label)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=30,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=["#667eea" if node in all_tiers else 
                      "#4CAF50" if node == "Accurate" else 
                      "#FF9800" if node == "Over Forecast" else "#F44336" 
                      for node in nodes]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                label=labels,
                color=["rgba(102, 126, 234, 0.6)" for _ in range(len(source_indices))]
            )
        )])
        
        fig.update_layout(
            title_text="<b>Forecast Accuracy Flow by SKU Tier</b>",
            font_size=12,
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Sankey chart error: {str(e)}")
        return None

# --- ====================================================== ---
# ---               DASHBOARD INITIALIZATION               ---
# --- ====================================================== ---

# Initialize connection
client = init_gsheet_connection()

if client is None:
    st.error("❌ Tidak dapat terhubung ke Google Sheets")
    st.stop()

# Load and process data
with st.spinner('🔄 Loading and processing data from Google Sheets...'):
    all_data = load_and_process_data(client)
    
    df_product = all_data.get('product', pd.DataFrame())
    df_product_active = all_data.get('product_active', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())

# Calculate metrics
forecast_metrics = calculate_forecast_accuracy_3months(df_forecast, df_po, df_product)
monthly_metrics_df = calculate_monthly_forecast_metrics(df_forecast, df_po)
inventory_metrics = calculate_inventory_metrics(df_stock, df_sales, df_product)

# Create Sankey chart
sankey_chart = create_sankey_chart_tier(df_forecast, df_po, df_product)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_sb2:
        if st.button("📊 Show Data Stats", use_container_width=True):
            st.session_state.show_stats = True
    
    st.markdown("---")
    st.markdown("### 📈 Data Overview")
    
    if not df_product_active.empty:
        st.metric("Active SKUs", len(df_product_active))
    
    if not df_stock.empty:
        total_stock = df_stock['Stock_Qty'].sum()
        st.metric("Total Stock", f"{total_stock:,.0f}")
    
    if forecast_metrics:
        accuracy = forecast_metrics.get('overall_accuracy', 0)
        st.metric("Forecast Accuracy", f"{accuracy:.1f}%")

# --- MAIN DASHBOARD ---

# HEADER METRICS - LAST 3 MONTHS
st.subheader("🎯 Forecast Accuracy - Last 3 Months (PO vs Rofo)")

if forecast_metrics and forecast_metrics.get('total_records', 0) > 0:
    # Display period
    period_months = forecast_metrics.get('period_months', [])
    if period_months:
        st.caption(f"**Analysis Period:** {', '.join(period_months)}")
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    with col_h1:
        under_pct = forecast_metrics.get('status_percentages', {}).get('Under', 0)
        st.markdown(f"""
        <div class="status-indicator status-under">
            <div style="font-size: 1.1rem; font-weight: 800;">UNDER FORECAST</div>
            <div style="font-size: 2.2rem; font-weight: 900;">{under_pct:.1f}%</div>
            <div style="font-size: 0.9rem;">PO < 80% of Rofo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h2:
        accurate_pct = forecast_metrics.get('status_percentages', {}).get('Accurate', 0)
        st.markdown(f"""
        <div class="status-indicator status-accurate">
            <div style="font-size: 1.1rem; font-weight: 800;">ACCURATE</div>
            <div style="font-size: 2.2rem; font-weight: 900;">{accurate_pct:.1f}%</div>
            <div style="font-size: 0.9rem;">80-120% of Rofo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h3:
        over_pct = forecast_metrics.get('status_percentages', {}).get('Over', 0)
        st.markdown(f"""
        <div class="status-indicator status-over">
            <div style="font-size: 1.1rem; font-weight: 800;">OVER FORECAST</div>
            <div style="font-size: 2.2rem; font-weight: 900;">{over_pct:.1f}%</div>
            <div style="font-size: 0.9rem;">PO > 120% of Rofo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h4:
        overall_acc = forecast_metrics.get('overall_accuracy', 0)
        mape_val = forecast_metrics.get('mape', 0)
        st.markdown(f"""
        <div class="metric-highlight">
            <div style="font-size: 0.9rem; color: #666;">OVERALL ACCURACY</div>
            <div style="font-size: 2rem; font-weight: 900; color: #667eea;">{overall_acc:.1f}%</div>
            <div style="font-size: 0.8rem; color: #888;">MAPE: {mape_val:.1f}%</div>
            <div style="font-size: 0.8rem; color: #888;">{forecast_metrics.get('total_records', 0)} records</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Insufficient data for last 3 months analysis")
    
    # Show available data info
    if not df_forecast.empty and not df_po.empty:
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        
        with st.expander("📊 Available Data Info"):
            st.write(f"**Forecast Months:** {len(forecast_months)} months")
            st.write(f"**PO Months:** {len(po_months)} months")
            
            if forecast_months:
                st.write("Latest forecast month:", forecast_months[-1].strftime('%b %Y'))
            if po_months:
                st.write("Latest PO month:", po_months[-1].strftime('%b %Y'))

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Monthly Performance",
    "📊 Tier Analysis",
    "📦 Inventory Health",
    "🔍 SKU Evaluation",
    "📈 Sales Analytics",
    "📋 Data Explorer"
])

# --- TAB 1: MONTHLY PERFORMANCE ---
with tab1:
    st.subheader("📅 Monthly Forecast Performance")
    
    if not monthly_metrics_df.empty:
        # Monthly accuracy chart
        col_m1, col_m2 = st.columns([2, 1])
        
        with col_m1:
            # Line chart for accuracy trend
            line_chart = alt.Chart(monthly_metrics_df).mark_line(point=True, size=3).encode(
                x=alt.X('Month_Formatted:N', title='Month', sort='ascending'),
                y=alt.Y('Accuracy:Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Month_Formatted', 'Accuracy']
            ).properties(height=400, title="Monthly Accuracy Trend")
            
            st.altair_chart(line_chart, use_container_width=True)
        
        with col_m2:
            # Monthly summary table
            st.dataframe(
                monthly_metrics_df[['Month_Formatted', 'Accuracy', 'Under', 'Accurate', 'Over', 'Total_SKUs']],
                column_config={
                    "Month_Formatted": "Month",
                    "Accuracy": st.column_config.ProgressColumn("Accuracy %", format="%.1f%%", min_value=0, max_value=100),
                    "Under": "Under",
                    "Accurate": "Accurate", 
                    "Over": "Over",
                    "Total_SKUs": "Total SKUs"
                },
                use_container_width=True,
                height=400
            )
        
        # Stacked bar chart for status distribution
        st.subheader("📊 Monthly Status Distribution")
        
        # Prepare data for stacked bar
        status_melt = monthly_metrics_df.melt(
            id_vars=['Month_Formatted'],
            value_vars=['Under', 'Accurate', 'Over'],
            var_name='Status',
            value_name='Count'
        )
        
        bars = alt.Chart(status_melt).mark_bar().encode(
            x=alt.X('Month_Formatted:N', title='Month', sort='ascending'),
            y=alt.Y('Count:Q', title='Number of SKUs'),
            color=alt.Color('Status:N', scale=alt.Scale(
                domain=['Under', 'Accurate', 'Over'],
                range=['#F44336', '#4CAF50', '#FF9800']
            )),
            tooltip=['Month_Formatted', 'Status', 'Count']
        ).properties(height=300)
        
        st.altair_chart(bars, use_container_width=True)
        
    else:
        st.info("📊 No monthly performance data available")

# --- TAB 2: TIER ANALYSIS ---
with tab2:
    st.subheader("🏷️ SKU Tier Analysis")
    
    if sankey_chart:
        st.markdown('<div class="sankey-container">', unsafe_allow_html=True)
        st.plotly_chart(sankey_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tier-wise metrics
    if forecast_metrics and 'detailed_data' in forecast_metrics:
        df_detailed = forecast_metrics['detailed_data']
        
        if 'SKU_Tier' in df_detailed.columns:
            # Tier accuracy analysis
            tier_accuracy = df_detailed.groupby('SKU_Tier').apply(
                lambda x: 100 - abs(x['PO_Rofo_Ratio'] - 100).mean()
            ).reset_index()
            tier_accuracy.columns = ['Tier', 'Accuracy']
            
            tier_summary = df_detailed.groupby(['SKU_Tier', 'Accuracy_Status']).size().unstack(fill_value=0)
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("📊 Tier Accuracy")
                st.dataframe(
                    tier_accuracy.sort_values('Accuracy', ascending=False),
                    column_config={
                        "Tier": "SKU Tier",
                        "Accuracy": st.column_config.ProgressColumn("Accuracy %", format="%.1f%%")
                    },
                    use_container_width=True,
                    height=300
                )
            
            with col_t2:
                st.subheader("📋 Tier Status Distribution")
                st.dataframe(
                    tier_summary,
                    use_container_width=True,
                    height=300
                )
    
    # Inventory by tier
    if 'tier_analysis' in inventory_metrics:
        tier_stock = inventory_metrics['tier_analysis']
        
        st.subheader("📦 Stock Distribution by Tier")
        
        # Bar chart
        bars_tier = alt.Chart(tier_stock).mark_bar().encode(
            x=alt.X('Tier:N', title='SKU Tier'),
            y=alt.Y('Total_Stock:Q', title='Total Stock'),
            color=alt.Color('Tier:N', scale=alt.Scale(scheme='set2')),
            tooltip=['Tier', 'SKU_Count', 'Total_Stock']
        ).properties(height=350)
        
        st.altair_chart(bars_tier, use_container_width=True)

# --- TAB 3: INVENTORY HEALTH ---
with tab3:
    st.subheader("📦 Inventory Health Dashboard")
    
    if inventory_metrics:
        # Inventory status cards
        inv_df = inventory_metrics.get('inventory_df', pd.DataFrame())
        
        if 'Inventory_Status' in inv_df.columns:
            status_counts = inv_df['Inventory_Status'].value_counts().to_dict()
            total_skus = len(inv_df)
            
            col_i1, col_i2, col_i3 = st.columns(3)
            
            with col_i1:
                need_count = status_counts.get('Need Replenishment', 0)
                need_pct = (need_count / total_skus * 100) if total_skus > 0 else 0
                st.markdown(f"""
                <div class="inventory-card card-replenish">
                    <div style="font-size: 1rem; font-weight: 800;">NEED REPLENISHMENT</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{need_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">{need_count} SKUs (Cover < 0.8 months)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_i2:
                ideal_count = status_counts.get('Ideal/Healthy', 0)
                ideal_pct = (ideal_count / total_skus * 100) if total_skus > 0 else 0
                st.markdown(f"""
                <div class="inventory-card card-ideal">
                    <div style="font-size: 1rem; font-weight: 800;">IDEAL/HEALTHY</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{ideal_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">{ideal_count} SKUs (0.8-1.5 months)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_i3:
                high_count = status_counts.get('High Stock', 0)
                high_pct = (high_count / total_skus * 100) if total_skus > 0 else 0
                st.markdown(f"""
                <div class="inventory-card card-high">
                    <div style="font-size: 1rem; font-weight: 800;">HIGH STOCK</div>
                    <div style="font-size: 1.8rem; font-weight: 900;">{high_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">{high_count} SKUs (Cover > 1.5 months)</div>
                </div>
                """, unsafe_allow_html=True)
        
        # High Stock SKUs for reduction
        st.subheader("📉 High Stock SKUs (Need Reduction)")
        
        high_stock_df = inventory_metrics.get('high_stock', pd.DataFrame())
        if not high_stock_df.empty:
            # Display with sorting options
            sort_option = st.selectbox(
                "Sort by",
                ["Cover Months (Highest First)", "Stock Quantity (Highest First)", "SKU Tier"]
            )
            
            if sort_option == "Cover Months (Highest First)":
                high_stock_df = high_stock_df.sort_values('Cover_Months', ascending=False)
            elif sort_option == "Stock Quantity (Highest First)":
                high_stock_df = high_stock_df.sort_values('Stock_Qty', ascending=False)
            else:
                high_stock_df = high_stock_df.sort_values(['SKU_Tier', 'Cover_Months'], ascending=[True, False])
            
            # Display table
            display_cols = ['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Stock_Qty', 'Cover_Months']
            available_cols = [col for col in display_cols if col in high_stock_df.columns]
            
            st.dataframe(
                high_stock_df[available_cols],
                column_config={
                    "SKU_ID": "SKU ID",
                    "Product_Name": "Product Name",
                    "SKU_Tier": "Tier",
                    "Brand": "Brand",
                    "Stock_Qty": st.column_config.NumberColumn("Stock Qty", format="%d"),
                    "Cover_Months": st.column_config.NumberColumn("Cover (Months)", format="%.2f")
                },
                use_container_width=True,
                height=400
            )
            
            # Summary
            col_hs1, col_hs2, col_hs3 = st.columns(3)
            with col_hs1:
                st.metric("Total High Stock SKUs", len(high_stock_df))
            with col_hs2:
                avg_cover = high_stock_df['Cover_Months'].mean()
                st.metric("Average Cover", f"{avg_cover:.1f} months")
            with col_hs3:
                total_qty = high_stock_df['Stock_Qty'].sum()
                st.metric("Total Stock Units", f"{total_qty:,.0f}")
        else:
            st.success("✅ No SKUs with High Stock status")
        
        # Overall inventory metrics
        st.subheader("📊 Inventory Overview")
        col_io1, col_io2, col_io3 = st.columns(3)
        with col_io1:
            st.metric("Total SKUs", inventory_metrics.get('total_skus', 0))
        with col_io2:
            st.metric("Total Stock", f"{inventory_metrics.get('total_stock', 0):,.0f}")
        with col_io3:
            avg_cover_all = inventory_metrics.get('avg_cover', 0)
            st.metric("Avg Cover Months", f"{avg_cover_all:.1f}")

# --- TAB 4: SKU EVALUATION ---
with tab4:
    st.subheader("🔍 SKU Performance Evaluation")
    
    if forecast_metrics and 'sku_accuracy' in forecast_metrics:
        sku_accuracy_df = forecast_metrics['sku_accuracy']
        detailed_df = forecast_metrics['detailed_data']
        
        # Separate Under and Over SKUs
        under_skus = detailed_df[detailed_df['Accuracy_Status'] == 'Under']['SKU_ID'].unique()
        over_skus = detailed_df[detailed_df['Accuracy_Status'] == 'Over']['SKU_ID'].unique()
        
        # Create tabs for Under and Over
        eval_tab1, eval_tab2 = st.tabs(["📉 UNDER Forecast SKUs", "📈 OVER Forecast SKUs"])
        
        with eval_tab1:
            if len(under_skus) > 0:
                under_df = sku_accuracy_df[sku_accuracy_df['SKU_ID'].isin(under_skus)].copy()
                
                # Add detailed metrics
                under_detailed = detailed_df[detailed_df['SKU_ID'].isin(under_skus)]
                under_summary = under_detailed.groupby('SKU_ID').agg({
                    'Forecast_Qty': 'sum',
                    'PO_Qty': 'sum',
                    'PO_Rofo_Ratio': 'mean'
                }).reset_index()
                
                under_df = pd.merge(under_df, under_summary, on='SKU_ID', how='left')
                under_df = under_df.sort_values('SKU_Accuracy', ascending=True)  # Worst first
                
                st.dataframe(
                    under_df,
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Product_Name": "Product Name",
                        "SKU_Accuracy": st.column_config.ProgressColumn("Accuracy %", format="%.1f%%"),
                        "Forecast_Qty": st.column_config.NumberColumn("Forecast Total"),
                        "PO_Qty": st.column_config.NumberColumn("PO Total"),
                        "PO_Rofo_Ratio": st.column_config.NumberColumn("Avg Ratio %", format="%.1f")
                    },
                    use_container_width=True,
                    height=500
                )
            else:
                st.success("✅ No SKUs with UNDER forecast")
        
        with eval_tab2:
            if len(over_skus) > 0:
                over_df = sku_accuracy_df[sku_accuracy_df['SKU_ID'].isin(over_skus)].copy()
                
                # Add detailed metrics
                over_detailed = detailed_df[detailed_df['SKU_ID'].isin(over_skus)]
                over_summary = over_detailed.groupby('SKU_ID').agg({
                    'Forecast_Qty': 'sum',
                    'PO_Qty': 'sum',
                    'PO_Rofo_Ratio': 'mean'
                }).reset_index()
                
                over_df = pd.merge(over_df, over_summary, on='SKU_ID', how='left')
                over_df = over_df.sort_values('SKU_Accuracy', ascending=True)  # Worst first
                
                st.dataframe(
                    over_df,
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Product_Name": "Product Name",
                        "SKU_Accuracy": st.column_config.ProgressColumn("Accuracy %", format="%.1f%%"),
                        "Forecast_Qty": st.column_config.NumberColumn("Forecast Total"),
                        "PO_Qty": st.column_config.NumberColumn("PO Total"),
                        "PO_Rofo_Ratio": st.column_config.NumberColumn("Avg Ratio %", format="%.1f")
                    },
                    use_container_width=True,
                    height=500
                )
            else:
                st.success("✅ No SKUs with OVER forecast")
        
        # Summary metrics
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("UNDER Forecast SKUs", len(under_skus))
        with col_e2:
            st.metric("OVER Forecast SKUs", len(over_skus))
        with col_e3:
            accurate_skus = detailed_df[detailed_df['Accuracy_Status'] == 'Accurate']['SKU_ID'].unique()
            st.metric("ACCURATE SKUs", len(accurate_skus))

# --- TAB 5: SALES ANALYTICS ---
with tab5:
    st.subheader("📈 Sales Performance Analytics")
    
    if not df_sales.empty:
        # Monthly sales trend
        monthly_sales = df_sales.groupby('Month').agg({
            'Sales_Qty': 'sum',
            'SKU_ID': 'nunique'
        }).reset_index()
        monthly_sales['Month_Formatted'] = monthly_sales['Month'].dt.strftime('%b %Y')
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            # Sales trend chart
            sales_chart = alt.Chart(monthly_sales).mark_line(point=True, size=3).encode(
                x=alt.X('Month_Formatted:N', title='Month', sort='ascending'),
                y=alt.Y('Sales_Qty:Q', title='Total Sales'),
                tooltip=['Month_Formatted', 'Sales_Qty', 'SKU_ID']
            ).properties(height=350, title="Monthly Sales Trend")
            
            st.altair_chart(sales_chart, use_container_width=True)
        
        with col_s2:
            # Top performing SKUs
            top_skus = df_sales.groupby(['SKU_ID', 'Product_Name']).agg({
                'Sales_Qty': 'sum',
                'Month': 'nunique'
            }).reset_index()
            top_skus = top_skus.sort_values('Sales_Qty', ascending=False).head(10)
            
            st.dataframe(
                top_skus,
                column_config={
                    "SKU_ID": "SKU ID",
                    "Product_Name": "Product Name",
                    "Sales_Qty": st.column_config.NumberColumn("Total Sales"),
                    "Month": "Months Active"
                },
                use_container_width=True,
                height=350
            )
        
        # Sales by Tier if available
        if 'SKU_Tier' in df_sales.columns:
            tier_sales = df_sales.groupby('SKU_Tier').agg({
                'Sales_Qty': 'sum',
                'SKU_ID': 'nunique'
            }).reset_index()
            tier_sales = tier_sales.sort_values('Sales_Qty', ascending=False)
            
            st.subheader("🏷️ Sales by SKU Tier")
            
            bars_tier_sales = alt.Chart(tier_sales).mark_bar().encode(
                x=alt.X('SKU_Tier:N', title='SKU Tier', sort='-y'),
                y=alt.Y('Sales_Qty:Q', title='Total Sales'),
                color=alt.Color('SKU_Tier:N', scale=alt.Scale(scheme='set2')),
                tooltip=['SKU_Tier', 'Sales_Qty', 'SKU_ID']
            ).properties(height=300)
            
            st.altair_chart(bars_tier_sales, use_container_width=True)

# --- TAB 6: DATA EXPLORER ---
with tab6:
    st.subheader("📋 Raw Data Explorer")
    
    dataset_options = {
        "Product Master": df_product,
        "Active Products": df_product_active,
        "Sales Data": df_sales,
        "Forecast Data": df_forecast,
        "PO Data": df_po,
        "Stock Data": df_stock
    }
    
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
    df_selected = dataset_options[selected_dataset]
    
    if not df_selected.empty:
        # Data info
        st.write(f"**Rows:** {df_selected.shape[0]:,} | **Columns:** {df_selected.shape[1]}")
        
        # Column selector
        if st.checkbox("Select Columns", False):
            all_columns = df_selected.columns.tolist()
            selected_columns = st.multiselect("Choose columns:", all_columns, default=all_columns[:10])
            df_display = df_selected[selected_columns]
        else:
            df_display = df_selected
        
        # Data preview
        st.dataframe(
            df_display,
            use_container_width=True,
            height=500
        )
        
        # Download option
        csv = df_selected.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_dataset.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No data available for selected dataset")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🚀 <strong>Inventory Intelligence Dashboard v4.0</strong> | Professional Inventory Control & Demand Planning</p>
    <p>✅ Last 3 Months Analysis | ✅ SKU Tier Sankey Chart | ✅ Monthly Performance Tracking | ✅ High Stock Evaluation</p>
</div>
""", unsafe_allow_html=True)
