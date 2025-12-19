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
    
    /* New CSS */
    .monthly-performance-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    
    .performance-under { border-left-color: #F44336; }
    .performance-accurate { border-left-color: #4CAF50; }
    .performance-over { border-left-color: #FF9800; }
    
    .highlight-row {
        background-color: #FFF9C4 !important;
        font-weight: bold !important;
    }
    
    .warning-badge {
        background: #FF5252;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .success-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Compact metrics */
    .compact-metric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    /* Brand performance */
    .brand-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
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

def calculate_monthly_performance(df_forecast, df_po, df_product):
    """Calculate performance for each month separately"""
    
    monthly_performance = {}
    
    if df_forecast.empty or df_po.empty:
        return monthly_performance
    
    try:
        # Get unique months from both datasets
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        all_months = sorted(set(list(forecast_months) + list(po_months)))
        
        for month in all_months:
            # Get data for this month
            df_forecast_month = df_forecast[df_forecast['Month'] == month].copy()
            df_po_month = df_po[df_po['Month'] == month].copy()
            
            if df_forecast_month.empty or df_po_month.empty:
                continue
            
            # Merge forecast and PO for this month
            df_merged = pd.merge(
                df_forecast_month,
                df_po_month,
                on=['SKU_ID'],
                how='inner',
                suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                # Add product info
                if not df_product.empty:
                    product_info = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
                    df_merged = pd.merge(df_merged, product_info, on='SKU_ID', how='left')
                
                # Calculate ratio
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
                monthly_accuracy = 100 - mape
                
                # Status counts
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                # Store results
                monthly_performance[month] = {
                    'accuracy': monthly_accuracy,
                    'mape': mape,
                    'status_counts': status_counts,
                    'status_percentages': status_percentages,
                    'total_records': total_records,
                    'data': df_merged,
                    'under_skus': df_merged[df_merged['Accuracy_Status'] == 'Under'].copy(),
                    'over_skus': df_merged[df_merged['Accuracy_Status'] == 'Over'].copy(),
                    'accurate_skus': df_merged[df_merged['Accuracy_Status'] == 'Accurate'].copy()
                }
        
        return monthly_performance
        
    except Exception as e:
        st.error(f"Monthly performance calculation error: {str(e)}")
        return monthly_performance

def get_last_3_months_performance(monthly_performance):
    """Get performance for last 3 months"""
    
    if not monthly_performance:
        return {}
    
    # Get last 3 months
    sorted_months = sorted(monthly_performance.keys())
    if len(sorted_months) >= 3:
        last_3_months = sorted_months[-3:]
    else:
        last_3_months = sorted_months
    
    last_3_data = {}
    for month in last_3_months:
        last_3_data[month] = monthly_performance[month]
    
    return last_3_data

def calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product):
    """Calculate inventory metrics using 3-month average sales"""
    
    metrics = {}
    
    if df_stock.empty or df_sales.empty:
        return metrics
    
    try:
        # Get last 3 months sales data
        if not df_sales.empty:
            sales_months = sorted(df_sales['Month'].unique())
            if len(sales_months) >= 3:
                last_3_sales_months = sales_months[-3:]
                df_sales_last_3 = df_sales[df_sales['Month'].isin(last_3_sales_months)].copy()
            else:
                df_sales_last_3 = df_sales.copy()
        
        # Calculate average monthly sales per SKU for last 3 months
        if not df_sales_last_3.empty:
            avg_monthly_sales = df_sales_last_3.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales_3M']
        
        # Merge with product info
        df_inventory = pd.merge(
            df_stock,
            df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']],
            on='SKU_ID',
            how='left'
        )
        
        # Merge with average sales
        df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
        df_inventory['Avg_Monthly_Sales_3M'] = df_inventory['Avg_Monthly_Sales_3M'].fillna(0)
        
        # Calculate cover months using 3-month average
        df_inventory['Cover_Months'] = np.where(
            df_inventory['Avg_Monthly_Sales_3M'] > 0,
            df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales_3M'],
            999  # For SKUs with no sales in last 3 months
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
        
        # Get low stock items
        low_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'Need Replenishment'].copy()
        low_stock_df = low_stock_df.sort_values('Cover_Months', ascending=True)
        
        # Tier analysis
        if 'SKU_Tier' in df_inventory.columns:
            tier_analysis = df_inventory.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'Stock_Qty': 'sum',
                'Avg_Monthly_Sales_3M': 'sum',
                'Cover_Months': 'mean'
            }).reset_index()
            tier_analysis.columns = ['Tier', 'SKU_Count', 'Total_Stock', 'Total_Sales_3M_Avg', 'Avg_Cover_Months']
            tier_analysis['Turnover'] = tier_analysis['Total_Sales_3M_Avg'] / tier_analysis['Total_Stock']
            metrics['tier_analysis'] = tier_analysis
        
        metrics['inventory_df'] = df_inventory
        metrics['high_stock'] = high_stock_df
        metrics['low_stock'] = low_stock_df
        metrics['total_stock'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        metrics['avg_cover'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
        
        # Calculate inventory value metrics
        metrics['inventory_value_score'] = (len(df_inventory[df_inventory['Inventory_Status'] == 'Ideal/Healthy']) / 
                                          len(df_inventory) * 100) if len(df_inventory) > 0 else 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
        return metrics

def calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product):
    """Calculate sales vs forecast and PO comparison"""
    
    results = {}
    
    if df_sales.empty or df_forecast.empty:
        return results
    
    try:
        # Get last 3 months for comparison
        sales_months = sorted(df_sales['Month'].unique())
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        
        # Find common months
        common_months = sorted(set(sales_months) & set(forecast_months) & set(po_months))
        
        if not common_months:
            return results
        
        # Use last common month
        last_month = common_months[-1]
        
        # Get data for last month
        df_sales_month = df_sales[df_sales['Month'] == last_month].copy()
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Merge all data
        df_merged = pd.merge(
            df_sales_month[['SKU_ID', 'Sales_Qty']],
            df_forecast_month[['SKU_ID', 'Forecast_Qty']],
            on='SKU_ID',
            how='inner'
        )
        
        df_merged = pd.merge(
            df_merged,
            df_po_month[['SKU_ID', 'PO_Qty']],
            on='SKU_ID',
            how='left'
        )
        
        # Add product info
        if not df_product.empty:
            product_info = df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand']].drop_duplicates()
            df_merged = pd.merge(df_merged, product_info, on='SKU_ID', how='left')
        
        # Calculate ratios
        df_merged['Sales_vs_Forecast_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        df_merged['Sales_vs_PO_Ratio'] = np.where(
            df_merged['PO_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['PO_Qty']) * 100,
            0
        )
        
        # Calculate deviations
        df_merged['Forecast_Deviation'] = abs(df_merged['Sales_vs_Forecast_Ratio'] - 100)
        df_merged['PO_Deviation'] = abs(df_merged['Sales_vs_PO_Ratio'] - 100)
        
        # Identify SKUs with high deviation (> 30%)
        high_deviation_skus = df_merged[
            (df_merged['Forecast_Deviation'] > 30) | 
            (df_merged['PO_Deviation'] > 30)
        ].copy()
        
        high_deviation_skus = high_deviation_skus.sort_values('Forecast_Deviation', ascending=False)
        
        # Calculate overall metrics
        avg_forecast_deviation = df_merged['Forecast_Deviation'].mean()
        avg_po_deviation = df_merged['PO_Deviation'].mean()
        
        results = {
            'last_month': last_month,
            'comparison_data': df_merged,
            'high_deviation_skus': high_deviation_skus,
            'avg_forecast_deviation': avg_forecast_deviation,
            'avg_po_deviation': avg_po_deviation,
            'total_skus_compared': len(df_merged)
        }
        
        return results
        
    except Exception as e:
        st.error(f"Sales vs forecast calculation error: {str(e)}")
        return results

def calculate_brand_performance(df_forecast, df_po, df_product):
    """Calculate forecast accuracy performance by brand"""
    
    if df_forecast.empty or df_po.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # Get last month data
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return pd.DataFrame()
        
        last_month = common_months[-1]
        
        # Get data for last month
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Merge forecast and PO
        df_merged = pd.merge(
            df_forecast_month,
            df_po_month,
            on=['SKU_ID'],
            how='inner'
        )
        
        # Add brand info
        if 'Brand' not in df_merged.columns and 'Brand' in df_product.columns:
            brand_info = df_product[['SKU_ID', 'Brand']].drop_duplicates()
            df_merged = pd.merge(df_merged, brand_info, on='SKU_ID', how='left')
        
        if 'Brand' not in df_merged.columns:
            return pd.DataFrame()
        
        # Calculate ratio and accuracy
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
        
        # Calculate brand performance
        brand_performance = df_merged.groupby('Brand').agg({
            'SKU_ID': 'count',
            'Forecast_Qty': 'sum',
            'PO_Qty': 'sum',
            'PO_Rofo_Ratio': lambda x: 100 - abs(x - 100).mean()  # Accuracy
        }).reset_index()
        
        brand_performance.columns = ['Brand', 'SKU_Count', 'Total_Forecast', 'Total_PO', 'Accuracy']
        
        # Calculate additional metrics
        brand_performance['PO_vs_Forecast_Ratio'] = (brand_performance['Total_PO'] / brand_performance['Total_Forecast'] * 100)
        brand_performance['Qty_Difference'] = brand_performance['Total_PO'] - brand_performance['Total_Forecast']
        
        # Get status counts
        status_counts = df_merged.groupby(['Brand', 'Accuracy_Status']).size().unstack(fill_value=0).reset_index()
        
        # Merge with performance data
        brand_performance = pd.merge(brand_performance, status_counts, on='Brand', how='left')
        
        # Fill NaN with 0 for status columns
        for status in ['Under', 'Accurate', 'Over']:
            if status not in brand_performance.columns:
                brand_performance[status] = 0
        
        # Sort by accuracy
        brand_performance = brand_performance.sort_values('Accuracy', ascending=False)
        
        return brand_performance
        
    except Exception as e:
        st.error(f"Brand performance calculation error: {str(e)}")
        return pd.DataFrame()

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
monthly_performance = calculate_monthly_performance(df_forecast, df_po, df_product)
last_3_months_performance = get_last_3_months_performance(monthly_performance)
inventory_metrics = calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product)
sales_vs_forecast = calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product)

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
    
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        accuracy = monthly_performance[last_month]['accuracy']
        st.metric("Latest Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    # Threshold Settings
    st.markdown("### ⚙️ Threshold Settings")
    under_threshold = st.slider("Under Forecast Threshold (%)", 0, 100, 80)
    over_threshold = st.slider("Over Forecast Threshold (%)", 100, 200, 120)
    
    st.markdown("---")
    
    # Inventory Thresholds
    st.markdown("### 📦 Inventory Thresholds")
    low_stock_threshold = st.slider("Low Stock (months)", 0.0, 2.0, 0.8, 0.1)
    high_stock_threshold = st.slider("High Stock (months)", 1.0, 6.0, 1.5, 0.1)

# --- MAIN DASHBOARD ---

# SECTION 1: LAST 3 MONTHS PERFORMANCE (DIPERBESAR)
st.subheader("🎯 Forecast Performance - 3 Bulan Terakhir")

if last_3_months_performance:
    # Display last 3 months performance - INI DIPERBESAR
    months_display = []
    
    # Create container untuk 3 bulan
    month_cols = st.columns(3)
    
    for i, (month, data) in enumerate(sorted(last_3_months_performance.items())):
        month_name = month.strftime('%b %Y')
        accuracy = data['accuracy']
        
        with month_cols[i]:
            st.markdown(f"""
            <div style="
                background: white; 
                border-radius: 15px; 
                padding: 1.5rem; 
                margin: 0.5rem 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                border-top: 5px solid #667eea;
            ">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #333;">{month_name}</h3>
                    <div style="font-size: 2rem; font-weight: 900; color: #667eea;">
                        {accuracy:.1f}%
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">Overall Accuracy</div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 1rem;">
                    <div style="text-align: center; padding: 0.5rem; background: #FFEBEE; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: #F44336;">{data['status_counts'].get('Under', 0)}</div>
                        <div style="font-size: 0.8rem; color: #F44336;">Under</div>
                    </div>
                    <div style="text-align: center; padding: 0.5rem; background: #E8F5E9; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: #4CAF50;">{data['status_counts'].get('Accurate', 0)}</div>
                        <div style="font-size: 0.8rem; color: #4CAF50;">Accurate</div>
                    </div>
                    <div style="text-align: center; padding: 0.5rem; background: #FFF3E0; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: #FF9800;">{data['status_counts'].get('Over', 0)}</div>
                        <div style="font-size: 0.8rem; color: #FF9800;">Over</div>
                    </div>
                </div>
                
                <div style="text-align: center; font-size: 0.9rem; color: #666;">
                    Total SKUs: {data['total_records']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        months_display.append(month_name)
    
    # Overall metrics for last 3 months - INI DIKECILKAN
    st.divider()
    
    # Calculate aggregated metrics for last 3 months
    total_under = sum(data['status_counts'].get('Under', 0) for data in last_3_months_performance.values())
    total_accurate = sum(data['status_counts'].get('Accurate', 0) for data in last_3_months_performance.values())
    total_over = sum(data['status_counts'].get('Over', 0) for data in last_3_months_performance.values())
    total_skus = sum(data['total_records'] for data in last_3_months_performance.values())
    
    avg_accuracy = np.mean([data['accuracy'] for data in last_3_months_performance.values()])
    
    # TOTAL ROFO DAN PO BULAN TERAKHIR (TAMBAHAN BARU)
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        total_rofo_last_month = last_month_data['Forecast_Qty'].sum()
        total_po_last_month = last_month_data['PO_Qty'].sum()
        selisih_qty = total_po_last_month - total_rofo_last_month
        selisih_persen = (selisih_qty / total_rofo_last_month * 100) if total_rofo_last_month > 0 else 0
    
    # Layout baru untuk summary metrics (lebih kecil)
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        under_pct = (total_under / total_skus * 100) if total_skus > 0 else 0
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">TOTAL UNDER</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #F44336;">{total_under}</div>
            <div style="font-size: 0.8rem; color: #888;">{under_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum2:
        accurate_pct = (total_accurate / total_skus * 100) if total_skus > 0 else 0
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">TOTAL ACCURATE</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #4CAF50;">{total_accurate}</div>
            <div style="font-size: 0.8rem; color: #888;">{accurate_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum3:
        over_pct = (total_over / total_skus * 100) if total_skus > 0 else 0
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">TOTAL OVER</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #FF9800;">{total_over}</div>
            <div style="font-size: 0.8rem; color: #888;">{over_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum4:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">AVG ACCURACY</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">{avg_accuracy:.1f}%</div>
            <div style="font-size: 0.8rem; color: #888;">{total_skus} records</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ROW BARU UNTUK TOTAL ROFO, PO, SELISIH (TAMBAHAN)
    st.divider()
    st.subheader("📊 Total Rofo vs PO - Bulan Terakhir")
    
    rofo_col1, rofo_col2, rofo_col3, rofo_col4 = st.columns(4)
    
    with rofo_col1:
        st.metric(
            "Total Rofo Qty",
            f"{total_rofo_last_month:,.0f}",
            help="Total quantity dari forecast/Rofo bulan terakhir"
        )
    
    with rofo_col2:
        st.metric(
            "Total PO Qty", 
            f"{total_po_last_month:,.0f}",
            help="Total quantity dari Purchase Order bulan terakhir"
        )
    
    with rofo_col3:
        st.metric(
            "Selisih Qty",
            f"{selisih_qty:+,.0f}",
            f"{selisih_persen:+.1f}%",
            delta_color="normal" if abs(selisih_persen) < 20 else "off",
            help="Selisih antara PO dan Rofo (POSITIVE = PO > Rofo)"
        )
    
    with rofo_col4:
        accurate_count = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].nunique()
        total_count = last_month_data['SKU_ID'].nunique()
        accurate_pct_last_month = (accurate_count / total_count * 100) if total_count > 0 else 0
        st.metric(
            "Accurate SKUs",
            f"{accurate_count}",
            f"{accurate_pct_last_month:.1f}%",
            help="Jumlah SKU dengan accuracy 80-120% di bulan terakhir"
        )

else:
    st.warning("⚠️ Insufficient data for monthly performance analysis")

st.divider()

# SECTION 2: LAST MONTH EVALUATION (UNDER & OVER ONLY)
st.subheader("📋 Evaluasi Rofo - Bulan Terakhir (Under & Over Forecast)")

if monthly_performance:
    # Get last month data
    sorted_months = sorted(monthly_performance.keys())
    if sorted_months:
        last_month = sorted_months[-1]
        last_month_data = monthly_performance[last_month]
        last_month_name = last_month.strftime('%b %Y')
        
        # Create tabs for Under and Over SKUs
        eval_tab1, eval_tab2 = st.tabs([f"📉 UNDER Forecast ({last_month_name})", f"📈 OVER Forecast ({last_month_name})"])
        
        with eval_tab1:
            under_skus_df = last_month_data['under_skus']
            if not under_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    under_skus_df = pd.merge(under_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # Prepare display columns - TAMBAHKAN Product_Name, Brand, Status
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                               'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                               'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in under_skus_df.columns]
                
                # Format the dataframe
                display_df = under_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary
                total_forecast = under_skus_df['Forecast_Qty'].sum()
                total_po = under_skus_df['PO_Qty'].sum()
                avg_ratio = under_skus_df['PO_Rofo_Ratio'].mean()
                
                st.caption(f"""
                **Total UNDER Forecast SKUs:** {len(under_skus_df)} | 
                **Average PO/Rofo Ratio:** {avg_ratio:.1f}% | 
                **Total Forecast:** {total_forecast:,.0f} | 
                **Total PO:** {total_po:,.0f} | 
                **Selisih:** {total_po - total_forecast:+,.0f} ({((total_po - total_forecast)/total_forecast*100 if total_forecast > 0 else 0):+.1f}%)
                """)
            else:
                st.success(f"✅ No SKUs with UNDER forecast in {last_month_name}")
        
        with eval_tab2:
            over_skus_df = last_month_data['over_skus']
            if not over_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    over_skus_df = pd.merge(over_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # Prepare display columns - TAMBAHKAN Product_Name, Brand, Status
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                               'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                               'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in over_skus_df.columns]
                
                # Format the dataframe
                display_df = over_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary
                total_forecast = over_skus_df['Forecast_Qty'].sum()
                total_po = over_skus_df['PO_Qty'].sum()
                avg_ratio = over_skus_df['PO_Rofo_Ratio'].mean()
                
                st.caption(f"""
                **Total OVER Forecast SKUs:** {len(over_skus_df)} | 
                **Average PO/Rofo Ratio:** {avg_ratio:.1f}% | 
                **Total Forecast:** {total_forecast:,.0f} | 
                **Total PO:** {total_po:,.0f} | 
                **Selisih:** {total_po - total_forecast:+,.0f} ({((total_po - total_forecast)/total_forecast*100 if total_forecast > 0 else 0):+.1f}%)
                """)
            else:
                st.success(f"✅ No SKUs with OVER forecast in {last_month_name}")

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Monthly Performance",
    "📊 Tier Analysis",
    "📦 Inventory Analysis",
    "🔍 SKU Evaluation",
    "📈 Sales Analysis",
    "📋 Data Explorer"
])

# --- TAB 1: MONTHLY PERFORMANCE ---
with tab1:
    st.subheader("📅 Monthly Performance Details")
    
    if monthly_performance:
        # Create monthly performance summary table
        summary_data = []
        for month, data in sorted(monthly_performance.items()):
            summary_data.append({
                'Month': month.strftime('%b %Y'),
                'Accuracy (%)': data['accuracy'],
                'Under': data['status_counts'].get('Under', 0),
                'Accurate': data['status_counts'].get('Accurate', 0),
                'Over': data['status_counts'].get('Over', 0),
                'Total SKUs': data['total_records'],
                'MAPE': data['mape']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary table
        st.dataframe(
            summary_df,
            column_config={
                "Accuracy (%)": st.column_config.ProgressColumn(
                    "Accuracy %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "MAPE": st.column_config.NumberColumn("MAPE %", format="%.1f%%")
            },
            use_container_width=True,
            height=400
        )
        
        # Accuracy trend chart
        st.subheader("📈 Accuracy Trend Over Time")
        
        trend_data = summary_df.copy()
        trend_data['Month'] = pd.to_datetime(trend_data['Month'])
        trend_data = trend_data.sort_values('Month')
        
        trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
            x=alt.X('Month:T', title='Month'),
            y=alt.Y('Accuracy (%):Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 100])),
            tooltip=['Month:T', 'Accuracy (%):Q', 'Under', 'Accurate', 'Over']
        ).properties(height=400)
        
        st.altair_chart(trend_chart, use_container_width=True)
        
        # Brand Performance Analysis
        st.divider()
        st.subheader("🏷️ Forecast Performance by Brand")
        
        brand_performance = calculate_brand_performance(df_forecast, df_po, df_product)
        
        if not brand_performance.empty:
            # Format the display
            display_brand_df = brand_performance.copy()
            
            # Format columns
            display_brand_df['Accuracy'] = display_brand_df['Accuracy'].apply(lambda x: f"{x:.1f}%")
            display_brand_df['PO_vs_Forecast_Ratio'] = display_brand_df['PO_vs_Forecast_Ratio'].apply(lambda x: f"{x:.1f}%")
            display_brand_df['Total_Forecast'] = display_brand_df['Total_Forecast'].apply(lambda x: f"{x:,.0f}")
            display_brand_df['Total_PO'] = display_brand_df['Total_PO'].apply(lambda x: f"{x:,.0f}")
            display_brand_df['Qty_Difference'] = display_brand_df['Qty_Difference'].apply(lambda x: f"{x:+,.0f}")
            
            # Rename columns
            column_names = {
                'Brand': 'Brand',
                'SKU_Count': 'SKU Count',
                'Total_Forecast': 'Total Rofo',
                'Total_PO': 'Total PO',
                'Accuracy': 'Accuracy %',
                'PO_vs_Forecast_Ratio': 'PO/Rofo %',
                'Qty_Difference': 'Qty Diff',
                'Under': 'Under',
                'Accurate': 'Accurate',
                'Over': 'Over'
            }
            
            display_brand_df = display_brand_df.rename(columns=column_names)
            
            # Display table
            st.dataframe(
                display_brand_df,
                use_container_width=True,
                height=400
            )
            
            # Chart for brand accuracy
            chart_brand_df = brand_performance.copy()
            
            # Create bar chart
            bars_brand = alt.Chart(chart_brand_df).mark_bar().encode(
                x=alt.X('Brand:N', title='Brand', sort='-y'),
                y=alt.Y('Accuracy:Q', title='Accuracy (%)'),
                color=alt.Color('Brand:N', scale=alt.Scale(scheme='set3')),
                tooltip=['Brand', 'Accuracy', 'SKU_Count', 'Total_Forecast', 'Total_PO']
            ).properties(height=300)
            
            st.altair_chart(bars_brand, use_container_width=True)
        else:
            st.info("📊 No brand performance data available")

# --- TAB 2: TIER ANALYSIS ---
with tab2:
    st.subheader("🏷️ SKU Tier Analysis")
    
    if monthly_performance and not df_product.empty:
        # Get last month data for tier analysis
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Tier analysis
        if 'SKU_Tier' in last_month_data.columns:
            tier_summary = last_month_data.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'PO_Rofo_Ratio': 'mean',
                'Forecast_Qty': 'sum',
                'PO_Qty': 'sum'
            }).reset_index()
            
            tier_summary.columns = ['Tier', 'SKU Count', 'Avg PO/Rofo %', 'Total Forecast', 'Total PO']
            tier_summary['Avg PO/Rofo %'] = tier_summary['Avg PO/Rofo %'].apply(lambda x: f"{x:.1f}%")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.dataframe(
                    tier_summary,
                    use_container_width=True,
                    height=300
                )
            
            with col_t2:
                # Tier accuracy chart
                tier_acc = last_month_data.groupby('SKU_Tier').apply(
                    lambda x: 100 - abs(x['PO_Rofo_Ratio'] - 100).mean()
                ).reset_index()
                tier_acc.columns = ['Tier', 'Accuracy']
                
                bars = alt.Chart(tier_acc).mark_bar().encode(
                    x=alt.X('Tier:N', title='SKU Tier'),
                    y=alt.Y('Accuracy:Q', title='Accuracy (%)'),
                    color=alt.Color('Tier:N', scale=alt.Scale(scheme='set2'))
                ).properties(height=300)
                
                st.altair_chart(bars, use_container_width=True)
        
        # Inventory tier analysis
        if 'tier_analysis' in inventory_metrics:
            st.subheader("📦 Inventory by Tier")
            
            tier_inv = inventory_metrics['tier_analysis']
            
            col_t3, col_t4 = st.columns(2)
            
            with col_t3:
                st.dataframe(
                    tier_inv,
                    use_container_width=True,
                    height=300
                )
            
            with col_t4:
                # Inventory coverage by tier
                bars_cov = alt.Chart(tier_inv).mark_bar().encode(
                    x=alt.X('Tier:N', title='SKU Tier'),
                    y=alt.Y('Avg_Cover_Months:Q', title='Avg Cover (Months)'),
                    color=alt.Color('Tier:N', scale=alt.Scale(scheme='set1'))
                ).properties(height=300)
                
                st.altair_chart(bars_cov, use_container_width=True)

# --- TAB 3: INVENTORY ANALYSIS ---
with tab3:
    st.subheader("📦 Inventory Analysis dengan 3-Month Average Sales")
    
    if inventory_metrics:
        # Inventory status cards
        inv_df = inventory_metrics['inventory_df']
        
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
        
        st.divider()
        
        # Detailed Inventory Table
        st.subheader("📋 Detailed Inventory Status")
        
        # Filter options
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=['Need Replenishment', 'Ideal/Healthy', 'High Stock'],
                default=['Need Replenishment', 'High Stock']
            )
        
        with col_f2:
            tier_filter = st.multiselect(
                "Filter by Tier",
                options=inv_df['SKU_Tier'].unique().tolist() if 'SKU_Tier' in inv_df.columns else [],
                default=inv_df['SKU_Tier'].unique().tolist() if 'SKU_Tier' in inv_df.columns else []
            )
        
        # Apply filters
        filtered_df = inv_df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df['Inventory_Status'].isin(status_filter)]
        if tier_filter and 'SKU_Tier' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['SKU_Tier'].isin(tier_filter)]
        
        # Prepare display columns
        display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Inventory_Status', 
                       'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
        
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        # Format the dataframe
        display_df = filtered_df[available_cols].copy()
        
        # Add formatted columns
        if 'Cover_Months' in display_df.columns:
            display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
        
        if 'Avg_Monthly_Sales_3M' in display_df.columns:
            display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
        
        # Rename columns for display
        column_names = {
            'SKU_ID': 'SKU ID',
            'Product_Name': 'Product Name',
            'Brand': 'Brand',
            'SKU_Tier': 'Tier',
            'Inventory_Status': 'Status',
            'Stock_Qty': 'Stock Available',
            'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
            'Cover_Months': 'Cover (Months)'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # Sort by status and cover months
        if 'Cover (Months)' in display_df.columns and 'Status' in display_df.columns:
            # Convert back to numeric for sorting
            display_df['Cover_Numeric'] = pd.to_numeric(display_df['Cover (Months)'].replace('N/A', np.nan), errors='coerce')
            display_df = display_df.sort_values(['Status', 'Cover_Numeric'], 
                                               ascending=[True, False if 'High' in status_filter else True])
            display_df = display_df.drop('Cover_Numeric', axis=1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )
        
        # Summary statistics
        st.caption(f"**Showing {len(filtered_df)} of {len(inv_df)} SKUs** | **Average Cover:** {inv_df[inv_df['Cover_Months'] < 999]['Cover_Months'].mean():.1f} months")

# --- TAB 4: SKU EVALUATION ---
with tab4:
    st.subheader("🔍 SKU Performance Evaluation")
    
    if monthly_performance:
        # Get last month for evaluation
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Add inventory data
        if 'inventory_df' in inventory_metrics:
            inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months', 'Inventory_Status']]
            last_month_data = pd.merge(last_month_data, inventory_data, on='SKU_ID', how='left')
        
        # Create comprehensive evaluation table
        eval_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 
                    'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 'Accuracy_Status',
                    'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months', 'Inventory_Status']
        
        available_cols = [col for col in eval_cols if col in last_month_data.columns]
        
        eval_df = last_month_data[available_cols].copy()
        
        # Format columns
        if 'PO_Rofo_Ratio' in eval_df.columns:
            eval_df['PO_Rofo_Ratio'] = eval_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
        
        if 'Cover_Months' in eval_df.columns:
            eval_df['Cover_Months'] = eval_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
        
        if 'Avg_Monthly_Sales_3M' in eval_df.columns:
            eval_df['Avg_Monthly_Sales_3M'] = eval_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
        
        # Rename columns
        column_names = {
            'SKU_ID': 'SKU ID',
            'Product_Name': 'Product Name',
            'Brand': 'Brand',
            'SKU_Tier': 'Tier',
            'Forecast_Qty': 'Forecast',
            'PO_Qty': 'PO',
            'PO_Rofo_Ratio': 'PO/Rofo %',
            'Accuracy_Status': 'Forecast Status',
            'Stock_Qty': 'Stock',
            'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
            'Cover_Months': 'Cover (Months)',
            'Inventory_Status': 'Inventory Status'
        }
        
        eval_df = eval_df.rename(columns=column_names)
        
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            ["Forecast Status", "PO/Rofo %", "Cover (Months)", "Tier"]
        )
        
        if sort_option == "Forecast Status":
            eval_df = eval_df.sort_values(['Forecast Status', 'PO/Rofo %'], 
                                         ascending=[True, True])
        elif sort_option == "PO/Rofo %":
            # Extract numeric value for sorting
            eval_df['PO_Rofo_Numeric'] = pd.to_numeric(eval_df['PO/Rofo %'].str.replace('%', ''), errors='coerce')
            eval_df = eval_df.sort_values('PO_Rofo_Numeric', ascending=False)
            eval_df = eval_df.drop('PO_Rofo_Numeric', axis=1)
        elif sort_option == "Cover (Months)":
            eval_df['Cover_Numeric'] = pd.to_numeric(eval_df['Cover (Months)'].replace('N/A', np.nan), errors='coerce')
            eval_df = eval_df.sort_values('Cover_Numeric', ascending=False)
            eval_df = eval_df.drop('Cover_Numeric', axis=1)
        elif sort_option == "Tier":
            eval_df = eval_df.sort_values('Tier', ascending=True)
        
        st.dataframe(
            eval_df,
            use_container_width=True,
            height=600
        )

# --- TAB 5: SALES ANALYSIS ---
with tab5:
    st.subheader("📈 Sales Analysis vs Forecast & PO")
    
    if sales_vs_forecast:
        last_month = sales_vs_forecast['last_month']
        last_month_name = last_month.strftime('%b %Y')
        
        st.markdown(f"### 📊 Overview - {last_month_name}")
        
        # Overview metrics
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric(
                "Avg Forecast Deviation",
                f"{sales_vs_forecast['avg_forecast_deviation']:.1f}%",
                delta=f"Target: < 20%",
                delta_color="off"
            )
        
        with col_s2:
            st.metric(
                "Avg PO Deviation",
                f"{sales_vs_forecast['avg_po_deviation']:.1f}%",
                delta=f"Target: < 20%",
                delta_color="off"
            )
        
        with col_s3:
            st.metric(
                "SKUs Compared",
                sales_vs_forecast['total_skus_compared'],
                delta=f"High Deviation: {len(sales_vs_forecast['high_deviation_skus'])}",
                delta_color="off"
            )
        
        st.divider()
        
        # High Deviation SKUs
        st.subheader("⚠️ SKUs with High Deviation (>30%)")
        
        high_dev_df = sales_vs_forecast['high_deviation_skus']
        
        if not high_dev_df.empty:
            # Prepare display
            display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier',
                          'Sales_Qty', 'Forecast_Qty', 'PO_Qty',
                          'Sales_vs_Forecast_Ratio', 'Sales_vs_PO_Ratio',
                          'Forecast_Deviation', 'PO_Deviation']
            
            available_cols = [col for col in display_cols if col in high_dev_df.columns]
            
            display_df = high_dev_df[available_cols].copy()
            
            # Format percentages
            percent_cols = ['Sales_vs_Forecast_Ratio', 'Sales_vs_PO_Ratio', 'Forecast_Deviation', 'PO_Deviation']
            for col in percent_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
            
            # Rename columns
            column_names = {
                'SKU_ID': 'SKU ID',
                'Product_Name': 'Product Name',
                'Brand': 'Brand',
                'SKU_Tier': 'Tier',
                'Sales_Qty': 'Sales Qty',
                'Forecast_Qty': 'Forecast Qty',
                'PO_Qty': 'PO Qty',
                'Sales_vs_Forecast_Ratio': 'Sales/Forecast %',
                'Sales_vs_PO_Ratio': 'Sales/PO %',
                'Forecast_Deviation': 'Forecast Dev %',
                'PO_Deviation': 'PO Dev %'
            }
            
            display_df = display_df.rename(columns=column_names)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Analysis
            st.markdown("#### 📋 Analysis")
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                # Most over-forecast
                most_over = high_dev_df.loc[high_dev_df['Sales_vs_Forecast_Ratio'].idxmin()]
                st.info(f"""
                **Most Over-Forecasted:** {most_over.get('Product_Name', most_over['SKU_ID'])}
                - Sales/Forecast: {most_over['Sales_vs_Forecast_Ratio']:.1f}%
                - Forecast: {most_over['Forecast_Qty']:,.0f} | Sales: {most_over['Sales_Qty']:,.0f}
                """)
            
            with col_a2:
                # Most under-forecast
                most_under = high_dev_df.loc[high_dev_df['Sales_vs_Forecast_Ratio'].idxmax()]
                st.warning(f"""
                **Most Under-Forecasted:** {most_under.get('Product_Name', most_under['SKU_ID'])}
                - Sales/Forecast: {most_under['Sales_vs_Forecast_Ratio']:.1f}%
                - Forecast: {most_under['Forecast_Qty']:,.0f} | Sales: {most_under['Sales_Qty']:,.0f}
                """)
        else:
            st.success(f"✅ No SKUs with high deviation (>30%) in {last_month_name}")
        
        # Sales vs Forecast/PO Comparison Chart
        st.divider()
        st.subheader("📈 Comparison Chart")
        
        comp_data = sales_vs_forecast['comparison_data'].copy()
        
        # Sample top 20 SKUs for chart clarity
        if len(comp_data) > 20:
            chart_data = comp_data.head(20)
        else:
            chart_data = comp_data
        
        # Create comparison chart
        chart_data_melted = chart_data.melt(
            id_vars=['SKU_ID', 'Product_Name'],
            value_vars=['Sales_Qty', 'Forecast_Qty', 'PO_Qty'],
            var_name='Metric',
            value_name='Quantity'
        )
        
        bars = alt.Chart(chart_data_melted).mark_bar().encode(
            x=alt.X('SKU_ID:N', title='SKU'),
            y=alt.Y('Quantity:Q', title='Quantity'),
            color=alt.Color('Metric:N', 
                          scale=alt.Scale(domain=['Sales_Qty', 'Forecast_Qty', 'PO_Qty'],
                                        range=['#4CAF50', '#667eea', '#FF9800'])),
            tooltip=['SKU_ID', 'Product_Name', 'Metric', 'Quantity']
        ).properties(height=400)
        
        st.altair_chart(bars, use_container_width=True)
    
    else:
        st.info("📊 Insufficient data for sales vs forecast comparison")

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
    <p>🚀 <strong>Inventory Intelligence Dashboard v5.0</strong> | Professional Inventory Control & Demand Planning</p>
    <p>✅ 3-Month Average Sales for Inventory | ✅ Monthly Performance Tracking | ✅ Detailed SKU Evaluation | ✅ Sales vs Forecast Analysis</p>
    <p>📊 Using 3-month average sales for accurate inventory coverage calculation</p>
</div>
""", unsafe_allow_html=True)
