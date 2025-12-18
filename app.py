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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        box-shadow: 0 2px 5px rgba(102, 126, 234, 0.3);
    }
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
            df['Status'] = 'Active'  # Default value
        
        # Filter only Active SKUs
        df_active = df[df['Status'].str.upper() == 'ACTIVE'].copy()
        
        # Get Product_Name from other sheets if missing
        if 'Product_Name' not in df_active.columns or df_active['Product_Name'].isnull().all():
            # Try to get from Rofo sheet
            try:
                ws_rofo = _client.open_by_url(gsheet_url).worksheet("Rofo")
                df_rofo = pd.DataFrame(ws_rofo.get_all_records())
                if 'Product_Name' in df_rofo.columns:
                    product_names = df_rofo[['SKU_ID', 'Product_Name']].drop_duplicates()
                    df_active = pd.merge(df_active, product_names, on='SKU_ID', how='left')
            except:
                pass
        
        data['product'] = df
        data['product_active'] = df_active
        
    except Exception as e:
        data['product'] = pd.DataFrame()
        data['product_active'] = pd.DataFrame()
    
    # 2. SALES DATA (Customer Sales - Monthly)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Sales")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Standardize SKU_ID column
        if 'SKU_ID' not in df.columns and 'Current_SKU' in df.columns:
            df['SKU_ID'] = df['Current_SKU']
        
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
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['sales'] = df_long
            
    except Exception as e:
        data['sales'] = pd.DataFrame()
    
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
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['forecast'] = df_long
            
    except Exception as e:
        data['forecast'] = pd.DataFrame()
    
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
            df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
            
            # Filter only active SKUs if available
            if 'product_active' in data and not data['product_active'].empty:
                active_skus = data['product_active']['SKU_ID'].tolist()
                df_long = df_long[df_long['SKU_ID'].isin(active_skus)]
            
            data['po'] = df_long
            
    except Exception as e:
        data['po'] = pd.DataFrame()
    
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
    
    return data

def parse_month_label(label):
    """Parse month label to datetime"""
    try:
        label_str = str(label).strip()
        
        # Try different date parsers
        for fmt in ['%b-%y', '%b %Y', '%Y-%m', '%b-%Y', '%B %Y', '%b %y']:
            try:
                return datetime.strptime(label_str, fmt)
            except:
                continue
        
        # If all fail, return current date
        return datetime.now()
    except:
        return datetime.now()

# --- ====================================================== ---
# ---               CORE ANALYTICS FUNCTIONS                ---
# --- ====================================================== ---

def calculate_forecast_accuracy_by_status(df_forecast, df_po, df_product):
    """Calculate forecast accuracy based on PO vs Rofo with status categorization"""
    
    metrics = {}
    
    if df_forecast.empty or df_po.empty:
        return metrics
    
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
            return metrics
        
        # Add Product_Name if available
        if not df_product.empty and 'SKU_ID' in df_product.columns and 'Product_Name' in df_product.columns:
            product_names = df_product[['SKU_ID', 'Product_Name']].drop_duplicates()
            df_merged = pd.merge(df_merged, product_names, on='SKU_ID', how='left')
        
        # Calculate PO/Rofo ratio (avoid division by zero)
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
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
        mape = df_merged['Absolute_Percentage_Error'].mean()
        
        # Overall accuracy
        overall_accuracy = 100 - mape
        
        # Count by status
        status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
        
        # Calculate percentages
        total_records = len(df_merged)
        status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
        
        # Brand-level analysis
        brand_metrics = {}
        if 'Brand' in df_merged.columns:
            for brand in df_merged['Brand'].unique():
                brand_data = df_merged[df_merged['Brand'] == brand]
                brand_mape = brand_data['Absolute_Percentage_Error'].mean()
                brand_accuracy = 100 - brand_mape
                brand_counts = brand_data['Accuracy_Status'].value_counts().to_dict()
                brand_metrics[brand] = {
                    'accuracy': brand_accuracy,
                    'mape': brand_mape,
                    'counts': brand_counts,
                    'total_records': len(brand_data)
                }
        
        # SKU-level accuracy
        sku_accuracy = df_merged.groupby('SKU_ID').apply(
            lambda x: 100 - x['Absolute_Percentage_Error'].mean()
        ).reset_index()
        sku_accuracy.columns = ['SKU_ID', 'SKU_Accuracy']
        
        # Add Product_Name to SKU accuracy
        if 'Product_Name' in df_merged.columns:
            sku_names = df_merged[['SKU_ID', 'Product_Name']].drop_duplicates()
            sku_accuracy = pd.merge(sku_accuracy, sku_names, on='SKU_ID', how='left')
        
        # Monthly accuracy trend
        monthly_accuracy = df_merged.groupby('Month').apply(
            lambda x: 100 - x['Absolute_Percentage_Error'].mean()
        ).reset_index()
        monthly_accuracy.columns = ['Month', 'Monthly_Accuracy']
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'mape': mape,
            'status_counts': status_counts,
            'status_percentages': status_percentages,
            'brand_metrics': brand_metrics,
            'sku_accuracy': sku_accuracy,
            'monthly_accuracy': monthly_accuracy,
            'detailed_data': df_merged
        }
        
    except Exception as e:
        st.error(f"Accuracy calculation error: {str(e)}")
    
    return metrics

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
            product_cols = ['SKU_ID', 'SKU_Tier', 'Brand', 'Product_Name']
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
                999  # Infinite cover if no sales
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
            
            # Calculate percentages
            total_skus = len(df_inventory)
            status_percentages = {k: (v/total_skus*100) for k, v in status_counts.items()}
            
            metrics['cover_months_avg'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
            metrics['status_counts'] = status_counts
            metrics['status_percentages'] = status_percentages
        
        metrics['total_stock_qty'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        metrics['inventory_df'] = df_inventory
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
    
    return metrics

def generate_stock_recommendations_filtered(df_inventory, df_product):
    """Generate recommendations only for Active SKUs"""
    
    recommendations = []
    
    if df_inventory.empty:
        return pd.DataFrame()
    
    try:
        # Ensure we're only working with Active SKUs
        if 'Status' in df_product.columns:
            active_skus = df_product[df_product['Status'].str.upper() == 'ACTIVE']['SKU_ID'].tolist()
            df_inventory = df_inventory[df_inventory['SKU_ID'].isin(active_skus)]
        
        for _, row in df_inventory.iterrows():
            sku_id = row['SKU_ID']
            current_stock = row['Stock_Qty']
            product_name = row.get('Product_Name', 'N/A')
            
            # Get MOQ
            moq = 0
            if not df_product.empty:
                product_info = df_product[df_product['SKU_ID'] == sku_id]
                if not product_info.empty and 'MOQ' in product_info.columns:
                    moq_value = product_info['MOQ'].iloc[0]
                    if pd.notna(moq_value):
                        moq = int(moq_value)
            
            # Determine recommendation based on inventory status
            if 'Inventory_Status' in row:
                status = row['Inventory_Status']
                
                if status == 'Need Replenishment':
                    rec_status = "🟡 NEED REPLENISHMENT"
                    rec_qty = max(moq, 50)  # Default minimum order
                    priority = 1
                elif status == 'High Stock':
                    rec_status = "🟠 HIGH STOCK"
                    rec_qty = 0
                    priority = 3
                else:  # Ideal/Healthy
                    rec_status = "🟢 HEALTHY"
                    rec_qty = 0
                    priority = 4
            else:
                # Fallback if no inventory status
                if current_stock == 0:
                    rec_status = "🔴 OUT OF STOCK"
                    rec_qty = max(moq, 100)
                    priority = 1
                elif current_stock < 10:
                    rec_status = "🟡 LOW STOCK"
                    rec_qty = max(moq, 50 - current_stock)
                    priority = 2
                else:
                    rec_status = "🟢 ADEQUATE"
                    rec_qty = 0
                    priority = 4
            
            recommendations.append({
                'SKU_ID': sku_id,
                'Product_Name': product_name,
                'Current_Stock': int(current_stock),
                'MOQ': moq,
                'Recommended_Qty': int(rec_qty),
                'Status': rec_status,
                'Priority': priority
            })
        
        return pd.DataFrame(recommendations).sort_values('Priority')
        
    except Exception as e:
        st.error(f"Recommendation generation error: {str(e)}")
        return pd.DataFrame()

# --- ====================================================== ---
# ---               VISUALIZATION FUNCTIONS                 ---
# --- ====================================================== ---

def create_accuracy_status_chart(status_counts, status_percentages):
    """Create chart showing accuracy status distribution"""
    
    if not status_counts:
        return None
    
    # Prepare data
    data = []
    for status, count in status_counts.items():
        percentage = status_percentages.get(status, 0)
        data.append({
            'Status': status,
            'Count': count,
            'Percentage': percentage
        })
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X('Status:N', title='Accuracy Status', sort=['Under', 'Accurate', 'Over']),
        y=alt.Y('Count:Q', title='Number of Records'),
        color=alt.Color('Status:N', 
                      scale=alt.Scale(
                          domain=['Under', 'Accurate', 'Over'],
                          range=['#FF9800', '#4CAF50', '#F44336']
                      ),
                      legend=None),
        tooltip=['Status', 'Count', alt.Tooltip('Percentage', format='.1f')]
    ).properties(
        height=300,
        title="Forecast Accuracy Status Distribution"
    )
    
    # Add text labels
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=12,
        fontWeight='bold'
    ).encode(
        text=alt.Text('Percentage:Q', format='.1f')
    )
    
    return bars + text

def create_monthly_accuracy_trend(monthly_accuracy):
    """Create line chart for monthly accuracy trend"""
    
    if monthly_accuracy.empty:
        return None
    
    # Create line chart
    line = alt.Chart(monthly_accuracy).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Month:T', title='Month', axis=alt.Axis(format="%b %Y")),
        y=alt.Y('Monthly_Accuracy:Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['Month:T', alt.Tooltip('Monthly_Accuracy', format='.1f')]
    ).properties(
        height=350,
        title="Monthly Forecast Accuracy Trend"
    )
    
    # Add target line at 80%
    target_data = pd.DataFrame({'y': [80]})
    target_line = alt.Chart(target_data).mark_rule(
        strokeDash=[5, 5], color='#FF9800', strokeWidth=2
    ).encode(y='y:Q')
    
    # Add area under line
    area = alt.Chart(monthly_accuracy).mark_area(
        opacity=0.3,
        line={'color': '#667eea'}
    ).encode(
        x='Month:T',
        y='Monthly_Accuracy:Q'
    )
    
    return (area + line + target_line).interactive()

def create_inventory_status_chart(status_counts, status_percentages):
    """Create chart showing inventory status distribution"""
    
    if not status_counts:
        return None
    
    # Prepare data
    data = []
    for status, count in status_counts.items():
        percentage = status_percentages.get(status, 0)
        data.append({
            'Status': status,
            'Count': count,
            'Percentage': percentage
        })
    
    df = pd.DataFrame(data)
    
    # Create donut chart
    base = alt.Chart(df).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color("Status:N", 
                      scale=alt.Scale(
                          domain=['Need Replenishment', 'Ideal/Healthy', 'High Stock'],
                          range=['#FF9800', '#4CAF50', '#F44336']
                      ),
                      legend=alt.Legend(title="Inventory Status", columns=1)),
        tooltip=['Status:N', 'Count:Q', alt.Tooltip('Percentage:Q', format='.1f')]
    )
    
    donut = base.mark_arc(innerRadius=60, outerRadius=120)
    text = base.mark_text(radius=140, size=12).encode(text="Count:Q")
    
    return (donut + text).properties(
        height=350,
        title="Inventory Status Distribution"
    )

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
forecast_metrics = calculate_forecast_accuracy_by_status(df_forecast, df_po, df_product)
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
    
    # Active SKU Status
    if not df_product.empty and 'Status' in df_product.columns:
        st.markdown("#### 📋 SKU Status")
        status_counts = df_product['Status'].value_counts()
        for status, count in status_counts.items():
            st.write(f"**{status}:** {count} SKUs")

# --- MAIN DASHBOARD CONTENT ---

# Header Metrics - Split by Status
st.subheader("🎯 Forecast Accuracy Metrics (PO vs Rofo)")

if forecast_metrics:
    # Get status metrics
    status_counts = forecast_metrics.get('status_counts', {})
    status_percentages = forecast_metrics.get('status_percentages', {})
    
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
    
    # Overall MAPE
    col_mape1, col_mape2 = st.columns(2)
    with col_mape1:
        mape = forecast_metrics.get('mape', 0)
        accuracy = forecast_metrics.get('overall_accuracy', 0)
        st.metric("Mean Absolute % Error (MAPE)", f"{mape:.1f}%")
    
    with col_mape2:
        st.metric("Overall Forecast Accuracy", f"{accuracy:.1f}%")

else:
    st.warning("⚠️ Tidak cukup data untuk menghitung forecast accuracy")

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Intelligence",
    "📦 Inventory Health",
    "🤖 Smart Recommendations",
    "📊 Sales Analytics",
    "📋 Data Explorer"
])

# --- TAB 1: FORECAST INTELLIGENCE ---
with tab1:
    st.subheader("📊 Forecast Performance Analysis (PO vs Rofo)")
    
    if not forecast_metrics:
        st.warning("⚠️ Forecast atau PO data tidak tersedia untuk analisis")
    else:
        # Charts Row
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            # Monthly Accuracy Trend
            monthly_acc = forecast_metrics.get('monthly_accuracy')
            if monthly_acc is not None and not monthly_acc.empty:
                trend_chart = create_monthly_accuracy_trend(monthly_acc)
                if trend_chart:
                    st.altair_chart(trend_chart, use_container_width=True)
        
        with col_f2:
            # Accuracy Status Chart
            status_counts = forecast_metrics.get('status_counts', {})
            status_percentages = forecast_metrics.get('status_percentages', {})
            if status_counts:
                status_chart = create_accuracy_status_chart(status_counts, status_percentages)
                if status_chart:
                    st.altair_chart(status_chart, use_container_width=True)
        
        # SKU Accuracy Ranking (Only Over & Under)
        st.subheader("📋 SKU Accuracy Evaluation (Over & Under Only)")
        
        sku_accuracy = forecast_metrics.get('sku_accuracy')
        if sku_accuracy is not None and not sku_accuracy.empty:
            # Filter only Under and Over accuracy SKUs
            detailed_data = forecast_metrics.get('detailed_data', pd.DataFrame())
            if not detailed_data.empty:
                # Get SKUs with Under or Over status
                problem_skus = detailed_data[
                    detailed_data['Accuracy_Status'].isin(['Under', 'Over'])
                ]['SKU_ID'].unique()
                
                # Filter SKU accuracy data
                problem_sku_accuracy = sku_accuracy[sku_accuracy['SKU_ID'].isin(problem_skus)].copy()
                
                # Add status information
                sku_status = detailed_data[['SKU_ID', 'Accuracy_Status']].drop_duplicates()
                problem_sku_accuracy = pd.merge(problem_sku_accuracy, sku_status, on='SKU_ID', how='left')
                
                # Sort by accuracy (worst first)
                problem_sku_accuracy = problem_sku_accuracy.sort_values('SKU_Accuracy')
                
                if not problem_sku_accuracy.empty:
                    # Display table
                    st.dataframe(
                        problem_sku_accuracy,
                        column_config={
                            "SKU_ID": "SKU ID",
                            "Product_Name": "Product Name",
                            "SKU_Accuracy": st.column_config.ProgressColumn(
                                "Accuracy",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "Accuracy_Status": st.column_config.SelectboxColumn(
                                "Status",
                                options=["Under", "Accurate", "Over"]
                            )
                        },
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("✅ Semua SKU memiliki akurasi yang akurat (80-120%)")
        
        # Brand-Level Analysis
        st.subheader("🏷️ Accuracy by Brand")
        
        brand_metrics = forecast_metrics.get('brand_metrics', {})
        if brand_metrics:
            brand_data = []
            for brand, metrics in brand_metrics.items():
                brand_data.append({
                    'Brand': brand,
                    'Accuracy': metrics['accuracy'],
                    'MAPE': metrics['mape'],
                    'Total Records': metrics['total_records']
                })
            
            df_brands = pd.DataFrame(brand_data).sort_values('Accuracy', ascending=False)
            
            # Display as metrics cards
            cols = st.columns(len(df_brands))
            for idx, (col, row) in enumerate(zip(cols, df_brands.itertuples())):
                with col:
                    accuracy_color = "#4CAF50" if row.Accuracy >= 80 else "#FF9800" if row.Accuracy >= 70 else "#F44336"
                    st.markdown(f"""
                    <div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <div style="font-size: 0.9rem; color: #666;">{row.Brand}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {accuracy_color};">{row.Accuracy:.1f}%</div>
                        <div style="font-size: 0.8rem;">MAPE: {row.MAPE:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

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
        
        # Charts
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            # Inventory Status Distribution Chart
            if status_counts:
                inventory_chart = create_inventory_status_chart(status_counts, status_percentages)
                if inventory_chart:
                    st.altair_chart(inventory_chart, use_container_width=True)
        
        with col_chart2:
            # Cover Months Summary
            if 'cover_months_avg' in inventory_metrics:
                avg_cover = inventory_metrics['cover_months_avg']
                st.metric("Average Cover Months", f"{avg_cover:.1f}")
            
            # Total Stock Summary
            st.metric("Total Stock Qty", f"{inventory_metrics.get('total_stock_qty', 0):,.0f}")
            st.metric("Total SKUs", inventory_metrics.get('total_skus', 0))
        
        # Detailed Inventory Table with Product_Name
        st.subheader("📋 Detailed Stock Position")
        
        inventory_df = inventory_metrics.get('inventory_df', pd.DataFrame())
        if not inventory_df.empty:
            # Ensure Product_Name is included
            display_cols = ['SKU_ID']
            
            # Add Product_Name if available
            if 'Product_Name' in inventory_df.columns:
                display_cols.append('Product_Name')
            
            # Add other columns
            display_cols.extend(['Stock_Qty'])
            
            if 'Cover_Months' in inventory_df.columns:
                display_cols.append('Cover_Months')
            
            if 'Inventory_Status' in inventory_df.columns:
                display_cols.append('Inventory_Status')
            
            if 'SKU_Tier' in inventory_df.columns:
                display_cols.append('SKU_Tier')
            
            if 'Brand' in inventory_df.columns:
                display_cols.append('Brand')
            
            # Format the dataframe
            df_display = inventory_df[display_cols].copy()
            
            # Sort by status priority
            status_order = {'Need Replenishment': 1, 'Ideal/Healthy': 2, 'High Stock': 3}
            if 'Inventory_Status' in df_display.columns:
                df_display['Status_Order'] = df_display['Inventory_Status'].map(status_order)
                df_display = df_display.sort_values(['Status_Order', 'Stock_Qty'], ascending=[True, False])
                df_display = df_display.drop('Status_Order', axis=1)
            
            st.dataframe(
                df_display,
                column_config={
                    "SKU_ID": "SKU ID",
                    "Product_Name": "Product Name",
                    "Stock_Qty": st.column_config.NumberColumn("Stock Qty", format="%d"),
                    "Cover_Months": st.column_config.NumberColumn("Cover (Months)", format="%.2f"),
                    "Inventory_Status": "Status",
                    "SKU_Tier": "Tier",
                    "Brand": "Brand"
                },
                use_container_width=True,
                height=400
            )

# --- TAB 3: SMART RECOMMENDATIONS ---
with tab3:
    st.subheader("🤖 Smart Stock Recommendations (Active SKUs Only)")
    
    if df_product_active.empty or df_stock.empty:
        st.warning("⚠️ Product atau stock data tidak tersedia")
    else:
        # Generate recommendations only for Active SKUs
        inventory_df = inventory_metrics.get('inventory_df', pd.DataFrame())
        recommendations = generate_stock_recommendations_filtered(inventory_df, df_product_active)
        
        if recommendations.empty:
            st.info("✅ Tidak ada rekomendasi untuk SKU aktif")
        else:
            # Filter urgent recommendations (Priority 1 & 2)
            urgent_recs = recommendations[recommendations['Priority'].isin([1, 2])]
            
            if not urgent_recs.empty:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%); 
                            color: white; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <h3 style="margin: 0; color: white;">🚨 URGENT ACTIONS REQUIRED</h3>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                        <strong>{len(urgent_recs)} Active SKUs</strong> need immediate attention
                    </p>
                    <p style="margin: 0; font-size: 1.1rem;">
                        Total recommended purchase: <strong>{urgent_recs['Recommended_Qty'].sum():,.0f} units</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display urgent recommendations with Product_Name
                st.dataframe(
                    urgent_recs[['SKU_ID', 'Product_Name', 'Current_Stock', 'MOQ', 'Recommended_Qty', 'Status']],
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Product_Name": "Product Name",
                        "Current_Stock": st.column_config.NumberColumn("Current Stock", format="%d"),
                        "MOQ": st.column_config.NumberColumn("MOQ", format="%d"),
                        "Recommended_Qty": st.column_config.NumberColumn("Rec. Order Qty", format="%d"),
                        "Status": "Action Required"
                    },
                    use_container_width=True,
                    height=300
                )
            
            # All recommendations for Active SKUs
            st.subheader("📊 All Active SKU Recommendations")
            
            if not recommendations.empty:
                st.dataframe(
                    recommendations[['SKU_ID', 'Product_Name', 'Current_Stock', 'MOQ', 'Recommended_Qty', 'Status']],
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Product_Name": "Product Name",
                        "Current_Stock": st.column_config.NumberColumn("Current Stock", format="%d"),
                        "MOQ": st.column_config.NumberColumn("MOQ", format="%d"),
                        "Recommended_Qty": st.column_config.NumberColumn("Rec. Order Qty", format="%d"),
                        "Status": "Status"
                    },
                    use_container_width=True,
                    height=400
                )

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
            # Sales trend over time
            st.subheader("📊 Monthly Sales Trend (Active SKUs)")
            
            # Aggregate sales by month
            monthly_sales = df_sales_active.groupby('Month')['Sales_Qty'].sum().reset_index()
            
            if not monthly_sales.empty:
                # Line chart
                trend_chart = alt.Chart(monthly_sales).mark_line(point=True, size=3).encode(
                    x=alt.X('Month:T', title='Month', axis=alt.Axis(format="%b %Y")),
                    y=alt.Y('Sales_Qty:Q', title='Total Sales (Units)'),
                    tooltip=['Month:T', alt.Tooltip('Sales_Qty', format=',.0f')]
                ).properties(height=400)
                
                st.altair_chart(trend_chart, use_container_width=True)
            
            # Top performing Active SKUs
            st.subheader("🏆 Top Performing Active SKUs")
            
            col_top1, col_top2 = st.columns(2)
            
            with col_top1:
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
                    # Create chart
                    if 'Product_Name' in sku_sales_total.columns:
                        sku_sales_total['Display_Name'] = sku_sales_total['SKU_ID'] + ' - ' + sku_sales_total['Product_Name'].str.slice(0, 20)
                    else:
                        sku_sales_total['Display_Name'] = sku_sales_total['SKU_ID']
                    
                    bars = alt.Chart(sku_sales_total).mark_bar().encode(
                        y=alt.Y('Display_Name:N', title='SKU', sort='-x'),
                        x=alt.X('Sales_Qty:Q', title='Total Sales'),
                        color=alt.value('#667eea'),
                        tooltip=['SKU_ID', 'Product_Name', alt.Tooltip('Sales_Qty', format=',.0f')]
                    ).properties(height=350, title="Top 10 Active SKUs by Total Sales")
                    
                    st.altair_chart(bars, use_container_width=True)

# --- TAB 5: DATA EXPLORER ---
with tab5:
    st.subheader("📋 Raw Data Explorer")
    
    # Dataset selection - Include Product_Name where available
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
            # Show shape
            st.write(f"**Shape:** {df_selected.shape[0]} rows × {df_selected.shape[1]} columns")
            
            # Show Product_Name if available
            if 'Product_Name' in df_selected.columns:
                st.write(f"**Contains Product Names:** Yes")
            
            # Data preview
            st.dataframe(
                df_selected,
                use_container_width=True,
                height=400
            )
            
            # Download option
            csv = df_selected.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_data} (CSV)",
                data=csv,
                file_name=f"{selected_data.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("⚠️ Tidak ada data yang tersedia")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Inventory Intelligence Dashboard v3.0 | Professional Inventory Control System</p>
    <p>✅ Forecast Accuracy (PO vs Rofo) | ✅ Inventory Status by Cover Months | ✅ Active SKU Filtering</p>
</div>
""", unsafe_allow_html=True)
