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
    
    .alert-box {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-weight: 600;
    }
    .alert-red {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
        color: #C62828;
    }
    .alert-yellow {
        background: linear-gradient(135deg, #FFFDE7 0%, #FFF9C4 100%);
        border-left: 5px solid #FFC107;
        color: #FF8F00;
    }
    .alert-green {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
        color: #2E7D32;
    }
    
    .data-table {
        font-size: 0.85rem;
    }
    .highlight-cell {
        background-color: rgba(102, 126, 234, 0.1) !important;
        font-weight: 600;
    }
    
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
    
    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        
        # Clean column names and select relevant columns
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Validate required columns
        required_cols = ['SKU_ID', 'SKU_Tier', 'Brand', 'MOQ', 'Status']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Product Master missing columns: {missing_cols}")
        
        # Ensure numeric columns
        if 'MOQ' in df.columns:
            df['MOQ'] = pd.to_numeric(df['MOQ'], errors='coerce').fillna(0).astype(int)
        
        data['product'] = df
    except Exception as e:
        data['product'] = pd.DataFrame()
        st.warning(f"Product Master error: {str(e)}")
    
    # 2. SALES DATA (Customer Sales - Monthly)
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Sales")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Transform from wide to long format
        # Identify month columns (assume they contain month/year or numbers)
        month_columns = [col for col in df.columns if any(x in col.lower() for x in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
        
        if month_columns:
            # Keep essential columns
            id_columns = ['SKU_ID', 'SKU_Name', 'Brand', 'SKU_Tier']
            available_id_cols = [col for col in id_columns if col in df.columns]
            
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
            
            data['sales'] = df_long
            data['sales_wide'] = df  # Keep wide format for reference
        else:
            data['sales'] = pd.DataFrame()
            
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
            available_id_cols = [col for col in id_columns if col in df.columns]
            
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
            
            data['forecast'] = df_long
            data['forecast_wide'] = df
        else:
            data['forecast'] = pd.DataFrame()
            
    except Exception as e:
        data['forecast'] = pd.DataFrame()
        st.warning(f"Forecast data error: {str(e)}")
    
    # 4. STOCK ON HAND
    try:
        ws = _client.open_by_url(gsheet_url).worksheet("Stock_Onhand")
        df = pd.DataFrame(ws.get_all_records())
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Use Quantity_Available as primary, fallback to Stock_Qty
        if 'Quantity_Available' in df.columns:
            stock_col = 'Quantity_Available'
        elif 'Stock_Qty' in df.columns:
            stock_col = 'Stock_Qty'
        elif 'STOCK_SAP' in df.columns:
            stock_col = 'STOCK_SAP'
        else:
            stock_col = df.columns[-1]  # Use last column as fallback
        
        # Create clean stock dataframe
        stock_data = pd.DataFrame({
            'SKU_ID': df['SKU_ID'] if 'SKU_ID' in df.columns else pd.Series([f"SKU_{i}" for i in range(len(df))]),
            'Stock_Qty': pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        })
        
        # Remove duplicates by SKU_ID (keep max stock)
        stock_data = stock_data.groupby('SKU_ID')['Stock_Qty'].max().reset_index()
        
        data['stock'] = stock_data
        
    except Exception as e:
        data['stock'] = pd.DataFrame()
        st.warning(f"Stock data error: {str(e)}")
    
    # 5. PO DATA (Optional for absorption analysis)
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
            
            data['po'] = df_long
        else:
            data['po'] = pd.DataFrame()
            
    except Exception as e:
        data['po'] = pd.DataFrame()
    
    return data

def parse_month_label(label):
    """Parse month label to datetime"""
    try:
        # Handle various formats: "Jan-25", "Jan 2025", "2025-01"
        label_str = str(label).strip()
        
        # Try different date parsers
        for fmt in ['%b-%y', '%b %Y', '%Y-%m', '%b-%Y', '%B %Y']:
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

def calculate_forecast_accuracy(df_forecast, df_sales):
    """Calculate forecast accuracy metrics"""
    
    metrics = {}
    
    if df_forecast.empty or df_sales.empty:
        metrics['overall_accuracy'] = 0
        metrics['mape'] = 100
        metrics['bias'] = 0
        metrics['sku_count'] = 0
        return metrics
    
    try:
        # Merge forecast and sales data
        df_merged = pd.merge(
            df_forecast,
            df_sales,
            on=['SKU_ID', 'Month'],
            how='inner',
            suffixes=('_forecast', '_sales')
        )
        
        if df_merged.empty:
            metrics['overall_accuracy'] = 0
            metrics['mape'] = 100
            metrics['bias'] = 0
            metrics['sku_count'] = 0
            return metrics
        
        # Calculate accuracy metrics
        df_merged['Absolute_Error'] = abs(df_merged['Forecast_Qty'] - df_merged['Sales_Qty'])
        df_merged['Percentage_Error'] = np.where(
            df_merged['Sales_Qty'] > 0,
            (df_merged['Absolute_Error'] / df_merged['Sales_Qty']) * 100,
            0
        )
        
        # Overall accuracy (100 - MAPE)
        mape = df_merged['Percentage_Error'].mean()
        overall_accuracy = 100 - mape
        
        # Forecast bias (positive = over-forecast, negative = under-forecast)
        bias = (df_merged['Forecast_Qty'] - df_merged['Sales_Qty']).mean()
        
        # SKU-level accuracy
        sku_accuracy = df_merged.groupby('SKU_ID').apply(
            lambda x: 100 - (abs(x['Forecast_Qty'] - x['Sales_Qty']).sum() / x['Sales_Qty'].sum() * 100)
            if x['Sales_Qty'].sum() > 0 else 0
        ).reset_index()
        sku_accuracy.columns = ['SKU_ID', 'SKU_Accuracy']
        
        metrics['overall_accuracy'] = overall_accuracy
        metrics['mape'] = mape
        metrics['bias'] = bias
        metrics['sku_count'] = len(sku_accuracy)
        metrics['high_accuracy_skus'] = len(sku_accuracy[sku_accuracy['SKU_Accuracy'] >= 85])
        metrics['sku_accuracy_df'] = sku_accuracy
        
        # Monthly accuracy
        monthly_accuracy = df_merged.groupby('Month').apply(
            lambda x: 100 - (abs(x['Forecast_Qty'] - x['Sales_Qty']).sum() / x['Sales_Qty'].sum() * 100)
            if x['Sales_Qty'].sum() > 0 else 0
        ).reset_index()
        monthly_accuracy.columns = ['Month', 'Monthly_Accuracy']
        metrics['monthly_accuracy'] = monthly_accuracy
        
    except Exception as e:
        st.error(f"Accuracy calculation error: {str(e)}")
        metrics['overall_accuracy'] = 0
    
    return metrics

def calculate_inventory_metrics(df_stock, df_product, df_sales=None):
    """Calculate inventory performance metrics"""
    
    metrics = {}
    
    if df_stock.empty:
        metrics['total_stock_value'] = 0
        metrics['total_skus'] = 0
        metrics['out_of_stock'] = 0
        metrics['low_stock'] = 0
        return metrics
    
    try:
        # Merge with product master for tier info
        df_inventory = df_stock.copy()
        
        if not df_product.empty and 'SKU_ID' in df_product.columns:
            df_inventory = pd.merge(
                df_inventory,
                df_product[['SKU_ID', 'SKU_Tier', 'Brand', 'MOQ', 'Status']],
                on='SKU_ID',
                how='left'
            )
        
        # Basic inventory metrics
        metrics['total_stock_qty'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        
        # Stock status classification
        metrics['out_of_stock'] = len(df_inventory[df_inventory['Stock_Qty'] == 0])
        metrics['low_stock'] = len(df_inventory[df_inventory['Stock_Qty'] < 10])  # Threshold can be adjusted
        
        # If we have sales data, calculate cover days
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
            
            metrics['avg_cover_months'] = df_inventory['Cover_Months'][df_inventory['Cover_Months'] < 999].mean()
            metrics['excess_stock'] = len(df_inventory[df_inventory['Cover_Months'] > 3])
            metrics['critical_stock'] = len(df_inventory[df_inventory['Cover_Months'] < 0.5])
            
            metrics['inventory_df'] = df_inventory
        
        # Tier-wise analysis
        if 'SKU_Tier' in df_inventory.columns:
            tier_summary = df_inventory.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'Stock_Qty': 'sum'
            }).reset_index()
            tier_summary.columns = ['Tier', 'SKU_Count', 'Total_Stock']
            metrics['tier_summary'] = tier_summary
        
        metrics['inventory_df'] = df_inventory
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
    
    return metrics

def generate_stock_recommendations(df_inventory, df_product, df_sales=None):
    """Generate intelligent stock recommendations"""
    
    recommendations = []
    
    if df_inventory.empty:
        return pd.DataFrame()
    
    try:
        # Ensure we have required columns
        if 'Stock_Qty' not in df_inventory.columns:
            return pd.DataFrame()
        
        for _, row in df_inventory.iterrows():
            sku_id = row['SKU_ID']
            current_stock = row['Stock_Qty']
            
            # Get product info
            moq = 0
            if not df_product.empty:
                product_info = df_product[df_product['SKU_ID'] == sku_id]
                if not product_info.empty and 'MOQ' in product_info.columns:
                    moq_value = product_info['MOQ'].iloc[0]
                    if pd.notna(moq_value):
                        moq = int(moq_value)
            
            # Calculate safety stock if we have sales data
            safety_stock = 0
            if df_sales is not None and not df_sales.empty:
                sku_sales = df_sales[df_sales['SKU_ID'] == sku_id]
                if not sku_sales.empty and len(sku_sales) >= 3:
                    avg_sales = sku_sales['Sales_Qty'].mean()
                    std_sales = sku_sales['Sales_Qty'].std()
                    # Simple safety stock: 1.65 * std * sqrt(lead_time)
                    safety_stock = max(0, 1.65 * std_sales * np.sqrt(0.5))
            
            # Determine recommendation
            if current_stock == 0:
                status = "🔴 URGENT: Out of Stock"
                rec_qty = max(moq, safety_stock) if safety_stock > 0 else moq
                priority = 1
            elif current_stock < safety_stock * 0.5:
                status = "🟡 WARNING: Below Safety Stock"
                rec_qty = max(moq, safety_stock - current_stock)
                priority = 2
            elif current_stock > safety_stock * 3 and safety_stock > 0:
                status = "🟠 EXCESS: Overstocked"
                rec_qty = 0
                priority = 3
            else:
                status = "🟢 HEALTHY: Optimal Stock"
                rec_qty = 0
                priority = 4
            
            recommendations.append({
                'SKU_ID': sku_id,
                'Current_Stock': int(current_stock),
                'Safety_Stock_Level': round(safety_stock),
                'MOQ': moq,
                'Recommended_Qty': int(rec_qty),
                'Status': status,
                'Priority': priority
            })
        
        return pd.DataFrame(recommendations).sort_values('Priority')
        
    except Exception as e:
        st.error(f"Recommendation generation error: {str(e)}")
        return pd.DataFrame()

# --- ====================================================== ---
# ---               VISUALIZATION FUNCTIONS                 ---
# --- ====================================================== ---

def create_forecast_vs_sales_chart(df_forecast, df_sales, selected_sku=None):
    """Create forecast vs sales comparison chart"""
    
    if df_forecast.empty or df_sales.empty:
        return None
    
    try:
        # Filter for selected SKU or aggregate all
        if selected_sku and selected_sku != 'All SKUs':
            df_f = df_forecast[df_forecast['SKU_ID'] == selected_sku]
            df_s = df_sales[df_sales['SKU_ID'] == selected_sku]
            title_suffix = f" - {selected_sku}"
        else:
            # Aggregate all SKUs by month
            df_f = df_forecast.groupby('Month')['Forecast_Qty'].sum().reset_index()
            df_s = df_sales.groupby('Month')['Sales_Qty'].sum().reset_index()
            title_suffix = " - All SKUs"
        
        # Merge data
        df_merged = pd.merge(df_f, df_s, on='Month', how='outer').sort_values('Month')
        
        # Create line chart
        chart_data = df_merged.melt(
            id_vars=['Month'],
            value_vars=['Forecast_Qty', 'Sales_Qty'],
            var_name='Type',
            value_name='Quantity'
        )
        
        chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X('Month:T', title='Month', axis=alt.Axis(format="%b %Y")),
            y=alt.Y('Quantity:Q', title='Quantity', scale=alt.Scale(zero=False)),
            color=alt.Color('Type:N', 
                          scale=alt.Scale(domain=['Forecast_Qty', 'Sales_Qty'],
                                        range=['#667eea', '#ff6b6b']),
                          legend=alt.Legend(title="Data Type")),
            strokeDash=alt.StrokeDash('Type:N',
                scale=alt.Scale(domain=['Forecast_Qty', 'Sales_Qty'],
                              range=[[5, 5], [0, 0]])
            ),
            tooltip=['Month:T', 'Type:N', alt.Tooltip('Quantity:Q', format=',.0f')]
        ).properties(
            height=400,
            title=f"Forecast vs Actual Sales{title_suffix}"
        )
        
        return chart
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return None

def create_inventory_health_chart(df_inventory):
    """Create inventory health visualization"""
    
    if df_inventory.empty:
        return None
    
    try:
        # Create status categories
        if 'Cover_Months' in df_inventory.columns:
            conditions = [
                (df_inventory['Stock_Qty'] == 0),
                (df_inventory['Cover_Months'] < 0.5),
                (df_inventory['Cover_Months'] > 3),
                (df_inventory['Cover_Months'] >= 0.5) & (df_inventory['Cover_Months'] <= 3)
            ]
            choices = ['Out of Stock', 'Critical (<0.5m)', 'Excess (>3m)', 'Healthy']
            df_inventory['Health_Status'] = np.select(conditions, choices, default='Unknown')
        else:
            # Simple classification based on stock quantity
            conditions = [
                (df_inventory['Stock_Qty'] == 0),
                (df_inventory['Stock_Qty'] < 10),
                (df_inventory['Stock_Qty'] > 100),
                (df_inventory['Stock_Qty'] >= 10) & (df_inventory['Stock_Qty'] <= 100)
            ]
            choices = ['Out of Stock', 'Low Stock', 'High Stock', 'Normal']
            df_inventory['Health_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Count by status
        status_counts = df_inventory['Health_Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Create donut chart
        base = alt.Chart(status_counts).encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color("Status:N", 
                          scale=alt.Scale(
                              domain=['Out of Stock', 'Critical (<0.5m)', 'Excess (>3m)', 'Healthy', 'Low Stock', 'High Stock', 'Normal'],
                              range=['#FF5252', '#FF9800', '#FFD740', '#4CAF50', '#FF9800', '#2196F3', '#4CAF50']
                          ),
                          legend=alt.Legend(title="Stock Status", columns=2)),
            tooltip=['Status:N', 'Count:Q']
        )
        
        donut = base.mark_arc(innerRadius=60, outerRadius=120)
        text = base.mark_text(radius=140, size=12).encode(text="Count:Q")
        
        chart = (donut + text).properties(
            height=350,
            title="Inventory Health Distribution"
        )
        
        return chart
        
    except Exception as e:
        st.error(f"Inventory chart error: {str(e)}")
        return None

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
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())

# Calculate metrics
forecast_metrics = calculate_forecast_accuracy(df_forecast, df_sales)
inventory_metrics = calculate_inventory_metrics(df_stock, df_product, df_sales)

# --- SIDEBAR FILTERS & CONTROLS ---
with st.sidebar:
    st.markdown("### 🔍 Dashboard Controls")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Data Filters")
    
    # SKU Filter
    sku_options = ['All SKUs']
    if not df_product.empty and 'SKU_ID' in df_product.columns:
        sku_options.extend(df_product['SKU_ID'].dropna().unique().tolist())
    
    selected_sku = st.selectbox(
        "Filter by SKU",
        options=sku_options,
        index=0,
        help="Select specific SKU or view all"
    )
    
    # Tier Filter
    tier_options = ['All Tiers']
    if not df_product.empty and 'SKU_Tier' in df_product.columns:
        tier_options.extend(df_product['SKU_Tier'].dropna().unique().tolist())
    
    selected_tier = st.selectbox(
        "Filter by Tier",
        options=tier_options,
        index=0
    )
    
    # Brand Filter
    brand_options = ['All Brands']
    if not df_product.empty and 'Brand' in df_product.columns:
        brand_options.extend(df_product['Brand'].dropna().unique().tolist())
    
    selected_brand = st.selectbox(
        "Filter by Brand",
        options=brand_options,
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📈 Data Summary")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if not df_product.empty:
            st.metric("Total SKUs", len(df_product))
    with col_s2:
        if not df_stock.empty:
            st.metric("In Stock", len(df_stock))
    
    # Data quality indicators
    st.markdown("#### 📋 Data Quality")
    
    data_issues = []
    if df_product.empty:
        data_issues.append("⚠️ Product Master kosong")
    if df_sales.empty:
        data_issues.append("⚠️ Sales data kosong")
    if df_forecast.empty:
        data_issues.append("⚠️ Forecast data kosong")
    if df_stock.empty:
        data_issues.append("⚠️ Stock data kosong")
    
    if data_issues:
        for issue in data_issues:
            st.warning(issue)
    else:
        st.success("✅ Semua data terload dengan baik")

# --- MAIN DASHBOARD CONTENT ---

# Header Metrics
st.subheader("🎯 Key Performance Indicators")

# Create 5 metric columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    accuracy = forecast_metrics.get('overall_accuracy', 0)
    accuracy_color = "#4CAF50" if accuracy >= 85 else "#FF9800" if accuracy >= 70 else "#F44336"
    st.markdown(f"""
    <div class="metric-card card-green">
        <div style="font-size: 0.8rem; opacity: 0.9; color: {accuracy_color};">Forecast Accuracy</div>
        <div style="font-size: 1.6rem; font-weight: bold; color: {accuracy_color};">{accuracy:.1f}%</div>
        <div style="font-size: 0.7rem;">MAPE: {forecast_metrics.get('mape', 0):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_stock = inventory_metrics.get('total_stock_qty', 0)
    st.markdown(f"""
    <div class="metric-card card-blue">
        <div style="font-size: 0.8rem; opacity: 0.9;">Total Inventory</div>
        <div style="font-size: 1.6rem; font-weight: bold;">{total_stock:,.0f}</div>
        <div style="font-size: 0.7rem;">units across {inventory_metrics.get('total_skus', 0)} SKUs</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    out_of_stock = inventory_metrics.get('out_of_stock', 0)
    oos_color = "#F44336" if out_of_stock > 0 else "#4CAF50"
    st.markdown(f"""
    <div class="metric-card card-red">
        <div style="font-size: 0.8rem; opacity: 0.9; color: {oos_color};">Out of Stock</div>
        <div style="font-size: 1.6rem; font-weight: bold; color: {oos_color};">{out_of_stock}</div>
        <div style="font-size: 0.7rem;">SKUs with zero stock</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if 'avg_cover_months' in inventory_metrics:
        cover = inventory_metrics['avg_cover_months']
        cover_color = "#4CAF50" if 1 <= cover <= 3 else "#FF9800" if cover < 1 else "#F44336"
        st.markdown(f"""
        <div class="metric-card card-orange">
            <div style="font-size: 0.8rem; opacity: 0.9; color: {cover_color};">Avg Stock Cover</div>
            <div style="font-size: 1.6rem; font-weight: bold; color: {cover_color};">{cover:.1f}</div>
            <div style="font-size: 0.7rem;">months of supply</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card card-orange">
            <div style="font-size: 0.8rem; opacity: 0.9;">Stock Analysis</div>
            <div style="font-size: 1.6rem; font-weight: bold;">N/A</div>
            <div style="font-size: 0.7rem;">Need sales data</div>
        </div>
        """, unsafe_allow_html=True)

with col5:
    bias = forecast_metrics.get('bias', 0)
    bias_text = f"{bias:+.0f}"
    bias_color = "#4CAF50" if abs(bias) < 10 else "#FF9800" if abs(bias) < 20 else "#F44336"
    bias_label = "Over-forecast" if bias > 0 else "Under-forecast" if bias < 0 else "Neutral"
    st.markdown(f"""
    <div class="metric-card card-purple">
        <div style="font-size: 0.8rem; opacity: 0.9; color: {bias_color};">Forecast Bias</div>
        <div style="font-size: 1.6rem; font-weight: bold; color: {bias_color};">{bias_text}</div>
        <div style="font-size: 0.7rem;">{bias_label}</div>
    </div>
    """, unsafe_allow_html=True)

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
    st.subheader("📊 Forecast Performance Analysis")
    
    if df_forecast.empty or df_sales.empty:
        st.warning("⚠️ Forecast atau sales data tidak tersedia untuk analisis")
    else:
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            # Forecast vs Sales Chart
            chart = create_forecast_vs_sales_chart(df_forecast, df_sales, selected_sku)
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Tidak cukup data untuk membuat chart")
        
        with col_f2:
            st.subheader("🎯 SKU Accuracy Ranking")
            
            if 'sku_accuracy_df' in forecast_metrics and not forecast_metrics['sku_accuracy_df'].empty:
                df_sku_acc = forecast_metrics['sku_accuracy_df'].sort_values('SKU_Accuracy', ascending=False)
                
                # Display top 10 SKUs
                st.dataframe(
                    df_sku_acc.head(10),
                    column_config={
                        "SKU_ID": "SKU",
                        "SKU_Accuracy": st.column_config.ProgressColumn(
                            "Accuracy",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                    use_container_width=True,
                    height=350
                )
            else:
                st.info("Tidak ada data accuracy per SKU")
        
        # Monthly Accuracy Trend
        st.subheader("📅 Monthly Accuracy Trend")
        
        if 'monthly_accuracy' in forecast_metrics and not forecast_metrics['monthly_accuracy'].empty:
            df_monthly_acc = forecast_metrics['monthly_accuracy'].sort_values('Month')
            
            # Line chart for monthly accuracy
            accuracy_chart = alt.Chart(df_monthly_acc).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Month:T', title='Month'),
                y=alt.Y('Monthly_Accuracy:Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Month:T', alt.Tooltip('Monthly_Accuracy', format='.1f')]
            ).properties(height=300)
            
            # Add target line at 85%
            target_line = alt.Chart(pd.DataFrame({'y': [85]})).mark_rule(
                strokeDash=[5, 5], color='red', strokeWidth=1
            ).encode(y='y:Q')
            
            st.altair_chart(accuracy_chart + target_line, use_container_width=True)
        
        # Forecast Error Analysis
        st.subheader("🔍 Error Analysis")
        
        col_err1, col_err2, col_err3 = st.columns(3)
        
        with col_err1:
            st.metric(
                "High Accuracy SKUs",
                f"{forecast_metrics.get('high_accuracy_skus', 0)}",
                f"of {forecast_metrics.get('sku_count', 0)} total"
            )
        
        with col_err2:
            bias = forecast_metrics.get('bias', 0)
            st.metric(
                "Forecast Bias",
                f"{bias:+.0f} units",
                "Over-forecast" if bias > 0 else "Under-forecast" if bias < 0 else "Neutral"
            )
        
        with col_err3:
            mape = forecast_metrics.get('mape', 0)
            st.metric(
                "Mean Absolute % Error",
                f"{mape:.1f}%",
                "Lower is better"
            )

# --- TAB 2: INVENTORY HEALTH ---
with tab2:
    st.subheader("📦 Inventory Status Dashboard")
    
    if df_stock.empty:
        st.warning("⚠️ Stock data tidak tersedia")
    else:
        # Apply filters to inventory data
        df_inventory_filtered = inventory_metrics.get('inventory_df', pd.DataFrame()).copy()
        
        if not df_inventory_filtered.empty:
            if selected_tier != 'All Tiers' and 'SKU_Tier' in df_inventory_filtered.columns:
                df_inventory_filtered = df_inventory_filtered[df_inventory_filtered['SKU_Tier'] == selected_tier]
            
            if selected_brand != 'All Brands' and 'Brand' in df_inventory_filtered.columns:
                df_inventory_filtered = df_inventory_filtered[df_inventory_filtered['Brand'] == selected_brand]
            
            if selected_sku != 'All SKUs':
                df_inventory_filtered = df_inventory_filtered[df_inventory_filtered['SKU_ID'] == selected_sku]
        
        col_inv1, col_inv2 = st.columns([2, 1])
        
        with col_inv1:
            # Inventory Health Chart
            health_chart = create_inventory_health_chart(df_inventory_filtered)
            if health_chart:
                st.altair_chart(health_chart, use_container_width=True)
            else:
                st.info("Tidak cukup data untuk health chart")
        
        with col_inv2:
            st.subheader("⚠️ Critical Alerts")
            
            if not df_inventory_filtered.empty:
                # Find critical items
                critical_items = []
                
                # Out of stock
                oos_items = df_inventory_filtered[df_inventory_filtered['Stock_Qty'] == 0]
                if not oos_items.empty:
                    critical_items.append(f"🔴 **{len(oos_items)} SKUs** out of stock")
                
                # Low cover (if calculated)
                if 'Cover_Months' in df_inventory_filtered.columns:
                    low_cover = df_inventory_filtered[df_inventory_filtered['Cover_Months'] < 0.5]
                    if not low_cover.empty:
                        critical_items.append(f"🟡 **{len(low_cover)} SKUs** with < 0.5 months cover")
                
                # Excess stock
                if 'Cover_Months' in df_inventory_filtered.columns:
                    excess = df_inventory_filtered[df_inventory_filtered['Cover_Months'] > 3]
                    if not excess.empty:
                        critical_items.append(f"🟠 **{len(excess)} SKUs** with > 3 months cover")
                
                if critical_items:
                    for item in critical_items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("""
                    <div class="alert-green">
                        ✅ No critical alerts - Inventory is healthy!
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tier-wise summary
            st.subheader("🏷️ Stock by Tier")
            
            if 'tier_summary' in inventory_metrics:
                df_tier = inventory_metrics['tier_summary']
                
                # Bar chart for tier distribution
                tier_chart = alt.Chart(df_tier).mark_bar().encode(
                    x=alt.X('Tier:N', title='Tier'),
                    y=alt.Y('Total_Stock:Q', title='Total Stock'),
                    color=alt.Color('Tier:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Tier', 'SKU_Count', 'Total_Stock']
                ).properties(height=200)
                
                st.altair_chart(tier_chart, use_container_width=True)
        
        # Detailed Inventory Table
        st.subheader("📋 Detailed Stock Position")
        
        if not df_inventory_filtered.empty:
            # Select columns to display
            display_cols = ['SKU_ID', 'Stock_Qty']
            
            if 'SKU_Tier' in df_inventory_filtered.columns:
                display_cols.append('SKU_Tier')
            if 'Brand' in df_inventory_filtered.columns:
                display_cols.append('Brand')
            if 'Cover_Months' in df_inventory_filtered.columns:
                display_cols.append('Cover_Months')
            if 'Health_Status' in df_inventory_filtered.columns:
                display_cols.append('Health_Status')
            
            # Format the dataframe for display
            df_display = df_inventory_filtered[display_cols].sort_values('Stock_Qty', ascending=False)
            
            st.dataframe(
                df_display,
                column_config={
                    "SKU_ID": "SKU ID",
                    "Stock_Qty": st.column_config.NumberColumn("Stock Qty", format="%d"),
                    "SKU_Tier": "Tier",
                    "Brand": "Brand",
                    "Cover_Months": st.column_config.NumberColumn("Cover (Months)", format="%.1f"),
                    "Health_Status": "Status"
                },
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Stock Report (CSV)",
                data=csv,
                file_name=f"stock_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# --- TAB 3: SMART RECOMMENDATIONS ---
with tab3:
    st.subheader("🤖 AI-Powered Stock Recommendations")
    
    if df_stock.empty or df_product.empty:
        st.warning("⚠️ Stock atau product data tidak tersedia untuk rekomendasi")
    else:
        # Generate recommendations
        recommendations = generate_stock_recommendations(
            inventory_metrics.get('inventory_df', pd.DataFrame()),
            df_product,
            df_sales
        )
        
        if recommendations.empty:
            st.info("Tidak ada rekomendasi yang dapat dihasilkan")
        else:
            # Priority recommendations (urgent)
            urgent_recs = recommendations[recommendations['Priority'] <= 2]
            
            if not urgent_recs.empty:
                st.markdown(f"""
                <div class="alert-red">
                    <h3>🚨 URGENT ACTIONS REQUIRED</h3>
                    <p><strong>{len(urgent_recs)} SKUs</strong> need immediate attention</p>
                    <p>Total recommended purchase: <strong>{urgent_recs['Recommended_Qty'].sum():,.0f} units</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display urgent recommendations
                st.dataframe(
                    urgent_recs[['SKU_ID', 'Current_Stock', 'Safety_Stock_Level', 
                               'MOQ', 'Recommended_Qty', 'Status']],
                    column_config={
                        "SKU_ID": "SKU ID",
                        "Current_Stock": st.column_config.NumberColumn("Current Stock", format="%d"),
                        "Safety_Stock_Level": st.column_config.NumberColumn("Safety Stock", format="%d"),
                        "MOQ": st.column_config.NumberColumn("MOQ", format="%d"),
                        "Recommended_Qty": st.column_config.NumberColumn("Rec. Order Qty", format="%d"),
                        "Status": "Action Required"
                    },
                    use_container_width=True,
                    height=300
                )
            
            # All recommendations
            st.subheader("📊 All SKU Recommendations")
            
            # Visualization: Current vs Recommended
            chart_data = recommendations.melt(
                id_vars=['SKU_ID'],
                value_vars=['Current_Stock', 'Safety_Stock_Level'],
                var_name='Stock_Type',
                value_name='Quantity'
            )
            
            # Get top 15 SKUs by safety stock
            top_skus = recommendations.nlargest(15, 'Safety_Stock_Level')['SKU_ID'].tolist()
            chart_data = chart_data[chart_data['SKU_ID'].isin(top_skus)]
            
            if not chart_data.empty:
                bars = alt.Chart(chart_data).mark_bar(size=12).encode(
                    x=alt.X('SKU_ID:N', title='SKU', sort='-y'),
                    y=alt.Y('Quantity:Q', title='Quantity'),
                    color=alt.Color('Stock_Type:N', 
                                  scale=alt.Scale(domain=['Current_Stock', 'Safety_Stock_Level'],
                                                range=['#667eea', '#ff6b6b']),
                                  legend=alt.Legend(title="Stock Type")),
                    column='Stock_Type:N',
                    tooltip=['SKU_ID', 'Stock_Type', alt.Tooltip('Quantity', format=',.0f')]
                ).properties(
                    height=300,
                    title="Current Stock vs Recommended Safety Stock (Top 15)"
                )
                
                st.altair_chart(bars, use_container_width=True)
            
            # Download all recommendations
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download All Recommendations (CSV)",
                data=csv,
                file_name=f"all_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# --- TAB 4: SALES ANALYTICS ---
with tab4:
    st.subheader("📈 Sales Performance Analysis")
    
    if df_sales.empty:
        st.warning("⚠️ Sales data tidak tersedia")
    else:
        # Sales trend over time
        st.subheader("📊 Monthly Sales Trend")
        
        # Aggregate sales by month
        monthly_sales = df_sales.groupby('Month')['Sales_Qty'].sum().reset_index()
        
        if not monthly_sales.empty:
            # Line chart
            trend_chart = alt.Chart(monthly_sales).mark_line(point=True, size=3).encode(
                x=alt.X('Month:T', title='Month', axis=alt.Axis(format="%b %Y")),
                y=alt.Y('Sales_Qty:Q', title='Total Sales (Units)'),
                tooltip=['Month:T', alt.Tooltip('Sales_Qty', format=',.0f')]
            ).properties(height=400)
            
            st.altair_chart(trend_chart, use_container_width=True)
        
        # Top performing SKUs
        st.subheader("🏆 Top Performing SKUs")
        
        col_top1, col_top2 = st.columns(2)
        
        with col_top1:
            # Total sales by SKU
            sku_sales_total = df_sales.groupby('SKU_ID')['Sales_Qty'].sum().reset_index()
            sku_sales_total = sku_sales_total.sort_values('Sales_Qty', ascending=False).head(10)
            
            if not sku_sales_total.empty:
                bars = alt.Chart(sku_sales_total).mark_bar().encode(
                    y=alt.Y('SKU_ID:N', title='SKU', sort='-x'),
                    x=alt.X('Sales_Qty:Q', title='Total Sales'),
                    color=alt.value('#667eea'),
                    tooltip=['SKU_ID', alt.Tooltip('Sales_Qty', format=',.0f')]
                ).properties(height=350, title="Top 10 SKUs by Total Sales")
                
                st.altair_chart(bars, use_container_width=True)
        
        with col_top2:
            # Monthly sales variability
            if 'SKU_ID' in df_sales.columns and len(df_sales['SKU_ID'].unique()) > 1:
                # Calculate coefficient of variation per SKU
                sku_variability = df_sales.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
                sku_variability['CoV'] = (sku_variability['std'] / sku_variability['mean']) * 100
                sku_variability = sku_variability.sort_values('CoV', ascending=False).head(10)
                
                if not sku_variability.empty:
                    bars_var = alt.Chart(sku_variability).mark_bar().encode(
                        y=alt.Y('SKU_ID:N', title='SKU', sort='-x'),
                        x=alt.X('CoV:Q', title='Variability (CoV %)'),
                        color=alt.value('#ff6b6b'),
                        tooltip=['SKU_ID', alt.Tooltip('CoV', format='.1f')]
                    ).properties(height=350, title="Top 10 Most Variable SKUs")
                    
                    st.altair_chart(bars_var, use_container_width=True)

# --- TAB 5: DATA EXPLORER ---
with tab5:
    st.subheader("📋 Raw Data Explorer")
    
    # Dataset selection
    dataset_options = {
        "Product Master": df_product,
        "Sales Data": df_sales,
        "Forecast Data": df_forecast,
        "Stock Data": df_stock,
        "PO Data": df_po
    }
    
    selected_data = st.selectbox("Select Dataset", list(dataset_options.keys()))
    df_selected = dataset_options[selected_data]
    
    if not df_selected.empty:
        # Data preview
        st.write(f"**Shape:** {df_selected.shape[0]} rows × {df_selected.shape[1]} columns")
        
        # Column selector
        if st.checkbox("Select specific columns", key="col_select"):
            all_columns = df_selected.columns.tolist()
            selected_columns = st.multiselect(
                "Choose columns to display",
                options=all_columns,
                default=all_columns[:min(10, len(all_columns))]
            )
            df_display = df_selected[selected_columns]
        else:
            df_display = df_selected
        
        # Dataframe display
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        
        # Numeric columns statistics
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            stats = df_selected[numeric_cols].describe().round(2)
            st.dataframe(stats, use_container_width=True)
        
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
        st.warning(f"Dataset {selected_data} is empty or could not be loaded")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Inventory Intelligence Dashboard v2.0 | Built with Streamlit | Data Source: Google Sheets</p>
    <p>For professional inventory control and demand planning</p>
</div>
""", unsafe_allow_html=True)
