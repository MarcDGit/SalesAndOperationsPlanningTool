import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """Advanced inventory optimization and analysis class"""
    
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        
    def detect_outliers(self, method='isolation_forest'):
        """Detect outliers in demand data using multiple methods"""
        if 'demand' not in self.data.columns:
            st.error("Demand column not found in data")
            return None
            
        outliers_results = {}
        
        # Statistical outliers (Z-score method)
        if method in ['all', 'zscore']:
            z_scores = np.abs(stats.zscore(self.data['demand'].dropna()))
            outliers_results['zscore'] = self.data[z_scores > 3].copy()
            
        # IQR method
        if method in ['all', 'iqr']:
            Q1 = self.data['demand'].quantile(0.25)
            Q3 = self.data['demand'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_results['iqr'] = self.data[(self.data['demand'] < lower_bound) | 
                                               (self.data['demand'] > upper_bound)].copy()
            
        # Isolation Forest method
        if method in ['all', 'isolation_forest']:
            features = ['demand']
            if 'inventory' in self.data.columns:
                features.append('inventory')
                
            X = self.data[features].dropna()
            if len(X) > 10:  # Minimum samples for isolation forest
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(X_scaled)
                
                outliers_results['isolation_forest'] = self.data.iloc[X.index[outlier_labels == -1]].copy()
            
        return outliers_results
    
    def calculate_safety_stock(self, service_level=0.95):
        """Calculate safety stock based on demand variability"""
        if 'demand' not in self.data.columns:
            return None
            
        results = []
        
        for location in self.data['location'].unique():
            for sku in self.data['sku'].unique():
                subset = self.data[(self.data['location'] == location) & 
                                 (self.data['sku'] == sku)]
                
                if len(subset) > 1:
                    demand_mean = subset['demand'].mean()
                    demand_std = subset['demand'].std()
                    
                    # Z-score for service level
                    z_score = stats.norm.ppf(service_level)
                    safety_stock = z_score * demand_std
                    
                    results.append({
                        'location': location,
                        'sku': sku,
                        'avg_demand': demand_mean,
                        'demand_std': demand_std,
                        'safety_stock': safety_stock,
                        'service_level': service_level
                    })
                    
        return pd.DataFrame(results)
    
    def calculate_reorder_points(self, lead_time_weeks=2):
        """Calculate reorder points based on demand and lead time"""
        safety_stock_df = self.calculate_safety_stock()
        
        if safety_stock_df is not None:
            safety_stock_df['lead_time_demand'] = safety_stock_df['avg_demand'] * lead_time_weeks
            safety_stock_df['reorder_point'] = safety_stock_df['lead_time_demand'] + safety_stock_df['safety_stock']
            
        return safety_stock_df
    
    def abc_analysis(self):
        """Perform ABC analysis based on demand value"""
        if 'demand' not in self.data.columns:
            return None
            
        # Calculate total demand value per SKU
        sku_analysis = self.data.groupby('sku').agg({
            'demand': ['sum', 'mean', 'std'],
            'location': 'nunique'
        }).round(2)
        
        sku_analysis.columns = ['total_demand', 'avg_demand', 'demand_std', 'locations']
        sku_analysis = sku_analysis.reset_index()
        
        # Sort by total demand
        sku_analysis = sku_analysis.sort_values('total_demand', ascending=False)
        sku_analysis['cumulative_demand'] = sku_analysis['total_demand'].cumsum()
        sku_analysis['cumulative_percent'] = (sku_analysis['cumulative_demand'] / 
                                            sku_analysis['total_demand'].sum()) * 100
        
        # Assign ABC categories
        conditions = [
            sku_analysis['cumulative_percent'] <= 80,
            sku_analysis['cumulative_percent'] <= 95,
            sku_analysis['cumulative_percent'] > 95
        ]
        choices = ['A', 'B', 'C']
        sku_analysis['abc_category'] = np.select(conditions, choices)
        
        return sku_analysis
    
    def inventory_turnover_analysis(self):
        """Calculate inventory turnover metrics"""
        if 'inventory' not in self.data.columns:
            return None
            
        results = []
        
        for location in self.data['location'].unique():
            for sku in self.data['sku'].unique():
                subset = self.data[(self.data['location'] == location) & 
                                 (self.data['sku'] == sku)]
                
                if len(subset) > 1:
                    total_demand = subset['demand'].sum()
                    avg_inventory = subset['inventory'].mean()
                    
                    if avg_inventory > 0:
                        turnover_ratio = total_demand / avg_inventory
                        days_on_hand = 365 / (turnover_ratio * 52)  # Convert weekly to daily
                        
                        results.append({
                            'location': location,
                            'sku': sku,
                            'total_demand': total_demand,
                            'avg_inventory': avg_inventory,
                            'turnover_ratio': turnover_ratio,
                            'days_on_hand': days_on_hand
                        })
                        
        return pd.DataFrame(results)

def load_and_process_data(uploaded_file):
    """Load and process uploaded CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            return None
            
        # Convert week column to datetime if it exists
        if 'week' in df.columns:
            df['week'] = pd.to_datetime(df['week'], errors='coerce')
            
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def validate_data_structure(df):
    """Validate that the data has required columns"""
    required_columns = ['week', 'sku', 'location', 'demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.info("Required columns: week, sku, location, demand")
        st.info("Optional columns: inventory")
        return False
        
    return True

def run():
    st.title("🏭 Supply Chain Inventory Planning & Optimization")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Upload & Overview", "Outlier Detection", "Inventory Optimization", "Parameter Proposals", "ABC Analysis"]
    )
    
    # Main content area
    if analysis_type == "Data Upload & Overview":
        st.header("📤 Data Upload & Overview")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your inventory data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain columns: week, sku, location, demand, and optionally inventory"
        )
        
        if uploaded_file is not None:
            # Load data
            df = load_and_process_data(uploaded_file)
            
            if df is not None and validate_data_structure(df):
                # Store data in session state
                st.session_state['inventory_data'] = df
                
                # Display data overview
                st.success("✅ Data loaded successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Unique SKUs", df['sku'].nunique())
                with col3:
                    st.metric("Locations", df['location'].nunique())
                with col4:
                    if 'week' in df.columns:
                        st.metric("Time Period", f"{df['week'].min().strftime('%Y-%m-%d')} to {df['week'].max().strftime('%Y-%m-%d')}")
                
                # Data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10))
                
                # Basic statistics
                st.subheader("📈 Basic Statistics")
                st.dataframe(df.describe())
                
                # Time series visualization
                if 'week' in df.columns:
                    st.subheader("📅 Time Series Overview")
                    
                    # Aggregate demand by week
                    weekly_demand = df.groupby('week')['demand'].sum().reset_index()
                    
                    fig = px.line(weekly_demand, x='week', y='demand', 
                                title='Total Demand Over Time')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            # Show sample data format
            st.info("👆 Please upload your data file to get started")
            st.subheader("📋 Expected Data Format")
            
            sample_data = {
                'week': ['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22'],
                'sku': ['SKU001', 'SKU001', 'SKU002', 'SKU002'],
                'location': ['Warehouse_A', 'Warehouse_A', 'Warehouse_B', 'Warehouse_B'],
                'demand': [150, 200, 300, 180],
                'inventory': [500, 350, 800, 620]
            }
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)
            st.caption("inventory column is optional but recommended for advanced analysis")
    
    elif analysis_type == "Outlier Detection":
        st.header("🔍 Demand Outlier Detection")
        
        if 'inventory_data' in st.session_state:
            df = st.session_state['inventory_data']
            optimizer = InventoryOptimizer(df)
            
            # Outlier detection method selection
            method = st.selectbox(
                "Select Outlier Detection Method",
                ["isolation_forest", "zscore", "iqr", "all"],
                help="Isolation Forest: ML-based anomaly detection\nZ-Score: Statistical method (>3 std dev)\nIQR: Interquartile range method"
            )
            
            if st.button("🔎 Detect Outliers"):
                outliers = optimizer.detect_outliers(method)
                
                if outliers:
                    for method_name, outlier_data in outliers.items():
                        if not outlier_data.empty:
                            st.subheader(f"📊 {method_name.upper()} Method Results")
                            st.write(f"Found {len(outlier_data)} outliers")
                            
                            # Display outliers
                            st.dataframe(outlier_data)
                            
                            # Visualization
                            fig = px.scatter(df, x='week', y='demand', 
                                           title=f'Demand with {method_name.upper()} Outliers Highlighted',
                                           color='sku')
                            
                            # Highlight outliers
                            fig.add_scatter(x=outlier_data['week'], y=outlier_data['demand'],
                                          mode='markers', marker=dict(color='red', size=10),
                                          name='Outliers')
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No outliers detected using {method_name} method")
        else:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
    
    elif analysis_type == "Inventory Optimization":
        st.header("⚙️ Inventory Optimization")
        
        if 'inventory_data' in st.session_state:
            df = st.session_state['inventory_data']
            optimizer = InventoryOptimizer(df)
            
            # Parameter inputs
            col1, col2 = st.columns(2)
            with col1:
                service_level = st.slider("Service Level", 0.80, 0.99, 0.95, 0.01)
            with col2:
                lead_time = st.number_input("Lead Time (weeks)", 1, 12, 2)
            
            if st.button("🎯 Calculate Optimization Parameters"):
                
                # Safety stock calculation
                st.subheader("🛡️ Safety Stock Analysis")
                safety_stock_df = optimizer.calculate_safety_stock(service_level)
                
                if safety_stock_df is not None and not safety_stock_df.empty:
                    st.dataframe(safety_stock_df)
                    
                    # Visualization
                    fig = px.bar(safety_stock_df, x='sku', y='safety_stock', 
                               color='location', title='Safety Stock by SKU and Location')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Reorder points
                st.subheader("📋 Reorder Points")
                reorder_df = optimizer.calculate_reorder_points(lead_time)
                
                if reorder_df is not None and not reorder_df.empty:
                    st.dataframe(reorder_df)
                    
                    # Visualization
                    fig = px.scatter(reorder_df, x='avg_demand', y='reorder_point',
                                   color='location', size='safety_stock',
                                   title='Reorder Points vs Average Demand')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Inventory turnover analysis
                if 'inventory' in df.columns:
                    st.subheader("🔄 Inventory Turnover Analysis")
                    turnover_df = optimizer.inventory_turnover_analysis()
                    
                    if turnover_df is not None and not turnover_df.empty:
                        st.dataframe(turnover_df)
                        
                        # Visualization
                        fig = px.scatter(turnover_df, x='turnover_ratio', y='days_on_hand',
                                       color='location', size='total_demand',
                                       title='Inventory Turnover vs Days on Hand')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
    
    elif analysis_type == "Parameter Proposals":
        st.header("💡 Inventory Parameter Proposals")
        
        if 'inventory_data' in st.session_state:
            df = st.session_state['inventory_data']
            optimizer = InventoryOptimizer(df)
            
            st.subheader("🎯 Recommended Parameters")
            
            # Calculate various parameters
            with st.spinner("Calculating recommendations..."):
                
                # Service level recommendations
                service_levels = [0.90, 0.95, 0.98]
                recommendations = []
                
                for sl in service_levels:
                    safety_stock_df = optimizer.calculate_safety_stock(sl)
                    if safety_stock_df is not None:
                        avg_safety_stock = safety_stock_df['safety_stock'].mean()
                        recommendations.append({
                            'Service Level': f"{sl*100}%",
                            'Avg Safety Stock': round(avg_safety_stock, 2),
                            'Total Safety Stock': round(safety_stock_df['safety_stock'].sum(), 2)
                        })
                
                if recommendations:
                    rec_df = pd.DataFrame(recommendations)
                    st.subheader("📊 Service Level Impact")
                    st.dataframe(rec_df)
                    
                    # Visualization
                    fig = px.bar(rec_df, x='Service Level', y='Total Safety Stock',
                               title='Total Safety Stock Required by Service Level')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Demand pattern analysis
                st.subheader("📈 Demand Pattern Analysis")
                
                # Seasonal analysis if we have enough data
                if len(df) > 52:  # At least one year of weekly data
                    weekly_demand = df.groupby('week')['demand'].sum().reset_index()
                    weekly_demand.set_index('week', inplace=True)
                    
                    try:
                        decomposition = seasonal_decompose(weekly_demand['demand'], 
                                                         model='additive', period=52)
                        
                        # Plot seasonal decomposition
                        fig = make_subplots(rows=4, cols=1, 
                                          subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'])
                        
                        fig.add_trace(go.Scatter(x=weekly_demand.index, y=weekly_demand['demand'],
                                               name='Original'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=weekly_demand.index, y=decomposition.trend,
                                               name='Trend'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=weekly_demand.index, y=decomposition.seasonal,
                                               name='Seasonal'), row=3, col=1)
                        fig.add_trace(go.Scatter(x=weekly_demand.index, y=decomposition.resid,
                                               name='Residual'), row=4, col=1)
                        
                        fig.update_layout(height=800, title_text="Seasonal Decomposition")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.info("Unable to perform seasonal decomposition. Need more data points.")
        else:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
    
    elif analysis_type == "ABC Analysis":
        st.header("📊 ABC Analysis")
        
        if 'inventory_data' in st.session_state:
            df = st.session_state['inventory_data']
            optimizer = InventoryOptimizer(df)
            
            if st.button("🔍 Perform ABC Analysis"):
                abc_results = optimizer.abc_analysis()
                
                if abc_results is not None:
                    st.subheader("📈 ABC Classification Results")
                    st.dataframe(abc_results)
                    
                    # Summary by category
                    st.subheader("📊 Category Summary")
                    category_summary = abc_results.groupby('abc_category').agg({
                        'sku': 'count',
                        'total_demand': 'sum',
                        'avg_demand': 'mean'
                    }).round(2)
                    category_summary.columns = ['SKU Count', 'Total Demand', 'Avg Demand']
                    st.dataframe(category_summary)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of SKU distribution
                        fig1 = px.pie(abc_results, names='abc_category', title='SKU Distribution by ABC Category')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Bar chart of demand distribution
                        category_demand = abc_results.groupby('abc_category')['total_demand'].sum().reset_index()
                        fig2 = px.bar(category_demand, x='abc_category', y='total_demand',
                                    title='Total Demand by ABC Category')
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Pareto chart
                    st.subheader("📈 Pareto Analysis")
                    fig3 = go.Figure()
                    
                    # Bar chart for individual SKU demand
                    fig3.add_trace(go.Bar(x=abc_results['sku'], y=abc_results['total_demand'],
                                        name='Demand', yaxis='y'))
                    
                    # Line chart for cumulative percentage
                    fig3.add_trace(go.Scatter(x=abc_results['sku'], y=abc_results['cumulative_percent'],
                                            mode='lines', name='Cumulative %', yaxis='y2'))
                    
                    # Add 80% line
                    fig3.add_hline(y=80, line_dash="dash", line_color="red", 
                                 annotation_text="80%", yref='y2')
                    
                    fig3.update_layout(
                        title='Pareto Analysis - Demand Distribution',
                        xaxis_title='SKU',
                        yaxis=dict(title='Demand', side='left'),
                        yaxis2=dict(title='Cumulative %', side='right', overlaying='y'),
                        height=500
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:**")
    st.markdown("- Use ABC analysis to prioritize inventory management efforts")
    st.markdown("- Monitor outliers regularly to identify demand anomalies")
    st.markdown("- Adjust safety stock based on service level requirements")
    st.markdown("- Consider seasonal patterns when setting reorder points")