import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_forecast_accuracy_metrics(actual, forecast):
    """
    Calculate forecast accuracy metrics: Bias, MAE, MAPE, RMSE
    
    Parameters:
    actual (array-like): Actual sales values
    forecast (array-like): Forecasted values
    
    Returns:
    dict: Dictionary containing all accuracy metrics
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return {
            'bias': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'rmse': np.nan,
            'count': 0
        }
    
    # Calculate error
    error = forecast - actual
    
    # Bias (Mean Error)
    bias = np.mean(error)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(error))
    
    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    mape = np.mean(np.abs(error / np.where(actual == 0, 1, actual))) * 100
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(error ** 2))
    
    return {
        'bias': bias,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'count': len(actual)
    }

def process_forecast_data(df):
    """
    Process the uploaded forecast data and calculate accuracy metrics
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create year-month column for aggregation
    df['year_month'] = df['date'].dt.to_period('M')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate accuracy metrics for overall data
    overall_metrics = calculate_forecast_accuracy_metrics(df['actual_sales'], df['forecast'])
    
    # Calculate metrics by month
    monthly_metrics = []
    for period in df['year_month'].unique():
        period_data = df[df['year_month'] == period]
        metrics = calculate_forecast_accuracy_metrics(period_data['actual_sales'], period_data['forecast'])
        metrics['period'] = str(period)
        metrics['year'] = period.year
        metrics['month'] = period.month
        monthly_metrics.append(metrics)
    
    monthly_df = pd.DataFrame(monthly_metrics)
    
    # Calculate metrics by location
    location_metrics = []
    for location in df['location'].unique():
        location_data = df[df['location'] == location]
        metrics = calculate_forecast_accuracy_metrics(location_data['actual_sales'], location_data['forecast'])
        metrics['location'] = location
        location_metrics.append(metrics)
    
    location_df = pd.DataFrame(location_metrics)
    
    # Calculate metrics by classification
    classification_metrics = []
    for classification in df['classification'].unique():
        class_data = df[df['classification'] == classification]
        metrics = calculate_forecast_accuracy_metrics(class_data['actual_sales'], class_data['forecast'])
        metrics['classification'] = classification
        classification_metrics.append(metrics)
    
    classification_df = pd.DataFrame(classification_metrics)
    
    # Calculate metrics by SKU
    sku_metrics = []
    for sku in df['sku'].unique():
        sku_data = df[df['sku'] == sku]
        metrics = calculate_forecast_accuracy_metrics(sku_data['actual_sales'], sku_data['forecast'])
        metrics['sku'] = sku
        sku_metrics.append(metrics)
    
    sku_df = pd.DataFrame(sku_metrics)
    
    return {
        'overall': overall_metrics,
        'monthly': monthly_df,
        'location': location_df,
        'classification': classification_df,
        'sku': sku_df,
        'raw_data': df
    }

def create_sample_data():
    """
    Create sample forecast accuracy data for demonstration
    """
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Sample SKUs, locations, and classifications
    skus = ['SKU_001', 'SKU_002', 'SKU_003', 'SKU_004', 'SKU_005']
    locations = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
    classifications = ['Category_A', 'Category_B', 'Category_C']
    
    data = []
    
    for _ in range(1000):  # Generate 1000 sample records
        date = np.random.choice(dates)
        sku = np.random.choice(skus)
        location = np.random.choice(locations)
        classification = np.random.choice(classifications)
        
        # Generate realistic sales and forecast data
        base_demand = np.random.uniform(50, 500)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        actual_sales = max(0, base_demand * seasonal_factor + np.random.normal(0, base_demand * 0.1))
        
        # Forecast with some error
        forecast_error = np.random.normal(0, actual_sales * 0.15)
        forecast = max(0, actual_sales + forecast_error)
        
        data.append({
            'sku': sku,
            'actual_sales': round(actual_sales, 2),
            'forecast': round(forecast, 2),
            'date': date.strftime('%Y-%m-%d'),
            'location': location,
            'classification': classification
        })
    
    return pd.DataFrame(data)

def plot_accuracy_trends(monthly_df):
    """
    Create plots for forecast accuracy trends over time
    """
    # Sort by period
    monthly_df = monthly_df.sort_values('period')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Bias Over Time', 'MAE Over Time', 'MAPE Over Time', 'RMSE Over Time'],
        vertical_spacing=0.12
    )
    
    # Bias
    fig.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['bias'], 
                  mode='lines+markers', name='Bias', line=dict(color='blue')),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['mae'], 
                  mode='lines+markers', name='MAE', line=dict(color='green')),
        row=1, col=2
    )
    
    # MAPE
    fig.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['mape'], 
                  mode='lines+markers', name='MAPE (%)', line=dict(color='orange')),
        row=2, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['rmse'], 
                  mode='lines+markers', name='RMSE', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Forecast Accuracy Metrics Over Time",
        showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_accuracy_comparison(results):
    """
    Create comparison plots for different dimensions
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Location comparison
        if not results['location'].empty:
            fig_location = px.bar(
                results['location'], 
                x='location', 
                y=['bias', 'mae', 'mape', 'rmse'],
                title="Forecast Accuracy by Location",
                barmode='group'
            )
            fig_location.update_layout(height=400)
            st.plotly_chart(fig_location, use_container_width=True)
    
    with col2:
        # Classification comparison
        if not results['classification'].empty:
            fig_class = px.bar(
                results['classification'], 
                x='classification', 
                y=['bias', 'mae', 'mape', 'rmse'],
                title="Forecast Accuracy by Classification",
                barmode='group'
            )
            fig_class.update_layout(height=400)
            st.plotly_chart(fig_class, use_container_width=True)

def display_ytd_summary(monthly_df):
    """
    Display Year-to-Date summary metrics
    """
    current_year = datetime.now().year
    ytd_data = monthly_df[monthly_df['year'] == current_year]
    
    if ytd_data.empty:
        # If no current year data, use the latest year available
        latest_year = monthly_df['year'].max()
        ytd_data = monthly_df[monthly_df['year'] == latest_year]
        st.info(f"Showing {latest_year} data (no current year data available)")
    
    if not ytd_data.empty:
        st.subheader(f"📊 Year-to-Date Summary ({ytd_data['year'].iloc[0]})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_bias = ytd_data['bias'].mean()
            st.metric("Average Bias", f"{avg_bias:.2f}")
        
        with col2:
            avg_mae = ytd_data['mae'].mean()
            st.metric("Average MAE", f"{avg_mae:.2f}")
        
        with col3:
            avg_mape = ytd_data['mape'].mean()
            st.metric("Average MAPE", f"{avg_mape:.1f}%")
        
        with col4:
            avg_rmse = ytd_data['rmse'].mean()
            st.metric("Average RMSE", f"{avg_rmse:.2f}")

def run():
    st.title("📊 Demand Planning Module")
    st.markdown("### Forecast Accuracy Evaluation System")
    
    st.markdown("""
    Upload your forecast data to analyze accuracy metrics including:
    - **Bias**: Average forecast error (positive = over-forecast, negative = under-forecast)
    - **MAE**: Mean Absolute Error
    - **MAPE**: Mean Absolute Percentage Error
    - **RMSE**: Root Mean Square Error
    
    Required columns: `sku`, `actual_sales`, `forecast`, `date`, `location`, `classification`
    """)
    
    # File upload section
    st.subheader("📂 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your forecast data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain columns: sku, actual_sales, forecast, date, location, classification"
        )
    
    with col2:
        if st.button("📋 Use Sample Data", help="Load sample data for demonstration"):
            sample_data = create_sample_data()
            st.session_state['forecast_data'] = sample_data
            st.success("Sample data loaded successfully!")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['forecast_data'] = df
            st.success(f"File uploaded successfully! {len(df)} records loaded.")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Display analysis if data is available
    if 'forecast_data' in st.session_state:
        df = st.session_state['forecast_data']
        
        # Validate required columns
        required_columns = ['sku', 'actual_sales', 'forecast', 'date', 'location', 'classification']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your data has all required columns and try again.")
            return
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique SKUs", df['sku'].nunique())
            with col3:
                st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
        
        # Process data and calculate metrics
        with st.spinner("Calculating forecast accuracy metrics..."):
            results = process_forecast_data(df)
        
        # Display overall metrics
        st.subheader("🎯 Overall Forecast Accuracy")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bias = results['overall']['bias']
            bias_color = "normal" if abs(bias) < 10 else "inverse"
            st.metric("Bias", f"{bias:.2f}", delta=None)
            if bias > 0:
                st.caption("📈 Over-forecasting")
            elif bias < 0:
                st.caption("📉 Under-forecasting")
            else:
                st.caption("🎯 Perfect bias")
        
        with col2:
            mae = results['overall']['mae']
            st.metric("MAE", f"{mae:.2f}")
            st.caption("Lower is better")
        
        with col3:
            mape = results['overall']['mape']
            mape_color = "normal" if mape < 20 else "inverse"
            st.metric("MAPE", f"{mape:.1f}%")
            if mape < 10:
                st.caption("🟢 Excellent")
            elif mape < 20:
                st.caption("🟡 Good")
            else:
                st.caption("🔴 Needs improvement")
        
        with col4:
            rmse = results['overall']['rmse']
            st.metric("RMSE", f"{rmse:.2f}")
            st.caption("Lower is better")
        
        # Year-to-Date Summary
        if not results['monthly'].empty:
            display_ytd_summary(results['monthly'])
        
        # Monthly trends
        st.subheader("📈 Accuracy Trends Over Time")
        
        if not results['monthly'].empty:
            trend_fig = plot_accuracy_trends(results['monthly'])
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Monthly data table
            with st.expander("📊 Monthly Metrics Table", expanded=False):
                monthly_display = results['monthly'].copy()
                monthly_display = monthly_display.round(2)
                st.dataframe(monthly_display, use_container_width=True)
        
        # Comparison by dimensions
        st.subheader("🔍 Accuracy Comparison")
        plot_accuracy_comparison(results)
        
        # Detailed breakdowns
        st.subheader("📋 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["By Location", "By Classification", "By SKU"])
        
        with tab1:
            if not results['location'].empty:
                location_df = results['location'].round(2)
                st.dataframe(location_df, use_container_width=True)
        
        with tab2:
            if not results['classification'].empty:
                classification_df = results['classification'].round(2)
                st.dataframe(classification_df, use_container_width=True)
        
        with tab3:
            if not results['sku'].empty:
                sku_df = results['sku'].round(2).sort_values('mape')
                
                # Show top 10 best and worst performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🏆 **Top 10 Best Performers (Lowest MAPE)**")
                    st.dataframe(sku_df.head(10), use_container_width=True)
                
                with col2:
                    st.write("⚠️ **Top 10 Worst Performers (Highest MAPE)**")
                    st.dataframe(sku_df.tail(10), use_container_width=True)
                
                # Full SKU table in expander
                with st.expander("📊 All SKU Metrics", expanded=False):
                    st.dataframe(sku_df, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare summary report
            summary_data = {
                'Overall Metrics': [results['overall']],
                'Monthly Metrics': results['monthly'].to_dict('records'),
                'Location Metrics': results['location'].to_dict('records'),
                'Classification Metrics': results['classification'].to_dict('records'),
                'SKU Metrics': results['sku'].to_dict('records')
            }
            
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Overall metrics
                pd.DataFrame([results['overall']]).to_excel(writer, sheet_name='Overall', index=False)
                
                # Monthly metrics
                if not results['monthly'].empty:
                    results['monthly'].to_excel(writer, sheet_name='Monthly', index=False)
                
                # Location metrics
                if not results['location'].empty:
                    results['location'].to_excel(writer, sheet_name='Location', index=False)
                
                # Classification metrics
                if not results['classification'].empty:
                    results['classification'].to_excel(writer, sheet_name='Classification', index=False)
                
                # SKU metrics
                if not results['sku'].empty:
                    results['sku'].to_excel(writer, sheet_name='SKU', index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📊 Download Detailed Report (Excel)",
                data=excel_data,
                file_name=f"forecast_accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Prepare CSV summary
            if not results['monthly'].empty:
                csv_data = results['monthly'].to_csv(index=False)
                st.download_button(
                    label="📈 Download Monthly Trends (CSV)",
                    data=csv_data,
                    file_name=f"monthly_forecast_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please upload your forecast data or use sample data to begin analysis.")
        
        # Show expected data format
        st.subheader("📋 Expected Data Format")
        
        sample_format = pd.DataFrame({
            'sku': ['SKU_001', 'SKU_002', 'SKU_001'],
            'actual_sales': [100.5, 250.0, 150.2],
            'forecast': [95.0, 260.5, 145.8],
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'location': ['Store_A', 'Store_B', 'Store_A'],
            'classification': ['Category_A', 'Category_B', 'Category_A']
        })
        
        st.dataframe(sample_format, use_container_width=True)
        
        st.markdown("""
        **Column Descriptions:**
        - `sku`: Stock Keeping Unit identifier
        - `actual_sales`: Actual sales/demand values
        - `forecast`: Forecasted sales/demand values
        - `date`: Date in YYYY-MM-DD format
        - `location`: Store/warehouse/region identifier
        - `classification`: Product category/classification
        """) 