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
    Calculate comprehensive forecast accuracy metrics including industry-standard accuracy percentages
    
    Parameters:
    actual (array-like): Actual sales values
    forecast (array-like): Forecasted values
    
    Returns:
    dict: Dictionary containing all accuracy metrics including 0-100% accuracy scales
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
            'forecast_accuracy': np.nan,
            'weighted_accuracy': np.nan,
            'tracking_signal': np.nan,
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
    
    # Industry-Standard Accuracy Metrics (0-100% scale)
    
    # 1. Forecast Accuracy (100% - MAPE) - Most common industry standard
    forecast_accuracy = max(0, 100 - mape)
    
    # 2. Weighted Accuracy - Based on relative error magnitude
    # Uses a more sophisticated formula that considers the distribution of errors
    relative_errors = np.abs(error / np.where(actual == 0, 1, actual))
    # Cap individual errors at 200% to prevent extreme outliers from dominating
    capped_relative_errors = np.minimum(relative_errors, 2.0)
    weighted_accuracy = max(0, 100 - (np.mean(capped_relative_errors) * 100))
    
    # 3. Tracking Signal - Bias to MAE ratio (industry standard for bias detection)
    # Typical acceptable range is -4 to +4
    tracking_signal = bias / mae if mae != 0 else 0
    
    return {
        'bias': bias,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'forecast_accuracy': forecast_accuracy,  # 0-100% scale
        'weighted_accuracy': weighted_accuracy,   # 0-100% scale  
        'tracking_signal': tracking_signal,       # -∞ to +∞ (typical: -4 to +4)
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
    
    # Add overall totals and averages
    overall_metrics['total_actual'] = df['actual_sales'].sum()
    overall_metrics['total_forecast'] = df['forecast'].sum()
    overall_metrics['avg_actual'] = df['actual_sales'].mean()
    overall_metrics['avg_forecast'] = df['forecast'].mean()
    
    # Calculate metrics by month
    monthly_metrics = []
    for period in df['year_month'].unique():
        period_data = df[df['year_month'] == period]
        metrics = calculate_forecast_accuracy_metrics(period_data['actual_sales'], period_data['forecast'])
        metrics['period'] = str(period)
        metrics['year'] = period.year
        metrics['month'] = period.month
        metrics['total_actual'] = period_data['actual_sales'].sum()
        metrics['total_forecast'] = period_data['forecast'].sum()
        metrics['avg_actual'] = period_data['actual_sales'].mean()
        metrics['avg_forecast'] = period_data['forecast'].mean()
        monthly_metrics.append(metrics)
    
    monthly_df = pd.DataFrame(monthly_metrics)
    
    # Calculate metrics by location
    location_metrics = []
    for location in df['location'].unique():
        location_data = df[df['location'] == location]
        metrics = calculate_forecast_accuracy_metrics(location_data['actual_sales'], location_data['forecast'])
        metrics['location'] = location
        metrics['total_actual'] = location_data['actual_sales'].sum()
        metrics['total_forecast'] = location_data['forecast'].sum()
        metrics['avg_actual'] = location_data['actual_sales'].mean()
        metrics['avg_forecast'] = location_data['forecast'].mean()
        location_metrics.append(metrics)
    
    location_df = pd.DataFrame(location_metrics)
    
    # Calculate metrics by classification
    classification_metrics = []
    for classification in df['classification'].unique():
        class_data = df[df['classification'] == classification]
        metrics = calculate_forecast_accuracy_metrics(class_data['actual_sales'], class_data['forecast'])
        metrics['classification'] = classification
        metrics['total_actual'] = class_data['actual_sales'].sum()
        metrics['total_forecast'] = class_data['forecast'].sum()
        metrics['avg_actual'] = class_data['actual_sales'].mean()
        metrics['avg_forecast'] = class_data['forecast'].mean()
        classification_metrics.append(metrics)
    
    classification_df = pd.DataFrame(classification_metrics)
    
    # Calculate metrics by SKU
    sku_metrics = []
    for sku in df['sku'].unique():
        sku_data = df[df['sku'] == sku]
        metrics = calculate_forecast_accuracy_metrics(sku_data['actual_sales'], sku_data['forecast'])
        metrics['sku'] = sku
        metrics['total_actual'] = sku_data['actual_sales'].sum()
        metrics['total_forecast'] = sku_data['forecast'].sum()
        metrics['avg_actual'] = sku_data['actual_sales'].mean()
        metrics['avg_forecast'] = sku_data['forecast'].mean()
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
        # Convert numpy datetime64 to pandas datetime to access dayofyear
        date_pd = pd.to_datetime(date)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date_pd.dayofyear / 365)
        actual_sales = max(0, base_demand * seasonal_factor + np.random.normal(0, base_demand * 0.1))
        
        # Forecast with some error
        forecast_error = np.random.normal(0, actual_sales * 0.15)
        forecast = max(0, actual_sales + forecast_error)
        
        data.append({
            'sku': sku,
            'actual_sales': round(actual_sales, 2),
            'forecast': round(forecast, 2),
            'date': date_pd.strftime('%Y-%m-%d'),
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
    
    # First, create accuracy percentage trends chart
    st.subheader("📈 Accuracy Trends (0-100% Scale)")
    
    fig_acc = go.Figure()
    
    fig_acc.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['forecast_accuracy'],
                  mode='lines+markers', name='Forecast Accuracy (%)',
                  line=dict(color='#2E8B57', width=3),
                  marker=dict(size=8))
    )
    
    fig_acc.add_trace(
        go.Scatter(x=monthly_df['period'], y=monthly_df['weighted_accuracy'],
                  mode='lines+markers', name='Weighted Accuracy (%)',
                  line=dict(color='#4682B4', width=3),
                  marker=dict(size=8))
    )
    
    # Add benchmark lines
    fig_acc.add_hline(y=90, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (90%)")
    fig_acc.add_hline(y=80, line_dash="dash", line_color="orange", 
                     annotation_text="Good (80%)")
    
    fig_acc.update_layout(
        height=400,
        title_text="Industry-Standard Accuracy Metrics Over Time",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        xaxis_title="Period",
        showlegend=True
    )
    
    fig_acc.update_xaxes(tickangle=45)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Traditional metrics chart
    st.subheader("📊 Traditional Metrics Trends")
    
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
        # Location comparison - Traditional Metrics
        if not results['location'].empty:
            fig_location = px.bar(
                results['location'], 
                x='location', 
                y=['bias', 'mae', 'mape', 'rmse'],
                title="Traditional Metrics by Location",
                barmode='group'
            )
            fig_location.update_layout(height=400)
            st.plotly_chart(fig_location, use_container_width=True)
            
        # Location comparison - Accuracy Percentages
        if not results['location'].empty:
            fig_location_acc = px.bar(
                results['location'], 
                x='location', 
                y=['forecast_accuracy', 'weighted_accuracy'],
                title="Forecast Accuracy % by Location",
                barmode='group',
                color_discrete_sequence=['#2E8B57', '#4682B4']
            )
            fig_location_acc.update_layout(height=400, yaxis_title="Accuracy (%)")
            fig_location_acc.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_location_acc, use_container_width=True)
    
    with col2:
        # Classification comparison - Traditional Metrics
        if not results['classification'].empty:
            fig_class = px.bar(
                results['classification'], 
                x='classification', 
                y=['bias', 'mae', 'mape', 'rmse'],
                title="Traditional Metrics by Classification",
                barmode='group'
            )
            fig_class.update_layout(height=400)
            st.plotly_chart(fig_class, use_container_width=True)
            
        # Classification comparison - Accuracy Percentages
        if not results['classification'].empty:
            fig_class_acc = px.bar(
                results['classification'], 
                x='classification', 
                y=['forecast_accuracy', 'weighted_accuracy'],
                title="Forecast Accuracy % by Classification",
                barmode='group',
                color_discrete_sequence=['#2E8B57', '#4682B4']
            )
            fig_class_acc.update_layout(height=400, yaxis_title="Accuracy (%)")
            fig_class_acc.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_class_acc, use_container_width=True)

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
        
        # Actual vs Forecast Summary
        st.markdown("### 📊 **Actual vs Forecast Summary**")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            total_actual = results['overall']['total_actual']
            st.metric("Total Actual Sales", f"{total_actual:,.0f}")
        
        with summary_col2:
            total_forecast = results['overall']['total_forecast']
            st.metric("Total Forecast", f"{total_forecast:,.0f}")
        
        with summary_col3:
            variance = total_forecast - total_actual
            variance_pct = (variance / total_actual * 100) if total_actual != 0 else 0
            st.metric("Total Variance", f"{variance:,.0f}", delta=f"{variance_pct:+.1f}%")
        
        with summary_col4:
            avg_actual = results['overall']['avg_actual']
            avg_forecast = results['overall']['avg_forecast']
            st.metric("Avg Actual", f"{avg_actual:.1f}")
            st.caption(f"Avg Forecast: {avg_forecast:.1f}")
        
        st.markdown("---")
        
        # Industry-Standard Accuracy Metrics (0-100% scale)
        st.markdown("### 🎯 **Industry-Standard Accuracy**")
        acc_col1, acc_col2, acc_col3 = st.columns(3)
        
        with acc_col1:
            forecast_acc = results['overall']['forecast_accuracy']
            acc_color = "normal" if forecast_acc >= 80 else "inverse"
            st.metric("Forecast Accuracy", f"{forecast_acc:.1f}%", delta=None)
            if forecast_acc >= 90:
                st.caption("🟢 Excellent (≥90%)")
            elif forecast_acc >= 80:
                st.caption("🟡 Good (80-90%)")
            else:
                st.caption("🔴 Needs improvement (<80%)")
        
        with acc_col2:
            weighted_acc = results['overall']['weighted_accuracy']
            st.metric("Weighted Accuracy", f"{weighted_acc:.1f}%")
            if weighted_acc >= 85:
                st.caption("🟢 Excellent")
            elif weighted_acc >= 70:
                st.caption("🟡 Good")
            else:
                st.caption("🔴 Needs improvement")
        
        with acc_col3:
            tracking_signal = results['overall']['tracking_signal']
            ts_color = "normal" if abs(tracking_signal) < 4 else "inverse"
            st.metric("Tracking Signal", f"{tracking_signal:.2f}")
            if abs(tracking_signal) < 2:
                st.caption("🟢 In control (<2)")
            elif abs(tracking_signal) < 4:
                st.caption("🟡 Monitor (2-4)")
            else:
                st.caption("🔴 Out of control (≥4)")
        
        st.markdown("---")
        st.markdown("### 📈 **Traditional Metrics**")
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
                
                # Reorder columns to show actual/forecast first
                cols = ['period', 'total_actual', 'total_forecast', 'avg_actual', 'avg_forecast',
                       'forecast_accuracy', 'weighted_accuracy', 'bias', 'mae', 'mape', 'rmse', 'tracking_signal', 'count']
                monthly_display = monthly_display[[col for col in cols if col in monthly_display.columns]]
                
                # Round numeric columns
                numeric_cols = monthly_display.select_dtypes(include=[np.number]).columns
                monthly_display[numeric_cols] = monthly_display[numeric_cols].round(2)
                
                st.dataframe(monthly_display, use_container_width=True)
        
        # Comparison by dimensions
        st.subheader("🔍 Accuracy Comparison")
        plot_accuracy_comparison(results)
        
        # Detailed breakdowns
        st.subheader("📋 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["By Location", "By Classification", "By SKU", "Raw Data"])
        
        with tab1:
            if not results['location'].empty:
                st.markdown("#### 📍 **Performance by Location**")
                location_df = results['location'].copy()
                
                # Reorder columns to show actual/forecast first
                cols = ['location', 'total_actual', 'total_forecast', 'avg_actual', 'avg_forecast', 
                       'forecast_accuracy', 'weighted_accuracy', 'bias', 'mae', 'mape', 'rmse', 'tracking_signal', 'count']
                location_df = location_df[[col for col in cols if col in location_df.columns]]
                
                # Round numeric columns
                numeric_cols = location_df.select_dtypes(include=[np.number]).columns
                location_df[numeric_cols] = location_df[numeric_cols].round(2)
                
                st.dataframe(location_df, use_container_width=True)
        
        with tab2:
            if not results['classification'].empty:
                st.markdown("#### 🏷️ **Performance by Classification**")
                classification_df = results['classification'].copy()
                
                # Reorder columns to show actual/forecast first
                cols = ['classification', 'total_actual', 'total_forecast', 'avg_actual', 'avg_forecast',
                       'forecast_accuracy', 'weighted_accuracy', 'bias', 'mae', 'mape', 'rmse', 'tracking_signal', 'count']
                classification_df = classification_df[[col for col in cols if col in classification_df.columns]]
                
                # Round numeric columns
                numeric_cols = classification_df.select_dtypes(include=[np.number]).columns
                classification_df[numeric_cols] = classification_df[numeric_cols].round(2)
                
                st.dataframe(classification_df, use_container_width=True)
        
        with tab3:
            if not results['sku'].empty:
                st.markdown("#### 📦 **Performance by SKU**")
                sku_df = results['sku'].copy()
                
                # Reorder columns to show actual/forecast first
                cols = ['sku', 'total_actual', 'total_forecast', 'avg_actual', 'avg_forecast',
                       'forecast_accuracy', 'weighted_accuracy', 'bias', 'mae', 'mape', 'rmse', 'tracking_signal', 'count']
                sku_df = sku_df[[col for col in cols if col in sku_df.columns]]
                
                # Round numeric columns
                numeric_cols = sku_df.select_dtypes(include=[np.number]).columns
                sku_df[numeric_cols] = sku_df[numeric_cols].round(2)
                
                # Sort by forecast accuracy (descending)
                sku_df = sku_df.sort_values('forecast_accuracy', ascending=False)
                
                # Show top 10 best and worst performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🏆 **Top 10 Best Performers (Highest Accuracy)**")
                    best_performers = sku_df.head(10)
                    st.dataframe(best_performers, use_container_width=True)
                
                with col2:
                    st.write("⚠️ **Bottom 10 Performers (Lowest Accuracy)**")
                    worst_performers = sku_df.tail(10)
                    st.dataframe(worst_performers, use_container_width=True)
                
                # Full SKU table
                with st.expander("📊 Complete SKU Analysis", expanded=False):
                    st.dataframe(sku_df, use_container_width=True)
        
        with tab4:
            st.markdown("#### 📋 **Raw Data with Calculated Errors**")
            raw_data = results['raw_data'].copy()
            
            # Add calculated fields
            raw_data['error'] = raw_data['forecast'] - raw_data['actual_sales']
            raw_data['abs_error'] = abs(raw_data['error'])
            raw_data['pct_error'] = (raw_data['error'] / raw_data['actual_sales'] * 100).round(2)
            raw_data['abs_pct_error'] = abs(raw_data['pct_error'])
            
            # Reorder columns
            cols = ['date', 'sku', 'location', 'classification', 'actual_sales', 'forecast', 
                   'error', 'abs_error', 'pct_error', 'abs_pct_error']
            raw_data = raw_data[[col for col in cols if col in raw_data.columns]]
            
            # Round numeric columns
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            raw_data[numeric_cols] = raw_data[numeric_cols].round(2)
            
            # Show summary statistics
            st.markdown("##### Summary Statistics")
            summary_stats = pd.DataFrame({
                'Metric': ['Count', 'Total Actual', 'Total Forecast', 'Avg Actual', 'Avg Forecast', 
                          'Total Error', 'Avg Error', 'Avg Abs Error', 'Avg % Error'],
                'Value': [
                    len(raw_data),
                    f"{raw_data['actual_sales'].sum():,.0f}",
                    f"{raw_data['forecast'].sum():,.0f}",
                    f"{raw_data['actual_sales'].mean():.2f}",
                    f"{raw_data['forecast'].mean():.2f}",
                    f"{raw_data['error'].sum():,.0f}",
                    f"{raw_data['error'].mean():.2f}",
                    f"{raw_data['abs_error'].mean():.2f}",
                    f"{raw_data['abs_pct_error'].mean():.2f}%"
                ]
            })
            st.dataframe(summary_stats, use_container_width=True)
            
            # Show raw data table
            st.markdown("##### Detailed Records")
            st.dataframe(raw_data, use_container_width=True)
        
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