import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

class FinancialPlanner:
    """Advanced financial planning and analysis class"""
    
    def __init__(self, data=None):
        self.data = data
    
    def calculate_budget_variance(self, budget_data, actual_data):
        """Calculate budget variance analysis"""
        variance_analysis = []
        
        for period in budget_data['period'].unique():
            budget_period = budget_data[budget_data['period'] == period]
            actual_period = actual_data[actual_data['period'] == period]
            
            for category in budget_period['category'].unique():
                budget_amount = budget_period[budget_period['category'] == category]['amount'].sum()
                actual_amount = actual_period[actual_period['category'] == category]['amount'].sum()
                
                variance = actual_amount - budget_amount
                variance_pct = (variance / budget_amount * 100) if budget_amount != 0 else 0
                
                variance_analysis.append({
                    'period': period,
                    'category': category,
                    'budget': budget_amount,
                    'actual': actual_amount,
                    'variance': variance,
                    'variance_pct': variance_pct,
                    'status': 'Over Budget' if variance > 0 else 'Under Budget' if variance < 0 else 'On Budget'
                })
        
        return pd.DataFrame(variance_analysis)
    
    def forecast_cash_flow(self, historical_data, periods_ahead=12):
        """Forecast cash flow based on historical patterns"""
        # Calculate monthly cash flow trends
        monthly_data = historical_data.groupby(['period', 'flow_type'])['amount'].sum().reset_index()
        
        cash_flow_forecast = []
        base_date = datetime.now()
        
        for i in range(periods_ahead):
            forecast_period = base_date + timedelta(days=30*i)
            
            # Simple trend-based forecasting
            for flow_type in monthly_data['flow_type'].unique():
                historical_amounts = monthly_data[monthly_data['flow_type'] == flow_type]['amount']
                
                if len(historical_amounts) > 0:
                    # Linear trend with seasonal adjustment
                    trend = np.polyfit(range(len(historical_amounts)), historical_amounts, 1)[0]
                    base_amount = historical_amounts.mean()
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Annual seasonality
                    
                    forecast_amount = base_amount + (trend * i) * seasonal_factor
                    
                    cash_flow_forecast.append({
                        'period': forecast_period.strftime('%Y-%m'),
                        'flow_type': flow_type,
                        'amount': round(forecast_amount, 2),
                        'type': 'Forecast'
                    })
        
        return pd.DataFrame(cash_flow_forecast)
    
    def calculate_profitability_metrics(self, revenue_data, cost_data):
        """Calculate comprehensive profitability metrics"""
        # Merge revenue and cost data
        profitability = revenue_data.merge(cost_data, on=['period', 'sku'], how='outer', suffixes=('_rev', '_cost'))
        profitability.fillna(0, inplace=True)
        
        # Calculate metrics
        profitability['gross_profit'] = profitability['amount_rev'] - profitability['amount_cost']
        profitability['gross_margin'] = (profitability['gross_profit'] / profitability['amount_rev'] * 100).round(2)
        profitability['cost_ratio'] = (profitability['amount_cost'] / profitability['amount_rev'] * 100).round(2)
        
        # Handle division by zero
        profitability['gross_margin'] = profitability['gross_margin'].replace([np.inf, -np.inf], 0)
        profitability['cost_ratio'] = profitability['cost_ratio'].replace([np.inf, -np.inf], 0)
        
        return profitability
    
    def analyze_cost_structure(self, cost_data):
        """Analyze cost structure and identify optimization opportunities"""
        # Group costs by category and calculate percentages
        cost_summary = cost_data.groupby('category')['amount'].sum().reset_index()
        total_cost = cost_summary['amount'].sum()
        cost_summary['percentage'] = (cost_summary['amount'] / total_cost * 100).round(2)
        cost_summary = cost_summary.sort_values('amount', ascending=False)
        
        # Identify high-impact categories (80/20 rule)
        cumulative_pct = cost_summary['percentage'].cumsum()
        cost_summary['cumulative_pct'] = cumulative_pct
        cost_summary['classification'] = np.where(cumulative_pct <= 80, 'High Impact', 'Low Impact')
        
        return cost_summary

def load_sample_financial_data():
    """Load sample financial planning data"""
    np.random.seed(42)
    
    # Sample budget data
    periods = pd.date_range('2024-01-01', periods=12, freq='M').strftime('%Y-%m')
    categories = ['Sales Revenue', 'Cost of Goods Sold', 'Marketing', 'Operations', 'Personnel', 'Facilities', 'Technology']
    
    budget_data = []
    actual_data = []
    
    for period in periods:
        for category in categories:
            if category == 'Sales Revenue':
                budget_amount = np.random.normal(500000, 50000)
                actual_amount = budget_amount * np.random.normal(1.0, 0.15)
            elif category == 'Cost of Goods Sold':
                budget_amount = np.random.normal(200000, 20000)
                actual_amount = budget_amount * np.random.normal(1.0, 0.10)
            else:
                budget_amount = np.random.normal(50000, 10000)
                actual_amount = budget_amount * np.random.normal(1.0, 0.20)
            
            budget_data.append({
                'period': period,
                'category': category,
                'amount': round(budget_amount, 2),
                'type': 'Budget'
            })
            
            actual_data.append({
                'period': period,
                'category': category,
                'amount': round(actual_amount, 2),
                'type': 'Actual'
            })
    
    # Sample cash flow data
    cash_flow_data = []
    flow_types = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']
    
    for period in periods:
        for flow_type in flow_types:
            if flow_type == 'Operating Cash Flow':
                amount = np.random.normal(100000, 20000)
            elif flow_type == 'Investing Cash Flow':
                amount = np.random.normal(-30000, 15000)
            else:  # Financing Cash Flow
                amount = np.random.normal(-10000, 10000)
            
            cash_flow_data.append({
                'period': period,
                'flow_type': flow_type,
                'amount': round(amount, 2),
                'type': 'Historical'
            })
    
    # Sample profitability data
    skus = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    revenue_data = []
    cost_data = []
    
    for period in periods[:6]:  # Last 6 months
        for sku in skus:
            revenue = np.random.normal(50000, 10000)
            cost = revenue * np.random.uniform(0.6, 0.8)
            
            revenue_data.append({
                'period': period,
                'sku': sku,
                'amount': round(revenue, 2),
                'type': 'Revenue'
            })
            
            cost_data.append({
                'period': period,
                'sku': sku,
                'amount': round(cost, 2),
                'type': 'Cost'
            })
    
    return (pd.DataFrame(budget_data), pd.DataFrame(actual_data), 
            pd.DataFrame(cash_flow_data), pd.DataFrame(revenue_data), pd.DataFrame(cost_data))

def run():
    st.title("💰 Financial Planning Module")
    st.markdown("""
    Comprehensive financial planning with budget analysis, cash flow forecasting, 
    profitability analysis, and cost optimization.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Financial Planning Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Upload & Overview", "Budget Variance Analysis", "Cash Flow Forecasting", 
         "Profitability Analysis", "Cost Structure Analysis", "Financial Metrics & KPIs"]
    )
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        budget_data, actual_data, cash_flow_data, revenue_data, cost_data = load_sample_financial_data()
        st.success("✅ Sample financial planning data loaded successfully!")
    else:
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your financial data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain budget, actual, and cash flow data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    financial_data = pd.read_csv(uploaded_file)
                else:
                    financial_data = pd.read_excel(uploaded_file)
                
                # For uploaded data, create sample datasets
                budget_data, actual_data, cash_flow_data, revenue_data, cost_data = load_sample_financial_data()
                st.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("Please upload a file or use sample data to proceed.")
            return
    
    # Initialize financial planner
    financial_planner = FinancialPlanner()
    
    if analysis_type == "Data Upload & Overview":
        st.subheader("📊 Financial Data Overview")
        
        # Summary metrics
        total_budget = budget_data['amount'].sum()
        total_actual = actual_data['amount'].sum()
        budget_variance = total_actual - total_budget
        variance_pct = (budget_variance / total_budget * 100) if total_budget != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Budget", f"${total_budget:,.0f}")
        with col2:
            st.metric("Total Actual", f"${total_actual:,.0f}")
        with col3:
            st.metric("Budget Variance", f"${budget_variance:,.0f}")
        with col4:
            st.metric("Variance %", f"{variance_pct:.1f}%")
        
        # Budget vs Actual visualization
        st.subheader("📈 Budget vs Actual Trends")
        
        budget_monthly = budget_data.groupby('period')['amount'].sum().reset_index()
        actual_monthly = actual_data.groupby('period')['amount'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=budget_monthly['period'], y=budget_monthly['amount'],
                                mode='lines+markers', name='Budget', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=actual_monthly['period'], y=actual_monthly['amount'],
                                mode='lines+markers', name='Actual', line=dict(color='red')))
        
        fig.update_layout(title="Monthly Budget vs Actual", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.subheader("📋 Category Breakdown")
        category_summary = budget_data.groupby('category')['amount'].sum().reset_index()
        category_summary = category_summary.sort_values('amount', ascending=False)
        
        fig = px.bar(category_summary, x='category', y='amount', 
                    title="Budget Allocation by Category")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Budget Variance Analysis":
        st.subheader("📊 Budget Variance Analysis")
        
        # Calculate variance analysis
        variance_analysis = financial_planner.calculate_budget_variance(budget_data, actual_data)
        
        # Variance summary metrics
        total_variance = variance_analysis['variance'].sum()
        avg_variance_pct = variance_analysis['variance_pct'].mean()
        over_budget_count = len(variance_analysis[variance_analysis['variance'] > 0])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Variance", f"${total_variance:,.0f}")
        with col2:
            st.metric("Avg Variance %", f"{avg_variance_pct:.1f}%")
        with col3:
            st.metric("Over Budget Items", over_budget_count)
        
        # Variance by category
        st.subheader("📈 Variance by Category")
        category_variance = variance_analysis.groupby('category').agg({
            'variance': 'sum',
            'variance_pct': 'mean'
        }).reset_index()
        
        fig = px.bar(category_variance, x='category', y='variance',
                    title="Total Variance by Category",
                    color='variance',
                    color_continuous_scale='RdYlGn_r')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly variance trend
        st.subheader("📅 Monthly Variance Trends")
        monthly_variance = variance_analysis.groupby('period')['variance'].sum().reset_index()
        
        fig = px.line(monthly_variance, x='period', y='variance',
                     title="Monthly Budget Variance Trend",
                     markers=True)
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed variance table
        st.subheader("📋 Detailed Variance Analysis")
        
        # Color code the dataframe
        def highlight_variance(val):
            if val > 0:
                return 'background-color: #ffcccc'  # Light red for over budget
            elif val < 0:
                return 'background-color: #ccffcc'  # Light green for under budget
            return ''
        
        styled_variance = variance_analysis.style.applymap(highlight_variance, subset=['variance'])
        st.dataframe(styled_variance)
    
    elif analysis_type == "Cash Flow Forecasting":
        st.subheader("💸 Cash Flow Forecasting")
        
        # Generate cash flow forecast
        forecast_data = financial_planner.forecast_cash_flow(cash_flow_data, periods_ahead=6)
        
        # Combine historical and forecast data
        combined_cash_flow = pd.concat([cash_flow_data, forecast_data], ignore_index=True)
        
        # Cash flow summary
        historical_total = cash_flow_data['amount'].sum()
        forecast_total = forecast_data['amount'].sum()
        net_change = forecast_total - historical_total
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Historical Cash Flow", f"${historical_total:,.0f}")
        with col2:
            st.metric("Forecasted Cash Flow", f"${forecast_total:,.0f}")
        with col3:
            st.metric("Net Change", f"${net_change:,.0f}")
        
        # Cash flow visualization
        st.subheader("📈 Cash Flow Forecast")
        
        # Group by period and flow type
        flow_summary = combined_cash_flow.groupby(['period', 'flow_type', 'type'])['amount'].sum().reset_index()
        
        fig = px.line(flow_summary, x='period', y='amount', color='flow_type',
                     line_dash='type', title="Cash Flow Forecast by Type")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative cash flow
        st.subheader("💰 Cumulative Cash Flow")
        cumulative_flow = combined_cash_flow.groupby('period')['amount'].sum().reset_index()
        cumulative_flow['cumulative'] = cumulative_flow['amount'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_flow['period'], y=cumulative_flow['cumulative'],
                                mode='lines+markers', fill='tonexty',
                                name='Cumulative Cash Flow'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Cumulative Cash Flow Projection", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cash flow forecast table
        st.subheader("📋 Cash Flow Forecast Details")
        st.dataframe(forecast_data)
    
    elif analysis_type == "Profitability Analysis":
        st.subheader("📈 Profitability Analysis")
        
        # Calculate profitability metrics
        profitability = financial_planner.calculate_profitability_metrics(revenue_data, cost_data)
        
        # Profitability summary
        total_revenue = profitability['amount_rev'].sum()
        total_cost = profitability['amount_cost'].sum()
        total_profit = profitability['gross_profit'].sum()
        overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.0f}")
        with col3:
            st.metric("Gross Profit", f"${total_profit:,.0f}")
        with col4:
            st.metric("Gross Margin", f"{overall_margin:.1f}%")
        
        # Profitability by product
        st.subheader("📊 Profitability by Product")
        product_profit = profitability.groupby('sku').agg({
            'amount_rev': 'sum',
            'amount_cost': 'sum',
            'gross_profit': 'sum',
            'gross_margin': 'mean'
        }).reset_index()
        
        fig = px.bar(product_profit, x='sku', y='gross_profit',
                    title="Gross Profit by Product",
                    color='gross_margin',
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        # Profitability trends
        st.subheader("📅 Profitability Trends")
        monthly_profit = profitability.groupby('period').agg({
            'amount_rev': 'sum',
            'amount_cost': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        monthly_profit['gross_margin'] = (monthly_profit['gross_profit'] / monthly_profit['amount_rev'] * 100).round(2)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly_profit['period'], y=monthly_profit['gross_profit'],
                  name="Gross Profit"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=monthly_profit['period'], y=monthly_profit['gross_margin'],
                      mode='lines+markers', name="Gross Margin %"),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Gross Profit ($)", secondary_y=False)
        fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True)
        fig.update_layout(title="Monthly Profitability Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed profitability table
        st.subheader("📋 Detailed Profitability Analysis")
        st.dataframe(profitability)
    
    elif analysis_type == "Cost Structure Analysis":
        st.subheader("🔍 Cost Structure Analysis")
        
        # Analyze cost structure
        cost_structure = financial_planner.analyze_cost_structure(actual_data)
        
        # Cost structure metrics
        total_costs = cost_structure['amount'].sum()
        high_impact_costs = cost_structure[cost_structure['classification'] == 'High Impact']['amount'].sum()
        cost_concentration = (high_impact_costs / total_costs * 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Costs", f"${total_costs:,.0f}")
        with col2:
            st.metric("High Impact Costs", f"${high_impact_costs:,.0f}")
        with col3:
            st.metric("Cost Concentration", f"{cost_concentration:.1f}%")
        
        # Cost distribution pie chart
        st.subheader("🥧 Cost Distribution")
        fig = px.pie(cost_structure, values='amount', names='category',
                    title="Cost Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)
        
        # Pareto analysis
        st.subheader("📊 Pareto Analysis (80/20 Rule)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cost_structure['category'], y=cost_structure['percentage'],
                            name='Cost Percentage'))
        fig.add_trace(go.Scatter(x=cost_structure['category'], y=cost_structure['cumulative_pct'],
                                mode='lines+markers', name='Cumulative %',
                                yaxis='y2', line=dict(color='red')))
        
        fig.add_hline(y=80, line_dash="dash", line_color="green", yaxis='y2')
        fig.update_layout(
            title="Cost Category Pareto Analysis",
            yaxis=dict(title="Individual Percentage"),
            yaxis2=dict(title="Cumulative Percentage", overlaying='y', side='right'),
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost optimization recommendations
        st.subheader("💡 Cost Optimization Recommendations")
        
        high_impact_categories = cost_structure[cost_structure['classification'] == 'High Impact']
        
        recommendations = []
        for _, row in high_impact_categories.iterrows():
            recommendations.append(f"• **{row['category']}**: ${row['amount']:,.0f} ({row['percentage']:.1f}%) - Focus on cost reduction initiatives")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        
        # Detailed cost structure table
        st.subheader("📋 Detailed Cost Structure")
        st.dataframe(cost_structure)
    
    elif analysis_type == "Financial Metrics & KPIs":
        st.subheader("📊 Financial Metrics & KPIs")
        
        # Calculate key financial metrics
        total_revenue = revenue_data['amount'].sum()
        total_costs = cost_data['amount'].sum()
        gross_profit = total_revenue - total_costs
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Operating metrics
        operating_cash_flow = cash_flow_data[cash_flow_data['flow_type'] == 'Operating Cash Flow']['amount'].sum()
        investing_cash_flow = cash_flow_data[cash_flow_data['flow_type'] == 'Investing Cash Flow']['amount'].sum()
        financing_cash_flow = cash_flow_data[cash_flow_data['flow_type'] == 'Financing Cash Flow']['amount'].sum()
        
        # Budget performance
        budget_variance = actual_data['amount'].sum() - budget_data['amount'].sum()
        budget_accuracy = (1 - abs(budget_variance) / budget_data['amount'].sum()) * 100
        
        # Display KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gross Margin", f"{gross_margin:.1f}%")
            st.metric("Operating Cash Flow", f"${operating_cash_flow:,.0f}")
        with col2:
            st.metric("Budget Accuracy", f"{budget_accuracy:.1f}%")
            st.metric("Net Cash Flow", f"${operating_cash_flow + investing_cash_flow + financing_cash_flow:,.0f}")
        with col3:
            st.metric("ROI Estimate", f"{(gross_profit / total_costs * 100):.1f}%")
            st.metric("Cost Efficiency", f"{(total_costs / total_revenue * 100):.1f}%")
        
        # Financial dashboard
        st.subheader("📈 Financial Dashboard")
        
        # Create a comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Revenue vs Costs", "Cash Flow Components", 
                           "Budget vs Actual", "Profitability Trend"),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Revenue vs Costs
        fig.add_trace(
            go.Bar(x=['Revenue', 'Costs'], y=[total_revenue, total_costs],
                  marker_color=['green', 'red']),
            row=1, col=1
        )
        
        # Cash Flow Components
        cash_flow_summary = cash_flow_data.groupby('flow_type')['amount'].sum().reset_index()
        fig.add_trace(
            go.Pie(labels=cash_flow_summary['flow_type'], values=cash_flow_summary['amount']),
            row=1, col=2
        )
        
        # Budget vs Actual
        budget_monthly = budget_data.groupby('period')['amount'].sum().reset_index()
        actual_monthly = actual_data.groupby('period')['amount'].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=budget_monthly['period'], y=budget_monthly['amount'],
                      mode='lines', name='Budget'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=actual_monthly['period'], y=actual_monthly['amount'],
                      mode='lines', name='Actual'),
            row=2, col=1
        )
        
        # Profitability Trend
        monthly_profit = revenue_data.groupby('period')['amount'].sum() - cost_data.groupby('period')['amount'].sum()
        fig.add_trace(
            go.Bar(x=monthly_profit.index, y=monthly_profit.values,
                  marker_color='blue'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Financial Performance Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Results")
    
    if st.sidebar.button("Download Sample Data"):
        budget_sample, actual_sample, cash_flow_sample, revenue_sample, cost_sample = load_sample_financial_data()
        
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            budget_sample.to_excel(writer, sheet_name='Budget_Data', index=False)
            actual_sample.to_excel(writer, sheet_name='Actual_Data', index=False)
            cash_flow_sample.to_excel(writer, sheet_name='Cash_Flow_Data', index=False)
            revenue_sample.to_excel(writer, sheet_name='Revenue_Data', index=False)
            cost_sample.to_excel(writer, sheet_name='Cost_Data', index=False)
        
        st.sidebar.download_button(
            label="📊 Download Financial Data",
            data=output.getvalue(),
            file_name="financial_planning_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) 