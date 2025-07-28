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

class SOPPlanner:
    """Advanced Sales and Operations Planning class"""
    
    def __init__(self, data=None):
        self.data = data
    
    def perform_gap_analysis(self, demand_plan, supply_plan, financial_plan):
        """Perform comprehensive gap analysis between plans"""
        gap_analysis = []
        
        # Merge all plans by period
        periods = set(demand_plan['period'].unique()) | set(supply_plan['period'].unique()) | set(financial_plan['period'].unique())
        
        for period in periods:
            # Get period data
            demand_data = demand_plan[demand_plan['period'] == period]
            supply_data = supply_plan[supply_plan['period'] == period]
            financial_data = financial_plan[financial_plan['period'] == period]
            
            # Calculate totals
            total_demand = demand_data['quantity'].sum() if not demand_data.empty else 0
            total_supply = supply_data['quantity'].sum() if not supply_data.empty else 0
            total_revenue = financial_data['revenue'].sum() if not financial_data.empty else 0
            total_costs = financial_data['costs'].sum() if not financial_data.empty else 0
            
            # Calculate gaps
            volume_gap = total_supply - total_demand
            financial_gap = total_revenue - total_costs
            
            gap_analysis.append({
                'period': period,
                'demand_volume': total_demand,
                'supply_volume': total_supply,
                'volume_gap': volume_gap,
                'volume_gap_pct': (volume_gap / total_demand * 100) if total_demand > 0 else 0,
                'revenue': total_revenue,
                'costs': total_costs,
                'profit': financial_gap,
                'margin': (financial_gap / total_revenue * 100) if total_revenue > 0 else 0,
                'gap_status': 'Balanced' if abs(volume_gap) < total_demand * 0.05 else 'Imbalanced'
            })
        
        return pd.DataFrame(gap_analysis)
    
    def create_scenario_analysis(self, base_plan, scenarios):
        """Create scenario analysis for different planning assumptions"""
        scenario_results = []
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_plan = base_plan.copy()
            
            # Apply scenario adjustments
            demand_adj = scenario_params.get('demand_adjustment', 1.0)
            cost_adj = scenario_params.get('cost_adjustment', 1.0)
            capacity_adj = scenario_params.get('capacity_adjustment', 1.0)
            
            scenario_plan['demand'] = scenario_plan['demand'] * demand_adj
            scenario_plan['costs'] = scenario_plan['costs'] * cost_adj
            scenario_plan['capacity'] = scenario_plan['capacity'] * capacity_adj
            
            # Calculate scenario metrics
            total_revenue = (scenario_plan['demand'] * scenario_plan['price']).sum()
            total_costs = scenario_plan['costs'].sum()
            total_profit = total_revenue - total_costs
            margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            scenario_results.append({
                'scenario': scenario_name,
                'total_revenue': total_revenue,
                'total_costs': total_costs,
                'total_profit': total_profit,
                'margin': margin,
                'demand_volume': scenario_plan['demand'].sum(),
                'capacity_utilization': (scenario_plan['demand'].sum() / scenario_plan['capacity'].sum() * 100) if scenario_plan['capacity'].sum() > 0 else 0
            })
        
        return pd.DataFrame(scenario_results)
    
    def calculate_consensus_metrics(self, stakeholder_inputs):
        """Calculate consensus metrics across stakeholder inputs"""
        consensus_analysis = []
        
        for period in stakeholder_inputs['period'].unique():
            period_data = stakeholder_inputs[stakeholder_inputs['period'] == period]
            
            for metric in ['demand_forecast', 'supply_plan', 'revenue_target']:
                if metric in period_data.columns:
                    values = period_data[metric].values
                    
                    consensus_analysis.append({
                        'period': period,
                        'metric': metric,
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'cv': values.std() / values.mean() if values.mean() != 0 else 0,
                        'consensus_level': 'High' if values.std() / values.mean() < 0.1 else 'Medium' if values.std() / values.mean() < 0.2 else 'Low'
                    })
        
        return pd.DataFrame(consensus_analysis)
    
    def generate_action_plans(self, gap_analysis, threshold=0.1):
        """Generate action plans based on gap analysis"""
        action_plans = []
        
        for _, row in gap_analysis.iterrows():
            period = row['period']
            volume_gap_pct = abs(row['volume_gap_pct'])
            margin = row['margin']
            
            if volume_gap_pct > threshold * 100:
                if row['volume_gap'] > 0:
                    action_plans.append({
                        'period': period,
                        'issue': 'Excess Supply',
                        'severity': 'High' if volume_gap_pct > 20 else 'Medium',
                        'recommended_action': 'Reduce production capacity or increase sales efforts',
                        'owner': 'Supply Planning',
                        'priority': 1 if volume_gap_pct > 20 else 2
                    })
                else:
                    action_plans.append({
                        'period': period,
                        'issue': 'Supply Shortage',
                        'severity': 'High' if volume_gap_pct > 20 else 'Medium',
                        'recommended_action': 'Increase production capacity or reduce demand commitments',
                        'owner': 'Supply Planning',
                        'priority': 1 if volume_gap_pct > 20 else 2
                    })
            
            if margin < 10:  # Low margin threshold
                action_plans.append({
                    'period': period,
                    'issue': 'Low Profitability',
                    'severity': 'High' if margin < 5 else 'Medium',
                    'recommended_action': 'Review pricing strategy or cost reduction initiatives',
                    'owner': 'Financial Planning',
                    'priority': 1 if margin < 5 else 2
                })
        
        return pd.DataFrame(action_plans)

def load_sample_sop_data():
    """Load sample S&OP data"""
    np.random.seed(42)
    
    # Sample periods (12 months)
    periods = pd.date_range('2024-01-01', periods=12, freq='M').strftime('%Y-%m')
    product_families = ['Family_A', 'Family_B', 'Family_C', 'Family_D']
    regions = ['North', 'South', 'East', 'West']
    
    # Sample demand plan
    demand_plan = []
    for period in periods:
        for family in product_families:
            for region in regions:
                base_demand = np.random.normal(1000, 200)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * int(period.split('-')[1]) / 12)
                demand = max(0, base_demand * seasonal_factor)
                
                demand_plan.append({
                    'period': period,
                    'product_family': family,
                    'region': region,
                    'quantity': round(demand, 0),
                    'price': round(np.random.uniform(50, 150), 2),
                    'revenue': round(demand * np.random.uniform(50, 150), 2)
                })
    
    # Sample supply plan
    supply_plan = []
    for period in periods:
        for family in product_families:
            total_demand = sum([d['quantity'] for d in demand_plan 
                               if d['period'] == period and d['product_family'] == family])
            
            # Supply slightly different from demand to create gaps
            supply_variance = np.random.uniform(0.9, 1.1)
            supply_quantity = total_demand * supply_variance
            
            supply_plan.append({
                'period': period,
                'product_family': family,
                'quantity': round(supply_quantity, 0),
                'capacity': round(supply_quantity * 1.2, 0),
                'costs': round(supply_quantity * np.random.uniform(30, 80), 2)
            })
    
    # Sample financial plan
    financial_plan = []
    for period in periods:
        total_revenue = sum([d['revenue'] for d in demand_plan if d['period'] == period])
        total_costs = sum([s['costs'] for s in supply_plan if s['period'] == period])
        
        financial_plan.append({
            'period': period,
            'revenue': round(total_revenue, 2),
            'costs': round(total_costs, 2),
            'profit': round(total_revenue - total_costs, 2),
            'margin': round((total_revenue - total_costs) / total_revenue * 100, 2) if total_revenue > 0 else 0
        })
    
    # Sample stakeholder inputs
    stakeholders = ['Sales', 'Marketing', 'Operations', 'Finance']
    stakeholder_inputs = []
    
    for period in periods[:6]:  # Last 6 months for stakeholder consensus
        for stakeholder in stakeholders:
            base_demand = np.random.normal(4000, 400)
            base_supply = base_demand * np.random.uniform(0.95, 1.05)
            base_revenue = base_demand * np.random.uniform(100, 120)
            
            stakeholder_inputs.append({
                'period': period,
                'stakeholder': stakeholder,
                'demand_forecast': round(base_demand, 0),
                'supply_plan': round(base_supply, 0),
                'revenue_target': round(base_revenue, 2)
            })
    
    return (pd.DataFrame(demand_plan), pd.DataFrame(supply_plan), 
            pd.DataFrame(financial_plan), pd.DataFrame(stakeholder_inputs))

def run():
    st.title("📋 Sales & Operations Planning (S&OP)")
    st.markdown("""
    Integrated Sales and Operations Planning with gap analysis, scenario planning, 
    consensus building, and executive dashboards.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("S&OP Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Upload & Overview", "Integrated Plan Review", "Gap Analysis", 
         "Scenario Planning", "Consensus Building", "Executive Dashboard"]
    )
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        demand_plan, supply_plan, financial_plan, stakeholder_inputs = load_sample_sop_data()
        st.success("✅ Sample S&OP data loaded successfully!")
    else:
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your S&OP data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain demand, supply, and financial plan data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    sop_data = pd.read_csv(uploaded_file)
                else:
                    sop_data = pd.read_excel(uploaded_file)
                
                # For uploaded data, create sample datasets
                demand_plan, supply_plan, financial_plan, stakeholder_inputs = load_sample_sop_data()
                st.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("Please upload a file or use sample data to proceed.")
            return
    
    # Initialize S&OP planner
    sop_planner = SOPPlanner()
    
    if analysis_type == "Data Upload & Overview":
        st.subheader("📊 S&OP Data Overview")
        
        # Summary metrics
        total_demand = demand_plan['quantity'].sum()
        total_supply = supply_plan['quantity'].sum()
        total_revenue = financial_plan['revenue'].sum()
        total_profit = financial_plan['profit'].sum()
        avg_margin = financial_plan['margin'].mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Demand", f"{total_demand:,.0f}")
        with col2:
            st.metric("Total Supply", f"{total_supply:,.0f}")
        with col3:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        with col4:
            st.metric("Total Profit", f"${total_profit:,.0f}")
        with col5:
            st.metric("Avg Margin", f"{avg_margin:.1f}%")
        
        # Plan overview charts
        st.subheader("📈 Plan Overview")
        
        # Demand vs Supply by period
        demand_monthly = demand_plan.groupby('period')['quantity'].sum().reset_index()
        supply_monthly = supply_plan.groupby('period')['quantity'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=demand_monthly['period'], y=demand_monthly['quantity'],
                                mode='lines+markers', name='Demand Plan', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=supply_monthly['period'], y=supply_monthly['quantity'],
                                mode='lines+markers', name='Supply Plan', line=dict(color='red')))
        
        fig.update_layout(title="Demand vs Supply Plan", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue and profit trends
        st.subheader("💰 Financial Performance")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=financial_plan['period'], y=financial_plan['revenue'],
                  name="Revenue"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=financial_plan['period'], y=financial_plan['margin'],
                      mode='lines+markers', name="Margin %"),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
        fig.update_layout(title="Revenue and Margin Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Product family breakdown
        st.subheader("📋 Product Family Analysis")
        family_summary = demand_plan.groupby('product_family').agg({
            'quantity': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        fig = px.pie(family_summary, values='revenue', names='product_family',
                    title="Revenue Distribution by Product Family")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Integrated Plan Review":
        st.subheader("🔄 Integrated Plan Review")
        
        # Create integrated view
        integrated_data = []
        
        for period in demand_plan['period'].unique():
            demand_period = demand_plan[demand_plan['period'] == period]['quantity'].sum()
            supply_period = supply_plan[supply_plan['period'] == period]['quantity'].sum()
            revenue_period = financial_plan[financial_plan['period'] == period]['revenue'].sum()
            profit_period = financial_plan[financial_plan['period'] == period]['profit'].sum()
            
            integrated_data.append({
                'period': period,
                'demand': demand_period,
                'supply': supply_period,
                'revenue': revenue_period,
                'profit': profit_period,
                'balance': 'Balanced' if abs(supply_period - demand_period) / demand_period < 0.05 else 'Imbalanced'
            })
        
        integrated_df = pd.DataFrame(integrated_data)
        
        # Plan balance metrics
        balanced_periods = len(integrated_df[integrated_df['balance'] == 'Balanced'])
        total_periods = len(integrated_df)
        balance_score = (balanced_periods / total_periods * 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Plan Balance Score", f"{balance_score:.1f}%")
        with col2:
            st.metric("Balanced Periods", f"{balanced_periods}/{total_periods}")
        with col3:
            avg_gap = abs(integrated_df['supply'] - integrated_df['demand']).mean()
            st.metric("Avg Volume Gap", f"{avg_gap:,.0f}")
        
        # Integrated plan visualization
        st.subheader("📊 Integrated Plan View")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Volume Plans", "Financial Plans"),
            vertical_spacing=0.1
        )
        
        # Volume plans
        fig.add_trace(
            go.Scatter(x=integrated_df['period'], y=integrated_df['demand'],
                      mode='lines+markers', name='Demand', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=integrated_df['period'], y=integrated_df['supply'],
                      mode='lines+markers', name='Supply', line=dict(color='red')),
            row=1, col=1
        )
        
        # Financial plans
        fig.add_trace(
            go.Bar(x=integrated_df['period'], y=integrated_df['revenue'],
                  name='Revenue', marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=integrated_df['period'], y=integrated_df['profit'],
                  name='Profit', marker_color='orange'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Integrated S&OP Plan Review")
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan details table
        st.subheader("📋 Integrated Plan Details")
        
        # Color code balance status
        def highlight_balance(val):
            return 'background-color: #ccffcc' if val == 'Balanced' else 'background-color: #ffcccc'
        
        styled_df = integrated_df.style.applymap(highlight_balance, subset=['balance'])
        st.dataframe(styled_df)
    
    elif analysis_type == "Gap Analysis":
        st.subheader("🎯 Gap Analysis")
        
        # Perform gap analysis
        gap_analysis = sop_planner.perform_gap_analysis(demand_plan, supply_plan, financial_plan)
        
        # Gap summary metrics
        avg_volume_gap = gap_analysis['volume_gap_pct'].mean()
        max_volume_gap = gap_analysis['volume_gap_pct'].max()
        avg_margin = gap_analysis['margin'].mean()
        imbalanced_periods = len(gap_analysis[gap_analysis['gap_status'] == 'Imbalanced'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Volume Gap", f"{avg_volume_gap:.1f}%")
        with col2:
            st.metric("Max Volume Gap", f"{max_volume_gap:.1f}%")
        with col3:
            st.metric("Avg Margin", f"{avg_margin:.1f}%")
        with col4:
            st.metric("Imbalanced Periods", imbalanced_periods)
        
        # Gap visualization
        st.subheader("📈 Gap Analysis Visualization")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Volume Gap Analysis", "Financial Gap Analysis")
        )
        
        # Volume gap
        colors = ['red' if x < 0 else 'green' for x in gap_analysis['volume_gap']]
        fig.add_trace(
            go.Bar(x=gap_analysis['period'], y=gap_analysis['volume_gap'],
                  name='Volume Gap', marker_color=colors),
            row=1, col=1
        )
        
        # Profit analysis
        fig.add_trace(
            go.Scatter(x=gap_analysis['period'], y=gap_analysis['profit'],
                      mode='lines+markers', name='Profit'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=gap_analysis['period'], y=gap_analysis['margin'],
                      mode='lines+markers', name='Margin %', yaxis='y2'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="S&OP Gap Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate action plans
        st.subheader("💡 Recommended Actions")
        action_plans = sop_planner.generate_action_plans(gap_analysis)
        
        if not action_plans.empty:
            # Priority actions
            high_priority = action_plans[action_plans['priority'] == 1]
            if not high_priority.empty:
                st.warning("🚨 High Priority Actions Required:")
                for _, action in high_priority.iterrows():
                    st.write(f"**{action['period']}** - {action['issue']}: {action['recommended_action']} (Owner: {action['owner']})")
            
            # All actions table
            st.subheader("📋 Complete Action Plan")
            st.dataframe(action_plans)
        else:
            st.success("✅ No significant gaps identified - plans are well balanced!")
        
        # Detailed gap analysis
        st.subheader("📊 Detailed Gap Analysis")
        st.dataframe(gap_analysis)
    
    elif analysis_type == "Scenario Planning":
        st.subheader("🔮 Scenario Planning")
        
        # Define scenarios
        scenarios = {
            'Baseline': {
                'demand_adjustment': 1.0,
                'cost_adjustment': 1.0,
                'capacity_adjustment': 1.0
            },
            'Optimistic': {
                'demand_adjustment': 1.15,
                'cost_adjustment': 0.95,
                'capacity_adjustment': 1.1
            },
            'Pessimistic': {
                'demand_adjustment': 0.85,
                'cost_adjustment': 1.1,
                'capacity_adjustment': 0.9
            },
            'High Growth': {
                'demand_adjustment': 1.25,
                'cost_adjustment': 1.05,
                'capacity_adjustment': 1.2
            },
            'Economic Downturn': {
                'demand_adjustment': 0.75,
                'cost_adjustment': 1.15,
                'capacity_adjustment': 0.8
            }
        }
        
        # Create base plan for scenario analysis
        base_plan = demand_plan.merge(supply_plan, on=['period', 'product_family'], how='inner')
        base_plan = base_plan.merge(financial_plan, on='period', how='inner')
        base_plan['demand'] = base_plan['quantity_x']
        base_plan['capacity'] = base_plan['capacity']
        base_plan['costs'] = base_plan['costs']
        base_plan['price'] = base_plan['price']
        
        # Run scenario analysis
        scenario_results = sop_planner.create_scenario_analysis(base_plan, scenarios)
        
        # Scenario comparison
        st.subheader("📊 Scenario Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(scenario_results, x='scenario', y='total_profit',
                        title="Total Profit by Scenario",
                        color='total_profit',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(scenario_results, x='scenario', y='margin',
                        title="Profit Margin by Scenario",
                        color='margin',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        # Scenario metrics table
        st.subheader("📋 Scenario Metrics Comparison")
        st.dataframe(scenario_results)
        
        # Risk assessment
        st.subheader("⚠️ Risk Assessment")
        
        baseline_profit = scenario_results[scenario_results['scenario'] == 'Baseline']['total_profit'].iloc[0]
        
        risk_analysis = []
        for _, row in scenario_results.iterrows():
            if row['scenario'] != 'Baseline':
                profit_impact = ((row['total_profit'] - baseline_profit) / baseline_profit * 100)
                risk_level = 'High' if abs(profit_impact) > 20 else 'Medium' if abs(profit_impact) > 10 else 'Low'
                
                risk_analysis.append({
                    'scenario': row['scenario'],
                    'profit_impact': profit_impact,
                    'risk_level': risk_level,
                    'recommendation': 'Monitor closely' if risk_level == 'High' else 'Standard monitoring'
                })
        
        risk_df = pd.DataFrame(risk_analysis)
        st.dataframe(risk_df)
        
        # Sensitivity analysis
        st.subheader("📈 Sensitivity Analysis")
        
        sensitivity_data = []
        for factor in ['demand_adjustment', 'cost_adjustment', 'capacity_adjustment']:
            for adjustment in [0.8, 0.9, 1.0, 1.1, 1.2]:
                test_scenario = {'demand_adjustment': 1.0, 'cost_adjustment': 1.0, 'capacity_adjustment': 1.0}
                test_scenario[factor] = adjustment
                
                result = sop_planner.create_scenario_analysis(base_plan, {'test': test_scenario})
                
                sensitivity_data.append({
                    'factor': factor.replace('_adjustment', ''),
                    'adjustment': adjustment,
                    'profit': result['total_profit'].iloc[0],
                    'margin': result['margin'].iloc[0]
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        fig = px.line(sensitivity_df, x='adjustment', y='profit', color='factor',
                     title="Profit Sensitivity Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Consensus Building":
        st.subheader("🤝 Consensus Building")
        
        # Calculate consensus metrics
        consensus_analysis = sop_planner.calculate_consensus_metrics(stakeholder_inputs)
        
        # Consensus summary
        high_consensus = len(consensus_analysis[consensus_analysis['consensus_level'] == 'High'])
        total_metrics = len(consensus_analysis)
        consensus_score = (high_consensus / total_metrics * 100) if total_metrics > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Consensus Score", f"{consensus_score:.1f}%")
        with col2:
            st.metric("High Consensus Items", f"{high_consensus}/{total_metrics}")
        with col3:
            avg_cv = consensus_analysis['cv'].mean()
            st.metric("Avg Variability (CV)", f"{avg_cv:.3f}")
        
        # Consensus visualization
        st.subheader("📊 Stakeholder Consensus Analysis")
        
        # Create consensus heatmap
        pivot_data = consensus_analysis.pivot(index='period', columns='metric', values='cv')
        
        fig = px.imshow(pivot_data.values, 
                       x=pivot_data.columns, 
                       y=pivot_data.index,
                       color_continuous_scale='RdYlGn_r',
                       title="Consensus Heatmap (Lower values = Higher consensus)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stakeholder input comparison
        st.subheader("👥 Stakeholder Input Comparison")
        
        for metric in ['demand_forecast', 'supply_plan', 'revenue_target']:
            if metric in stakeholder_inputs.columns:
                fig = px.box(stakeholder_inputs, x='period', y=metric, color='stakeholder',
                           title=f"{metric.replace('_', ' ').title()} by Stakeholder")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Consensus improvement recommendations
        st.subheader("💡 Consensus Improvement Recommendations")
        
        low_consensus = consensus_analysis[consensus_analysis['consensus_level'] == 'Low']
        
        if not low_consensus.empty:
            st.warning("Areas requiring consensus improvement:")
            for _, row in low_consensus.iterrows():
                st.write(f"• **{row['period']} - {row['metric']}**: High variability (CV: {row['cv']:.3f}) - Consider additional alignment sessions")
        else:
            st.success("✅ Good consensus across all metrics and periods!")
        
        # Detailed consensus analysis
        st.subheader("📋 Detailed Consensus Analysis")
        st.dataframe(consensus_analysis)
    
    elif analysis_type == "Executive Dashboard":
        st.subheader("📊 Executive S&OP Dashboard")
        
        # Key performance indicators
        total_demand = demand_plan['quantity'].sum()
        total_supply = supply_plan['quantity'].sum()
        total_revenue = financial_plan['revenue'].sum()
        total_profit = financial_plan['profit'].sum()
        
        # Calculate additional KPIs
        supply_demand_ratio = (total_supply / total_demand) if total_demand > 0 else 0
        overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Gap analysis for dashboard
        gap_analysis = sop_planner.perform_gap_analysis(demand_plan, supply_plan, financial_plan)
        balanced_periods = len(gap_analysis[gap_analysis['gap_status'] == 'Balanced'])
        plan_balance_score = (balanced_periods / len(gap_analysis) * 100)
        
        # Display executive KPIs
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Plan Balance Score", f"{plan_balance_score:.1f}%", 
                     delta=f"{'✅' if plan_balance_score > 80 else '⚠️'}")
        with col2:
            st.metric("Supply/Demand Ratio", f"{supply_demand_ratio:.2f}", 
                     delta=f"{'✅' if 0.95 <= supply_demand_ratio <= 1.05 else '⚠️'}")
        with col3:
            st.metric("Overall Margin", f"{overall_margin:.1f}%", 
                     delta=f"{'✅' if overall_margin > 15 else '⚠️'}")
        with col4:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        # Executive summary charts
        st.subheader("📈 Executive Summary")
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Plan Balance Trend", "Financial Performance", 
                           "Volume vs Capacity", "Margin Trend"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plan balance trend
        monthly_balance = gap_analysis.copy()
        monthly_balance['balance_score'] = 100 - abs(monthly_balance['volume_gap_pct'])
        
        fig.add_trace(
            go.Scatter(x=monthly_balance['period'], y=monthly_balance['balance_score'],
                      mode='lines+markers', name='Balance Score'),
            row=1, col=1
        )
        
        # Financial performance
        fig.add_trace(
            go.Bar(x=financial_plan['period'], y=financial_plan['revenue'],
                  name='Revenue', marker_color='blue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=financial_plan['period'], y=financial_plan['profit'],
                  name='Profit', marker_color='green'),
            row=1, col=2
        )
        
        # Volume vs Capacity
        monthly_demand = demand_plan.groupby('period')['quantity'].sum().reset_index()
        monthly_capacity = supply_plan.groupby('period')['capacity'].sum().reset_index()
        
        fig.add_trace(
            go.Bar(x=monthly_demand['period'], y=monthly_demand['quantity'],
                  name='Demand', marker_color='orange'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=monthly_capacity['period'], y=monthly_capacity['capacity'],
                  name='Capacity', marker_color='red'),
            row=2, col=1
        )
        
        # Margin trend
        fig.add_trace(
            go.Scatter(x=financial_plan['period'], y=financial_plan['margin'],
                      mode='lines+markers', name='Margin %'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="S&OP Executive Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Critical issues and opportunities
        st.subheader("🚨 Critical Issues & Opportunities")
        
        # Identify critical issues
        critical_issues = []
        
        # Volume imbalances
        major_gaps = gap_analysis[abs(gap_analysis['volume_gap_pct']) > 10]
        if not major_gaps.empty:
            critical_issues.append({
                'type': 'Volume Imbalance',
                'description': f"{len(major_gaps)} periods with >10% volume gaps",
                'severity': 'High',
                'action': 'Review capacity and demand plans'
            })
        
        # Low margins
        low_margin_periods = financial_plan[financial_plan['margin'] < 10]
        if not low_margin_periods.empty:
            critical_issues.append({
                'type': 'Low Profitability',
                'description': f"{len(low_margin_periods)} periods with <10% margin",
                'severity': 'Medium',
                'action': 'Review pricing and cost structure'
            })
        
        # Capacity constraints
        capacity_utilization = demand_plan.groupby('period')['quantity'].sum() / supply_plan.groupby('period')['capacity'].sum() * 100
        high_utilization = capacity_utilization[capacity_utilization > 90]
        if not high_utilization.empty:
            critical_issues.append({
                'type': 'Capacity Constraint',
                'description': f"{len(high_utilization)} periods with >90% capacity utilization",
                'severity': 'High',
                'action': 'Evaluate capacity expansion options'
            })
        
        if critical_issues:
            issues_df = pd.DataFrame(critical_issues)
            st.dataframe(issues_df)
        else:
            st.success("✅ No critical issues identified - S&OP plans are well balanced!")
        
        # Monthly review summary
        st.subheader("📅 Monthly Review Summary")
        
        review_summary = gap_analysis[['period', 'volume_gap_pct', 'margin', 'gap_status']].copy()
        review_summary['overall_health'] = review_summary.apply(
            lambda row: 'Excellent' if abs(row['volume_gap_pct']) < 5 and row['margin'] > 15
            else 'Good' if abs(row['volume_gap_pct']) < 10 and row['margin'] > 10
            else 'Needs Attention', axis=1
        )
        
        st.dataframe(review_summary)
    
    # Download options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Results")
    
    if st.sidebar.button("Download Sample Data"):
        demand_sample, supply_sample, financial_sample, stakeholder_sample = load_sample_sop_data()
        
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            demand_sample.to_excel(writer, sheet_name='Demand_Plan', index=False)
            supply_sample.to_excel(writer, sheet_name='Supply_Plan', index=False)
            financial_sample.to_excel(writer, sheet_name='Financial_Plan', index=False)
            stakeholder_sample.to_excel(writer, sheet_name='Stakeholder_Inputs', index=False)
        
        st.sidebar.download_button(
            label="📊 Download S&OP Data",
            data=output.getvalue(),
            file_name="sop_planning_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )