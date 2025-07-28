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

class SOEManager:
    """Advanced Sales and Operations Execution management class"""
    
    def __init__(self, data=None):
        self.data = data
    
    def calculate_execution_metrics(self, plan_data, actual_data):
        """Calculate execution performance metrics"""
        execution_metrics = []
        
        for period in plan_data['period'].unique():
            plan_period = plan_data[plan_data['period'] == period]
            actual_period = actual_data[actual_data['period'] == period]
            
            for metric in ['sales', 'production', 'inventory']:
                if metric in plan_period.columns and metric in actual_period.columns:
                    plan_value = plan_period[metric].sum()
                    actual_value = actual_period[metric].sum()
                    
                    variance = actual_value - plan_value
                    variance_pct = (variance / plan_value * 100) if plan_value != 0 else 0
                    
                    execution_metrics.append({
                        'period': period,
                        'metric': metric,
                        'plan': plan_value,
                        'actual': actual_value,
                        'variance': variance,
                        'variance_pct': variance_pct,
                        'execution_score': max(0, 100 - abs(variance_pct)),
                        'status': 'On Track' if abs(variance_pct) <= 5 else 'At Risk' if abs(variance_pct) <= 15 else 'Critical'
                    })
        
        return pd.DataFrame(execution_metrics)
    
    def monitor_kpi_performance(self, kpi_data, targets):
        """Monitor KPI performance against targets"""
        kpi_performance = []
        
        for period in kpi_data['period'].unique():
            period_data = kpi_data[kpi_data['period'] == period]
            
            for kpi in targets.keys():
                if kpi in period_data.columns:
                    actual_value = period_data[kpi].iloc[0] if not period_data.empty else 0
                    target_value = targets[kpi]
                    
                    achievement = (actual_value / target_value * 100) if target_value != 0 else 0
                    gap = actual_value - target_value
                    
                    status = 'Exceeding' if achievement > 105 else 'Meeting' if achievement >= 95 else 'Below Target'
                    
                    kpi_performance.append({
                        'period': period,
                        'kpi': kpi,
                        'target': target_value,
                        'actual': actual_value,
                        'achievement': achievement,
                        'gap': gap,
                        'status': status,
                        'trend': self._calculate_trend(kpi_data, kpi, period)
                    })
        
        return pd.DataFrame(kpi_performance)
    
    def _calculate_trend(self, data, kpi, current_period):
        """Calculate trend for KPI"""
        try:
            periods = sorted(data['period'].unique())
            current_idx = periods.index(current_period)
            
            if current_idx > 0:
                prev_period = periods[current_idx - 1]
                current_value = data[data['period'] == current_period][kpi].iloc[0]
                prev_value = data[data['period'] == prev_period][kpi].iloc[0]
                
                if current_value > prev_value:
                    return 'Improving'
                elif current_value < prev_value:
                    return 'Declining'
                else:
                    return 'Stable'
            return 'No Trend'
        except:
            return 'No Trend'
    
    def generate_corrective_actions(self, execution_metrics, kpi_performance):
        """Generate corrective actions based on performance gaps"""
        corrective_actions = []
        
        # Actions based on execution metrics
        critical_metrics = execution_metrics[execution_metrics['status'] == 'Critical']
        for _, row in critical_metrics.iterrows():
            action_type = 'Immediate Action Required'
            if row['variance_pct'] > 0:
                description = f"{row['metric'].title()} is {row['variance_pct']:.1f}% above plan"
                recommendation = f"Investigate reasons for excess {row['metric']} and adjust future plans"
            else:
                description = f"{row['metric'].title()} is {abs(row['variance_pct']):.1f}% below plan"
                recommendation = f"Implement recovery actions to address {row['metric']} shortfall"
            
            corrective_actions.append({
                'period': row['period'],
                'area': row['metric'].title(),
                'type': action_type,
                'description': description,
                'recommendation': recommendation,
                'priority': 'High',
                'owner': 'Operations Team'
            })
        
        # Actions based on KPI performance
        poor_kpis = kpi_performance[kpi_performance['status'] == 'Below Target']
        for _, row in poor_kpis.iterrows():
            corrective_actions.append({
                'period': row['period'],
                'area': row['kpi'].title(),
                'type': 'Performance Improvement',
                'description': f"{row['kpi']} achieving only {row['achievement']:.1f}% of target",
                'recommendation': f"Develop action plan to improve {row['kpi']} performance",
                'priority': 'Medium' if row['achievement'] > 80 else 'High',
                'owner': 'Department Head'
            })
        
        return pd.DataFrame(corrective_actions)
    
    def calculate_execution_dashboard_metrics(self, execution_data):
        """Calculate key metrics for execution dashboard"""
        overall_execution_score = execution_data['execution_score'].mean()
        
        on_track_count = len(execution_data[execution_data['status'] == 'On Track'])
        total_metrics = len(execution_data)
        on_track_percentage = (on_track_count / total_metrics * 100) if total_metrics > 0 else 0
        
        critical_issues = len(execution_data[execution_data['status'] == 'Critical'])
        
        return {
            'overall_execution_score': overall_execution_score,
            'on_track_percentage': on_track_percentage,
            'critical_issues': critical_issues,
            'total_metrics': total_metrics
        }

def load_sample_soe_data():
    """Load sample S&OE data"""
    np.random.seed(42)
    
    # Sample periods (last 12 weeks for execution tracking)
    periods = [(datetime.now() - timedelta(weeks=i)).strftime('%Y-W%U') for i in range(11, -1, -1)]
    
    # Sample plan data
    plan_data = []
    for period in periods:
        plan_data.append({
            'period': period,
            'sales': round(np.random.normal(10000, 1000), 0),
            'production': round(np.random.normal(9500, 800), 0),
            'inventory': round(np.random.normal(5000, 500), 0),
            'revenue': round(np.random.normal(500000, 50000), 0),
            'costs': round(np.random.normal(350000, 30000), 0)
        })
    
    # Sample actual data (with some variance from plan)
    actual_data = []
    for i, period in enumerate(periods):
        plan_row = plan_data[i]
        
        # Add realistic variance to actuals
        variance_factor = np.random.uniform(0.85, 1.15)
        
        actual_data.append({
            'period': period,
            'sales': round(plan_row['sales'] * variance_factor, 0),
            'production': round(plan_row['production'] * np.random.uniform(0.9, 1.1), 0),
            'inventory': round(plan_row['inventory'] * np.random.uniform(0.8, 1.2), 0),
            'revenue': round(plan_row['revenue'] * variance_factor, 0),
            'costs': round(plan_row['costs'] * np.random.uniform(0.95, 1.05), 0)
        })
    
    # Sample KPI data
    kpi_data = []
    for period in periods:
        kpi_data.append({
            'period': period,
            'service_level': round(np.random.uniform(92, 99), 1),
            'inventory_turns': round(np.random.uniform(8, 15), 1),
            'forecast_accuracy': round(np.random.uniform(75, 95), 1),
            'on_time_delivery': round(np.random.uniform(90, 98), 1),
            'capacity_utilization': round(np.random.uniform(75, 95), 1),
            'quality_score': round(np.random.uniform(95, 99.5), 1)
        })
    
    # Sample operational issues
    issues_data = []
    issue_types = ['Production Delay', 'Quality Issue', 'Supply Shortage', 'Equipment Failure', 'Demand Spike']
    
    for i, period in enumerate(periods[-4:]):  # Last 4 weeks
        num_issues = np.random.randint(0, 3)
        for j in range(num_issues):
            issues_data.append({
                'period': period,
                'issue_type': np.random.choice(issue_types),
                'description': f"Sample {np.random.choice(issue_types).lower()} in {period}",
                'severity': np.random.choice(['Low', 'Medium', 'High']),
                'status': np.random.choice(['Open', 'In Progress', 'Resolved']),
                'owner': np.random.choice(['Production Team', 'Quality Team', 'Supply Team']),
                'created_date': period,
                'target_resolution': period
            })
    
    return (pd.DataFrame(plan_data), pd.DataFrame(actual_data), 
            pd.DataFrame(kpi_data), pd.DataFrame(issues_data))

def run():
    st.title("⚡ Sales & Operations Execution (S&OE)")
    st.markdown("""
    Real-time execution monitoring, performance tracking, and corrective action management 
    for sales and operations execution.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("S&OE Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Real-time Dashboard", "Execution Performance", "KPI Monitoring", 
         "Issue Tracking", "Corrective Actions", "Weekly Review"]
    )
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        plan_data, actual_data, kpi_data, issues_data = load_sample_soe_data()
        st.success("✅ Sample S&OE data loaded successfully!")
    else:
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your S&OE data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain plan vs actual execution data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    soe_data = pd.read_csv(uploaded_file)
                else:
                    soe_data = pd.read_excel(uploaded_file)
                
                # For uploaded data, create sample datasets
                plan_data, actual_data, kpi_data, issues_data = load_sample_soe_data()
                st.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("Please upload a file or use sample data to proceed.")
            return
    
    # Initialize S&OE manager
    soe_manager = SOEManager()
    
    # Calculate execution metrics
    execution_metrics = soe_manager.calculate_execution_metrics(plan_data, actual_data)
    
    # Define KPI targets
    kpi_targets = {
        'service_level': 95.0,
        'inventory_turns': 12.0,
        'forecast_accuracy': 85.0,
        'on_time_delivery': 95.0,
        'capacity_utilization': 85.0,
        'quality_score': 98.0
    }
    
    # Calculate KPI performance
    kpi_performance = soe_manager.monitor_kpi_performance(kpi_data, kpi_targets)
    
    if analysis_type == "Real-time Dashboard":
        st.subheader("📊 Real-time Execution Dashboard")
        
        # Dashboard metrics
        dashboard_metrics = soe_manager.calculate_execution_dashboard_metrics(execution_metrics)
        
        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Execution Score", f"{dashboard_metrics['overall_execution_score']:.1f}%",
                     delta=f"{'🟢' if dashboard_metrics['overall_execution_score'] > 90 else '🟡' if dashboard_metrics['overall_execution_score'] > 80 else '🔴'}")
        with col2:
            st.metric("On Track %", f"{dashboard_metrics['on_track_percentage']:.1f}%",
                     delta=f"{'🟢' if dashboard_metrics['on_track_percentage'] > 80 else '🟡' if dashboard_metrics['on_track_percentage'] > 60 else '🔴'}")
        with col3:
            st.metric("Critical Issues", dashboard_metrics['critical_issues'],
                     delta=f"{'🔴' if dashboard_metrics['critical_issues'] > 0 else '🟢'}")
        with col4:
            open_issues = len(issues_data[issues_data['status'] == 'Open'])
            st.metric("Open Issues", open_issues,
                     delta=f"{'🔴' if open_issues > 2 else '🟡' if open_issues > 0 else '🟢'}")
        
        # Real-time execution charts
        st.subheader("📈 Real-time Performance")
        
        # Create dashboard with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Plan vs Actual Sales", "Execution Score Trend", 
                           "KPI Performance", "Issue Status"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Plan vs Actual Sales
        sales_data = execution_metrics[execution_metrics['metric'] == 'sales']
        fig.add_trace(
            go.Scatter(x=sales_data['period'], y=sales_data['plan'],
                      mode='lines+markers', name='Plan', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=sales_data['period'], y=sales_data['actual'],
                      mode='lines+markers', name='Actual', line=dict(color='red')),
            row=1, col=1
        )
        
        # Execution Score Trend
        avg_scores = execution_metrics.groupby('period')['execution_score'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=avg_scores['period'], y=avg_scores['execution_score'],
                      mode='lines+markers', name='Execution Score'),
            row=1, col=2
        )
        
        # KPI Performance
        latest_kpis = kpi_performance[kpi_performance['period'] == kpi_performance['period'].max()]
        fig.add_trace(
            go.Bar(x=latest_kpis['kpi'], y=latest_kpis['achievement'],
                  name='KPI Achievement %'),
            row=2, col=1
        )
        
        # Issue Status
        issue_status = issues_data['status'].value_counts()
        fig.add_trace(
            go.Pie(labels=issue_status.index, values=issue_status.values,
                  name="Issue Status"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="S&OE Real-time Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Current week status
        st.subheader("📅 Current Week Status")
        
        current_period = execution_metrics['period'].max()
        current_execution = execution_metrics[execution_metrics['period'] == current_period]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Execution Performance:**")
            for _, row in current_execution.iterrows():
                status_icon = "🟢" if row['status'] == 'On Track' else "🟡" if row['status'] == 'At Risk' else "🔴"
                st.write(f"{status_icon} {row['metric'].title()}: {row['variance_pct']:+.1f}% vs plan")
        
        with col2:
            current_kpis = kpi_performance[kpi_performance['period'] == current_period]
            st.write("**KPI Status:**")
            for _, row in current_kpis.iterrows():
                status_icon = "🟢" if row['status'] == 'Meeting' or row['status'] == 'Exceeding' else "🔴"
                st.write(f"{status_icon} {row['kpi'].title()}: {row['achievement']:.1f}% of target")
    
    elif analysis_type == "Execution Performance":
        st.subheader("📊 Execution Performance Analysis")
        
        # Performance summary
        avg_execution_score = execution_metrics['execution_score'].mean()
        on_track_metrics = len(execution_metrics[execution_metrics['status'] == 'On Track'])
        total_metrics = len(execution_metrics)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Execution Score", f"{avg_execution_score:.1f}%")
        with col2:
            st.metric("On Track Metrics", f"{on_track_metrics}/{total_metrics}")
        with col3:
            critical_metrics = len(execution_metrics[execution_metrics['status'] == 'Critical'])
            st.metric("Critical Metrics", critical_metrics)
        
        # Execution performance by metric
        st.subheader("📈 Performance by Metric")
        
        metric_performance = execution_metrics.groupby('metric').agg({
            'execution_score': 'mean',
            'variance_pct': 'mean'
        }).reset_index()
        
        fig = px.bar(metric_performance, x='metric', y='execution_score',
                    title="Average Execution Score by Metric",
                    color='execution_score',
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan vs Actual trends
        st.subheader("📊 Plan vs Actual Trends")
        
        for metric in execution_metrics['metric'].unique():
            metric_data = execution_metrics[execution_metrics['metric'] == metric]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metric_data['period'], 
                y=metric_data['plan'],
                mode='lines+markers',
                name='Plan',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=metric_data['period'], 
                y=metric_data['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='red')
            ))
            
            fig.update_layout(title=f"{metric.title()} - Plan vs Actual", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Variance analysis
        st.subheader("📋 Variance Analysis")
        
        # Color code variances
        def highlight_variance(val):
            if abs(val) <= 5:
                return 'background-color: #ccffcc'  # Green for good performance
            elif abs(val) <= 15:
                return 'background-color: #ffffcc'  # Yellow for at risk
            else:
                return 'background-color: #ffcccc'  # Red for critical
        
        styled_execution = execution_metrics.style.applymap(highlight_variance, subset=['variance_pct'])
        st.dataframe(styled_execution)
    
    elif analysis_type == "KPI Monitoring":
        st.subheader("📊 KPI Performance Monitoring")
        
        # KPI summary
        meeting_kpis = len(kpi_performance[kpi_performance['status'].isin(['Meeting', 'Exceeding'])])
        total_kpis = len(kpi_performance)
        avg_achievement = kpi_performance['achievement'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KPIs on Target", f"{meeting_kpis}/{total_kpis}")
        with col2:
            st.metric("Average Achievement", f"{avg_achievement:.1f}%")
        with col3:
            exceeding_kpis = len(kpi_performance[kpi_performance['status'] == 'Exceeding'])
            st.metric("Exceeding Target", exceeding_kpis)
        
        # KPI performance heatmap
        st.subheader("🔥 KPI Performance Heatmap")
        
        kpi_pivot = kpi_performance.pivot(index='period', columns='kpi', values='achievement')
        
        fig = px.imshow(kpi_pivot.values,
                       x=kpi_pivot.columns,
                       y=kpi_pivot.index,
                       color_continuous_scale='RdYlGn',
                       aspect='auto',
                       title="KPI Achievement % Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual KPI trends
        st.subheader("📈 KPI Trends")
        
        selected_kpis = st.multiselect(
            "Select KPIs to display",
            options=kpi_performance['kpi'].unique(),
            default=kpi_performance['kpi'].unique()[:3]
        )
        
        if selected_kpis:
            fig = go.Figure()
            
            for kpi in selected_kpis:
                kpi_trend = kpi_performance[kpi_performance['kpi'] == kpi]
                fig.add_trace(go.Scatter(
                    x=kpi_trend['period'],
                    y=kpi_trend['achievement'],
                    mode='lines+markers',
                    name=kpi.title()
                ))
            
            # Add target line
            fig.add_hline(y=100, line_dash="dash", line_color="black", 
                         annotation_text="Target (100%)")
            
            fig.update_layout(title="KPI Achievement Trends", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # KPI gap analysis
        st.subheader("📊 KPI Gap Analysis")
        
        latest_period = kpi_performance['period'].max()
        latest_kpis = kpi_performance[kpi_performance['period'] == latest_period]
        
        fig = px.bar(latest_kpis, x='kpi', y='gap',
                    title="Current KPI Gaps vs Target",
                    color='gap',
                    color_continuous_scale='RdYlGn')
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed KPI performance
        st.subheader("📋 Detailed KPI Performance")
        st.dataframe(kpi_performance)
    
    elif analysis_type == "Issue Tracking":
        st.subheader("🎯 Operational Issue Tracking")
        
        if not issues_data.empty:
            # Issue summary
            open_issues = len(issues_data[issues_data['status'] == 'Open'])
            high_severity = len(issues_data[issues_data['severity'] == 'High'])
            resolved_issues = len(issues_data[issues_data['status'] == 'Resolved'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Issues", open_issues, 
                         delta=f"{'🔴' if open_issues > 2 else '🟡' if open_issues > 0 else '🟢'}")
            with col2:
                st.metric("High Severity", high_severity,
                         delta=f"{'🔴' if high_severity > 0 else '🟢'}")
            with col3:
                resolution_rate = (resolved_issues / len(issues_data) * 100) if len(issues_data) > 0 else 0
                st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
            
            # Issue distribution
            st.subheader("📊 Issue Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                # Issues by type
                issue_types = issues_data['issue_type'].value_counts()
                fig = px.pie(values=issue_types.values, names=issue_types.index,
                           title="Issues by Type")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Issues by severity
                severity_counts = issues_data['severity'].value_counts()
                fig = px.bar(x=severity_counts.index, y=severity_counts.values,
                           title="Issues by Severity",
                           color=severity_counts.index,
                           color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Issue timeline
            st.subheader("📅 Issue Timeline")
            
            issue_timeline = issues_data.groupby(['period', 'status']).size().reset_index(name='count')
            fig = px.bar(issue_timeline, x='period', y='count', color='status',
                        title="Issues by Period and Status")
            st.plotly_chart(fig, use_container_width=True)
            
            # Issue details table
            st.subheader("📋 Issue Details")
            
            # Filter options
            status_filter = st.selectbox("Filter by Status", 
                                       options=['All'] + list(issues_data['status'].unique()))
            
            if status_filter != 'All':
                filtered_issues = issues_data[issues_data['status'] == status_filter]
            else:
                filtered_issues = issues_data
            
            # Color code by severity
            def highlight_severity(row):
                if row['severity'] == 'High':
                    return ['background-color: #ffcccc'] * len(row)
                elif row['severity'] == 'Medium':
                    return ['background-color: #ffffcc'] * len(row)
                else:
                    return ['background-color: #ccffcc'] * len(row)
            
            styled_issues = filtered_issues.style.apply(highlight_severity, axis=1)
            st.dataframe(styled_issues)
        else:
            st.info("No operational issues to display")
    
    elif analysis_type == "Corrective Actions":
        st.subheader("🔧 Corrective Action Management")
        
        # Generate corrective actions
        corrective_actions = soe_manager.generate_corrective_actions(execution_metrics, kpi_performance)
        
        if not corrective_actions.empty:
            # Action summary
            high_priority = len(corrective_actions[corrective_actions['priority'] == 'High'])
            total_actions = len(corrective_actions)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Actions Required", total_actions)
            with col2:
                st.metric("High Priority Actions", high_priority,
                         delta=f"{'🔴' if high_priority > 0 else '🟢'}")
            
            # Priority actions
            if high_priority > 0:
                st.subheader("🚨 High Priority Actions")
                high_priority_actions = corrective_actions[corrective_actions['priority'] == 'High']
                
                for _, action in high_priority_actions.iterrows():
                    st.error(f"**{action['area']}** ({action['period']}): {action['description']}")
                    st.write(f"📋 Recommendation: {action['recommendation']}")
                    st.write(f"👤 Owner: {action['owner']}")
                    st.write("---")
            
            # Action distribution
            st.subheader("📊 Action Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                # Actions by area
                area_counts = corrective_actions['area'].value_counts()
                fig = px.bar(x=area_counts.index, y=area_counts.values,
                           title="Actions by Area")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Actions by priority
                priority_counts = corrective_actions['priority'].value_counts()
                fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                           title="Actions by Priority",
                           color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Action plan table
            st.subheader("📋 Complete Action Plan")
            
            # Color code by priority
            def highlight_priority(row):
                if row['priority'] == 'High':
                    return ['background-color: #ffcccc'] * len(row)
                elif row['priority'] == 'Medium':
                    return ['background-color: #ffffcc'] * len(row)
                else:
                    return ['background-color: #ccffcc'] * len(row)
            
            styled_actions = corrective_actions.style.apply(highlight_priority, axis=1)
            st.dataframe(styled_actions)
        else:
            st.success("✅ No corrective actions required - execution is on track!")
    
    elif analysis_type == "Weekly Review":
        st.subheader("📅 Weekly Execution Review")
        
        # Select week for review
        available_periods = sorted(execution_metrics['period'].unique(), reverse=True)
        selected_period = st.selectbox("Select Week for Review", available_periods)
        
        # Week summary
        week_execution = execution_metrics[execution_metrics['period'] == selected_period]
        week_kpis = kpi_performance[kpi_performance['period'] == selected_period]
        week_issues = issues_data[issues_data['period'] == selected_period] if not issues_data.empty else pd.DataFrame()
        
        # Week metrics
        week_score = week_execution['execution_score'].mean()
        on_track = len(week_execution[week_execution['status'] == 'On Track'])
        total = len(week_execution)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Week Execution Score", f"{week_score:.1f}%")
        with col2:
            st.metric("Metrics On Track", f"{on_track}/{total}")
        with col3:
            week_issues_count = len(week_issues) if not week_issues.empty else 0
            st.metric("Issues This Week", week_issues_count)
        
        # Week performance details
        st.subheader("📊 Week Performance Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Execution Performance:**")
            for _, row in week_execution.iterrows():
                status_icon = "🟢" if row['status'] == 'On Track' else "🟡" if row['status'] == 'At Risk' else "🔴"
                st.write(f"{status_icon} **{row['metric'].title()}**: {row['variance_pct']:+.1f}% vs plan")
                st.write(f"   Plan: {row['plan']:,.0f} | Actual: {row['actual']:,.0f}")
        
        with col2:
            if not week_kpis.empty:
                st.write("**KPI Performance:**")
                for _, row in week_kpis.iterrows():
                    status_icon = "🟢" if row['status'] in ['Meeting', 'Exceeding'] else "🔴"
                    st.write(f"{status_icon} **{row['kpi'].title()}**: {row['achievement']:.1f}% of target")
                    st.write(f"   Target: {row['target']} | Actual: {row['actual']}")
        
        # Week issues
        if not week_issues.empty:
            st.subheader("⚠️ Issues This Week")
            for _, issue in week_issues.iterrows():
                severity_icon = "🔴" if issue['severity'] == 'High' else "🟡" if issue['severity'] == 'Medium' else "🟢"
                st.write(f"{severity_icon} **{issue['issue_type']}** ({issue['severity']} severity)")
                st.write(f"   {issue['description']}")
                st.write(f"   Status: {issue['status']} | Owner: {issue['owner']}")
        
        # Week comparison
        st.subheader("📈 Week-over-Week Comparison")
        
        if len(available_periods) > 1:
            prev_period = available_periods[available_periods.index(selected_period) + 1] if available_periods.index(selected_period) < len(available_periods) - 1 else None
            
            if prev_period:
                prev_execution = execution_metrics[execution_metrics['period'] == prev_period]
                prev_score = prev_execution['execution_score'].mean()
                
                score_change = week_score - prev_score
                st.metric("Score Change vs Previous Week", f"{score_change:+.1f}%",
                         delta=f"{'🟢' if score_change > 0 else '🔴' if score_change < 0 else '➡️'}")
                
                # Detailed comparison
                comparison_data = []
                for metric in week_execution['metric'].unique():
                    current = week_execution[week_execution['metric'] == metric]['execution_score'].iloc[0]
                    previous = prev_execution[prev_execution['metric'] == metric]['execution_score'].iloc[0]
                    change = current - previous
                    
                    comparison_data.append({
                        'metric': metric.title(),
                        'current_score': current,
                        'previous_score': previous,
                        'change': change
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = px.bar(comparison_df, x='metric', y='change',
                           title=f"Score Change: {selected_period} vs {prev_period}",
                           color='change',
                           color_continuous_scale='RdYlGn')
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Results")
    
    if st.sidebar.button("Download Sample Data"):
        plan_sample, actual_sample, kpi_sample, issues_sample = load_sample_soe_data()
        
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            plan_sample.to_excel(writer, sheet_name='Plan_Data', index=False)
            actual_sample.to_excel(writer, sheet_name='Actual_Data', index=False)
            kpi_sample.to_excel(writer, sheet_name='KPI_Data', index=False)
            issues_sample.to_excel(writer, sheet_name='Issues_Data', index=False)
        
        st.sidebar.download_button(
            label="📊 Download S&OE Data",
            data=output.getvalue(),
            file_name="soe_execution_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )