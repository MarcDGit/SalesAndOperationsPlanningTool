import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import io

class SupplyPlanner:
    """Advanced supply planning and optimization class"""
    
    def __init__(self, data=None):
        self.data = data
        
    def calculate_capacity_requirements(self, demand_data, lead_times, capacity_constraints):
        """Calculate capacity requirements based on demand and constraints"""
        # Calculate required capacity per period
        capacity_req = demand_data.copy()
        
        for sku in capacity_req['sku'].unique():
            sku_mask = capacity_req['sku'] == sku
            lead_time = lead_times.get(sku, 1)
            
            # Shift demand forward by lead time
            capacity_req.loc[sku_mask, 'required_capacity'] = capacity_req.loc[sku_mask, 'demand'].shift(-lead_time)
        
        # Group by period and calculate total capacity needed
        capacity_summary = capacity_req.groupby('week').agg({
            'required_capacity': 'sum',
            'demand': 'sum'
        }).reset_index()
        
        return capacity_summary
    
    def optimize_procurement_plan(self, demand_forecast, supplier_data, constraints):
        """Optimize procurement plan considering supplier constraints and costs"""
        procurement_plan = []
        
        for period in demand_forecast['week'].unique():
            period_demand = demand_forecast[demand_forecast['week'] == period]
            
            for _, row in period_demand.iterrows():
                sku = row['sku']
                required_qty = row['demand']
                
                # Find best supplier based on cost and capacity
                available_suppliers = supplier_data[supplier_data['sku'] == sku]
                
                if not available_suppliers.empty:
                    # Sort by cost and select best option
                    best_supplier = available_suppliers.sort_values('unit_cost').iloc[0]
                    
                    procurement_plan.append({
                        'week': period,
                        'sku': sku,
                        'supplier': best_supplier['supplier'],
                        'quantity': required_qty,
                        'unit_cost': best_supplier['unit_cost'],
                        'total_cost': required_qty * best_supplier['unit_cost'],
                        'lead_time': best_supplier['lead_time']
                    })
        
        return pd.DataFrame(procurement_plan)
    
    def calculate_master_production_schedule(self, demand_plan, production_capacity, inventory_levels):
        """Calculate master production schedule (MPS)"""
        mps = []
        
        for period in demand_plan['week'].unique():
            period_demand = demand_plan[demand_plan['week'] == period]
            
            for _, row in period_demand.iterrows():
                sku = row['sku']
                demand = row['demand']
                current_inventory = inventory_levels.get(sku, 0)
                
                # Calculate net requirements
                net_requirement = max(0, demand - current_inventory)
                
                # Check production capacity
                available_capacity = production_capacity.get(sku, demand)
                production_qty = min(net_requirement, available_capacity)
                
                mps.append({
                    'week': period,
                    'sku': sku,
                    'gross_requirement': demand,
                    'current_inventory': current_inventory,
                    'net_requirement': net_requirement,
                    'planned_production': production_qty,
                    'projected_inventory': current_inventory + production_qty - demand
                })
                
                # Update inventory for next period
                inventory_levels[sku] = current_inventory + production_qty - demand
        
        return pd.DataFrame(mps)

def load_sample_supply_data():
    """Load sample supply planning data"""
    np.random.seed(42)
    
    # Sample demand forecast
    weeks = pd.date_range('2024-01-01', periods=26, freq='W')
    skus = ['SKU_001', 'SKU_002', 'SKU_003', 'SKU_004', 'SKU_005']
    locations = ['Plant_A', 'Plant_B', 'Plant_C']
    
    demand_data = []
    for week in weeks:
        for sku in skus:
            for location in locations:
                base_demand = np.random.normal(100, 20)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week.week / 52)
                demand = max(0, base_demand * seasonal_factor)
                
                demand_data.append({
                    'week': week,
                    'sku': sku,
                    'location': location,
                    'demand': round(demand, 1),
                    'forecast_accuracy': np.random.uniform(0.85, 0.98)
                })
    
    # Sample supplier data
    supplier_data = []
    for sku in skus:
        num_suppliers = np.random.randint(2, 4)
        for i in range(num_suppliers):
            supplier_data.append({
                'sku': sku,
                'supplier': f'Supplier_{chr(65+i)}',
                'unit_cost': round(np.random.uniform(10, 50), 2),
                'lead_time': np.random.randint(1, 4),
                'min_order_qty': np.random.randint(50, 200),
                'max_capacity': np.random.randint(500, 2000),
                'quality_score': np.random.uniform(0.8, 1.0)
            })
    
    # Sample capacity data
    capacity_data = []
    for location in locations:
        for sku in skus:
            capacity_data.append({
                'location': location,
                'sku': sku,
                'max_capacity': np.random.randint(300, 800),
                'setup_time': np.random.uniform(1, 4),
                'efficiency': np.random.uniform(0.85, 0.95)
            })
    
    return pd.DataFrame(demand_data), pd.DataFrame(supplier_data), pd.DataFrame(capacity_data)

def run():
    st.title("🚚 Supply Planning Module")
    st.markdown("""
    Comprehensive supply planning with capacity optimization, procurement planning, 
    and master production scheduling.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Supply Planning Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Upload & Overview", "Capacity Planning", "Procurement Optimization", 
         "Master Production Schedule", "Supplier Analysis", "Supply Chain Metrics"]
    )
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        demand_data, supplier_data, capacity_data = load_sample_supply_data()
        st.success("✅ Sample supply planning data loaded successfully!")
    else:
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your supply planning data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain demand forecast, supplier, and capacity data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    demand_data = pd.read_csv(uploaded_file)
                else:
                    demand_data = pd.read_excel(uploaded_file)
                
                # For uploaded data, create sample supplier and capacity data
                supplier_data, capacity_data = load_sample_supply_data()[1:3]
                st.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("Please upload a file or use sample data to proceed.")
            return
    
    # Initialize supply planner
    supply_planner = SupplyPlanner(demand_data)
    
    if analysis_type == "Data Upload & Overview":
        st.subheader("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total SKUs", demand_data['sku'].nunique())
        with col2:
            st.metric("Total Locations", demand_data['location'].nunique())
        with col3:
            st.metric("Planning Horizon (Weeks)", demand_data['week'].nunique())
        
        # Demand trend visualization
        st.subheader("📈 Demand Trends")
        weekly_demand = demand_data.groupby('week')['demand'].sum().reset_index()
        
        fig = px.line(weekly_demand, x='week', y='demand', 
                     title="Total Demand Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # SKU analysis
        st.subheader("📋 SKU Analysis")
        sku_summary = demand_data.groupby('sku').agg({
            'demand': ['sum', 'mean', 'std']
        }).round(2)
        sku_summary.columns = ['Total Demand', 'Avg Weekly Demand', 'Demand Std Dev']
        st.dataframe(sku_summary)
    
    elif analysis_type == "Capacity Planning":
        st.subheader("🏭 Capacity Planning Analysis")
        
        # Calculate capacity requirements
        lead_times = {sku: np.random.randint(1, 4) for sku in demand_data['sku'].unique()}
        capacity_constraints = {loc: 1000 for loc in demand_data['location'].unique()}
        
        # Group demand by week for capacity planning
        weekly_demand = demand_data.groupby(['week', 'sku'])['demand'].sum().reset_index()
        capacity_req = supply_planner.calculate_capacity_requirements(
            weekly_demand, lead_times, capacity_constraints
        )
        
        # Visualize capacity requirements vs available capacity
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=capacity_req['week'], 
            y=capacity_req['required_capacity'],
            mode='lines+markers',
            name='Required Capacity',
            line=dict(color='red')
        ))
        
        # Add capacity limit line
        max_capacity = 800
        fig.add_hline(y=max_capacity, line_dash="dash", line_color="green",
                     annotation_text="Max Capacity")
        
        fig.update_layout(
            title="Capacity Requirements vs Available Capacity",
            xaxis_title="Week",
            yaxis_title="Capacity Units",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Capacity utilization metrics
        capacity_req['capacity_utilization'] = (capacity_req['required_capacity'] / max_capacity * 100).round(2)
        capacity_req['capacity_gap'] = (capacity_req['required_capacity'] - max_capacity).clip(lower=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_utilization = capacity_req['capacity_utilization'].mean()
            st.metric("Avg Capacity Utilization", f"{avg_utilization:.1f}%")
        with col2:
            max_utilization = capacity_req['capacity_utilization'].max()
            st.metric("Peak Capacity Utilization", f"{max_utilization:.1f}%")
        with col3:
            capacity_shortfall = capacity_req['capacity_gap'].sum()
            st.metric("Total Capacity Shortfall", f"{capacity_shortfall:.0f}")
        
        # Detailed capacity table
        st.subheader("📋 Detailed Capacity Analysis")
        st.dataframe(capacity_req)
    
    elif analysis_type == "Procurement Optimization":
        st.subheader("🛒 Procurement Optimization")
        
        # Optimize procurement plan
        constraints = {'budget': 100000, 'quality_threshold': 0.8}
        procurement_plan = supply_planner.optimize_procurement_plan(
            demand_data, supplier_data, constraints
        )
        
        # Procurement summary
        total_cost = procurement_plan['total_cost'].sum()
        total_qty = procurement_plan['quantity'].sum()
        avg_cost = procurement_plan['unit_cost'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Procurement Cost", f"${total_cost:,.0f}")
        with col2:
            st.metric("Total Quantity", f"{total_qty:,.0f}")
        with col3:
            st.metric("Average Unit Cost", f"${avg_cost:.2f}")
        
        # Cost by supplier visualization
        supplier_costs = procurement_plan.groupby('supplier')['total_cost'].sum().reset_index()
        fig = px.pie(supplier_costs, values='total_cost', names='supplier',
                    title="Procurement Cost Distribution by Supplier")
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly procurement schedule
        st.subheader("📅 Weekly Procurement Schedule")
        weekly_procurement = procurement_plan.groupby('week').agg({
            'total_cost': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=weekly_procurement['week'], y=weekly_procurement['total_cost'],
                  name="Procurement Cost"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=weekly_procurement['week'], y=weekly_procurement['quantity'],
                      mode='lines+markers', name="Quantity"),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Cost ($)", secondary_y=False)
        fig.update_yaxes(title_text="Quantity", secondary_y=True)
        fig.update_layout(title="Weekly Procurement Plan", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed procurement plan
        st.subheader("📋 Detailed Procurement Plan")
        st.dataframe(procurement_plan)
    
    elif analysis_type == "Master Production Schedule":
        st.subheader("🏭 Master Production Schedule (MPS)")
        
        # Calculate MPS
        production_capacity = {sku: 500 for sku in demand_data['sku'].unique()}
        inventory_levels = {sku: np.random.randint(50, 200) for sku in demand_data['sku'].unique()}
        
        weekly_demand = demand_data.groupby(['week', 'sku'])['demand'].sum().reset_index()
        mps = supply_planner.calculate_master_production_schedule(
            weekly_demand, production_capacity, inventory_levels
        )
        
        # MPS summary metrics
        total_production = mps['planned_production'].sum()
        total_requirements = mps['gross_requirement'].sum()
        production_efficiency = (total_production / total_requirements * 100) if total_requirements > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Planned Production", f"{total_production:,.0f}")
        with col2:
            st.metric("Total Requirements", f"{total_requirements:,.0f}")
        with col3:
            st.metric("Production Efficiency", f"{production_efficiency:.1f}%")
        
        # Production schedule visualization
        production_by_week = mps.groupby('week')['planned_production'].sum().reset_index()
        fig = px.bar(production_by_week, x='week', y='planned_production',
                    title="Weekly Production Schedule")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Inventory projection
        st.subheader("📦 Inventory Projections")
        inventory_projection = mps.groupby('week')['projected_inventory'].sum().reset_index()
        fig = px.line(inventory_projection, x='week', y='projected_inventory',
                     title="Projected Inventory Levels")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed MPS
        st.subheader("📋 Detailed Master Production Schedule")
        st.dataframe(mps)
    
    elif analysis_type == "Supplier Analysis":
        st.subheader("🏢 Supplier Analysis")
        
        # Supplier performance metrics
        supplier_summary = supplier_data.groupby('supplier').agg({
            'unit_cost': 'mean',
            'lead_time': 'mean',
            'max_capacity': 'sum',
            'quality_score': 'mean'
        }).round(2)
        
        # Calculate supplier score (weighted)
        supplier_summary['supplier_score'] = (
            (1 / supplier_summary['unit_cost']) * 0.3 +  # Lower cost is better
            (1 / supplier_summary['lead_time']) * 0.2 +   # Shorter lead time is better
            (supplier_summary['max_capacity'] / supplier_summary['max_capacity'].max()) * 0.3 +  # Higher capacity is better
            supplier_summary['quality_score'] * 0.2       # Higher quality is better
        ).round(3)
        
        st.dataframe(supplier_summary)
        
        # Supplier comparison radar chart
        st.subheader("📊 Supplier Performance Comparison")
        
        # Normalize metrics for radar chart
        metrics = ['unit_cost', 'lead_time', 'max_capacity', 'quality_score']
        normalized_data = supplier_data.copy()
        
        # Inverse normalize cost and lead time (lower is better)
        normalized_data['cost_score'] = 1 / normalized_data['unit_cost']
        normalized_data['speed_score'] = 1 / normalized_data['lead_time']
        
        # Create radar chart for top 3 suppliers
        top_suppliers = supplier_summary.nlargest(3, 'supplier_score').index[:3]
        
        fig = go.Figure()
        
        for supplier in top_suppliers:
            supplier_subset = normalized_data[normalized_data['supplier'] == supplier]
            if not supplier_subset.empty:
                values = [
                    supplier_subset['cost_score'].mean(),
                    supplier_subset['speed_score'].mean(),
                    supplier_subset['max_capacity'].mean() / normalized_data['max_capacity'].max(),
                    supplier_subset['quality_score'].mean()
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Cost Efficiency', 'Speed', 'Capacity', 'Quality'],
                    fill='toself',
                    name=supplier
                ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Top 3 Suppliers Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Supply Chain Metrics":
        st.subheader("📈 Supply Chain Performance Metrics")
        
        # Calculate key supply chain metrics
        weekly_demand = demand_data.groupby('week')['demand'].sum().reset_index()
        
        # Demand variability
        demand_cv = weekly_demand['demand'].std() / weekly_demand['demand'].mean()
        
        # Average lead time
        avg_lead_time = supplier_data['lead_time'].mean()
        
        # Supplier concentration (HHI)
        supplier_volume = supplier_data.groupby('supplier')['max_capacity'].sum()
        supplier_share = supplier_volume / supplier_volume.sum()
        hhi = (supplier_share ** 2).sum()
        
        # Service level (simulated)
        service_level = np.random.uniform(0.92, 0.98)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Demand Variability (CV)", f"{demand_cv:.3f}")
            st.metric("Average Lead Time", f"{avg_lead_time:.1f} weeks")
        with col2:
            st.metric("Supplier Concentration (HHI)", f"{hhi:.3f}")
            st.metric("Service Level", f"{service_level:.1%}")
        
        # Supply chain risk assessment
        st.subheader("⚠️ Supply Chain Risk Assessment")
        
        risk_factors = {
            'Demand Volatility': 'High' if demand_cv > 0.3 else 'Medium' if demand_cv > 0.15 else 'Low',
            'Lead Time Risk': 'High' if avg_lead_time > 3 else 'Medium' if avg_lead_time > 2 else 'Low',
            'Supplier Concentration': 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low',
            'Service Level': 'Low' if service_level < 0.95 else 'Medium' if service_level < 0.98 else 'High'
        }
        
        risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Level'])
        st.dataframe(risk_df)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = []
        
        if demand_cv > 0.3:
            recommendations.append("• Implement demand sensing to reduce forecast variability")
        if avg_lead_time > 3:
            recommendations.append("• Evaluate supplier lead time reduction opportunities")
        if hhi > 0.25:
            recommendations.append("• Diversify supplier base to reduce concentration risk")
        if service_level < 0.95:
            recommendations.append("• Optimize inventory levels to improve service level")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("✅ Supply chain performance is within acceptable parameters")
    
    # Download options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Results")
    
    if st.sidebar.button("Download Sample Data"):
        demand_sample, supplier_sample, capacity_sample = load_sample_supply_data()
        
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            demand_sample.to_excel(writer, sheet_name='Demand_Data', index=False)
            supplier_sample.to_excel(writer, sheet_name='Supplier_Data', index=False)
            capacity_sample.to_excel(writer, sheet_name='Capacity_Data', index=False)
        
        st.sidebar.download_button(
            label="📊 Download Supply Planning Data",
            data=output.getvalue(),
            file_name="supply_planning_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )