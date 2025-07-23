# main.py

import streamlit as st
from modules import demand_planning, supply_planning, inventory_planning, financial_planning, sop

def main():
    st.set_page_config(
        page_title="Supply Chain Planning Suite",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏭 Sales and Operations Planning Suite")
    
    # Add description
    st.markdown("""
    Welcome to the comprehensive Sales and Operations Planning tool. This suite includes advanced 
    inventory planning capabilities with outlier detection, optimization algorithms, and parameter proposals.
    """)

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    selected_module = st.sidebar.radio(
        "Select Module", 
        ["🏭 Inventory Planning", "📊 Demand Planning", "🚚 Supply Planning", "💰 Financial Planning", "📋 SOP"],
        help="Choose the planning module you want to work with"
    )
    
    # Add sample data download link
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Sample Data")
    
    with open("sample_inventory_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="📋 Download Sample Data",
            data=file,
            file_name="sample_inventory_data.csv",
            mime="text/csv",
            help="Download sample inventory data to test the application"
        )

    # Load selected module
    if selected_module == "🏭 Inventory Planning":
        inventory_planning.run()
    elif selected_module == "📊 Demand Planning":
        demand_planning.run()
    elif selected_module == "🚚 Supply Planning":
        supply_planning.run()
    elif selected_module == "💰 Financial Planning":
        financial_planning.run()
    elif selected_module == "📋 SOP":
        sop.run()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This comprehensive supply chain planning suite provides advanced analytics "
        "for demand forecasting, inventory optimization, and operational planning."
    )

if __name__ == "__main__":
    main()
