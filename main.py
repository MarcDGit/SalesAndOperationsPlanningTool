# main.py

import streamlit as st
from modules import demand_planning, supply_planning, inventory_planning, financial_planning, sop

def main():
    st.title("Sales and Operations Planning Tool")

    # Sidebar navigation
    selected_module = st.sidebar.radio("Select Module", ["Demand Planning", "Supply Planning", "Inventory Planning", "Financial Planning", "SOP"])

    # Load selected module
    if selected_module == "Demand Planning":
        demand_planning.run()
    elif selected_module == "Supply Planning":
        supply_planning.run()
    elif selected_module == "Inventory Planning":
        inventory_planning.run()
    elif selected_module == "Financial Planning":
        financial_planning.run()
    elif selected_module == "SOP":
        sop.run()

if __name__ == "__main__":
    main()
