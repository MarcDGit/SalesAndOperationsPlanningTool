## Overview

This tool provides several modules to support the Sales & Operations Planning process.  The *Inventory Planning Module* lets planners upload historical demand data, detect outliers, and automatically generate inventory parameters such as safety stock, reorder point and EOQ.

## Quick-start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Launch the web application**

```bash
streamlit run main.py
```

3. **Navigate**

Use the sidebar to select the *Inventory Planning* module.  Upload your CSV or Excel file and follow the on-screen prompts.

## Data requirements

Your flat-file should include at least:

| Column      | Description                                  |
|-------------|----------------------------------------------|
| `week`      | Week number or date                          |
| `sku`       | Stock Keeping Unit identifier                |
| `location`  | Warehouse or node                            |
| `demand`    | Demand for the given `week-sku-location`     |
| `inventory` | Optional – current inventory on hand         |

Additional columns are accepted and can be used to group time-series independently.

## Functionality

1. **Outlier detection**  – configurable Z-score method applied per SKU/Location (or any grouping chosen).
2. **Inventory parameter optimisation**  – calculates safety stock, reorder point and EOQ based on uploaded history, planner inputs for lead-time, service level, ordering and holding cost.

Each result can be downloaded as CSV or Excel for further analysis. 
