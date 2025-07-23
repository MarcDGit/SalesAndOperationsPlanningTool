import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from io import BytesIO


def detect_outliers(df: pd.DataFrame, target_col: str, group_cols=None, z_threshold: float = 3.0):
    """Return dataframe with z-scores and outlier flag based on the chosen group columns."""
    group_cols = group_cols or []

    df_out = df.copy()

    if group_cols:
        # Compute z-score within each group
        df_out["zscore"] = (
            df_out.groupby(group_cols)[target_col]
            .transform(lambda x: np.abs(stats.zscore(x, nan_policy="omit")))
        )
    else:
        df_out["zscore"] = np.abs(stats.zscore(df_out[target_col], nan_policy="omit"))

    df_out["is_outlier"] = df_out["zscore"] > z_threshold
    return df_out


def inventory_parameters(
    df: pd.DataFrame,
    demand_col: str,
    group_cols=None,
    lead_time: float = 1.0,
    service_level: float = 0.95,
    order_cost: float = 100.0,
    holding_cost: float = 2.0,
):
    """Compute basic inventory parameters (safety stock, reorder point, EOQ)."""

    group_cols = group_cols or []
    z_service = stats.norm.ppf(service_level)

    # Aggregate demand statistics
    grouped = df.groupby(group_cols)[demand_col]
    mu = grouped.mean()
    sigma = grouped.std(ddof=1).fillna(0)
    total_demand = grouped.sum()

    # Safety stock & Reorder point
    safety_stock = z_service * sigma * np.sqrt(lead_time)
    reorder_point = mu * lead_time + safety_stock

    # EOQ calculation – annualised simplistic view
    eoq = np.sqrt((2 * total_demand * order_cost) / holding_cost)

    result = pd.DataFrame({
        "average_demand": mu,
        "demand_std": sigma,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "eoq": eoq,
    }).reset_index()

    return result


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Transform dataframe to Excel bytes so user can download."""
    with BytesIO() as b_io:
        df.to_excel(b_io, index=False, engine="openpyxl")
        return b_io.getvalue()


def run():
    st.title("Inventory Planning Module")

    st.markdown(
        """
        Upload a **CSV** or **Excel** file containing at least the following columns:
        - `week` : Time bucket (week number or date)
        - `sku` : Stock keeping unit identifier
        - `location` : Location / warehouse
        - `demand` : Demand for that week-SKU-location
        - `inventory` *(optional)* : On-hand inventory
        """
    )

    file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    if file is None:
        st.info("Awaiting file upload …")
        return

    # Read uploaded file
    try:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    if df.empty:
        st.warning("The uploaded file is empty.")
        return

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    analysis_option = st.selectbox(
        "Select analysis type", [
            "Outlier Detection",
            "Inventory Parameter Optimization",
        ]
    )

    # Determine common columns automatically
    default_group_cols = []
    for col in ["sku", "location"]:
        if col in df.columns:
            default_group_cols.append(col)

    demand_col = "demand" if "demand" in df.columns else df.columns[0]

    if analysis_option == "Outlier Detection":
        st.header("Outlier Detection")
        group_cols = st.multiselect(
            "Group columns (compute outliers within each group)",
            options=list(df.columns),
            default=default_group_cols,
        )
        z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0, step=0.1)
        target_col = st.selectbox("Target demand column", list(df.columns), index=list(df.columns).index(demand_col))

        if st.button("Run Outlier Detection"):
            result_df = detect_outliers(df, target_col, group_cols, z_threshold=z_thresh)
            st.success("Outlier detection completed ✔️")
            st.write(f"Total outliers found: {result_df['is_outlier'].sum()}")
            st.dataframe(result_df[result_df["is_outlier"]].head())

            # Offer download
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="outlier_detection_results.csv",
                mime="text/csv",
            )

    elif analysis_option == "Inventory Parameter Optimization":
        st.header("Inventory Parameter Optimization")

        group_cols = st.multiselect(
            "Group columns (each group treated as an item)",
            options=list(df.columns),
            default=default_group_cols,
        )

        demand_col = st.selectbox("Demand column", list(df.columns), index=list(df.columns).index(demand_col))

        lead_time = st.number_input("Lead time (in weeks)", min_value=0.1, value=1.0, step=0.1)
        service_level = st.slider("Service level target", 0.5, 0.999, 0.95, step=0.005)
        order_cost = st.number_input("Order cost per order", min_value=0.0, value=100.0, step=10.0)
        holding_cost = st.number_input("Holding cost per unit per year", min_value=0.01, value=2.0, step=0.1)

        if st.button("Generate Inventory Parameters"):
            params_df = inventory_parameters(
                df,
                demand_col=demand_col,
                group_cols=group_cols,
                lead_time=lead_time,
                service_level=service_level,
                order_cost=order_cost,
                holding_cost=holding_cost,
            )
            st.success("Inventory parameters generated ✔️")
            st.dataframe(params_df.head())

            # Download options
            st.download_button(
                "Download as CSV",
                data=params_df.to_csv(index=False).encode("utf-8"),
                file_name="inventory_parameters.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download as Excel",
                data=to_excel_bytes(params_df),
                file_name="inventory_parameters.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )