
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="HHS Care Load Forecasting",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

dashboard_forecast = pd.read_csv(
    "dashboard_forecast.csv",
    parse_dates=["Date"]
)

future_model_comparison = pd.read_csv(
    "future_model_comparison.csv",
    parse_dates=["Date"]
)

scenario_summary = pd.read_csv(
    "scenario_summary.csv"
)

daily_data = pd.read_csv(
    "daily_data.csv",
    parse_dates=["Date"]
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

st.markdown(
    """
    **HHS Care Forecasting Dashboard**

    This dashboard provides short-term forecasts for:
    - Children in HHS Care
    - Discharge demand
    - Forecast uncertainty
    - Model comparison
    - Intake-vs-exit scenarios
    - Historical relative-load monitoring
    """
)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Forecast Controls")

horizon = st.sidebar.radio(
    "Forecast Horizon",
    [7, 14, 30],
    index=2
)

model_choice = st.sidebar.radio(
    "HHS Care Model",
    ["Random Forest", "Gradient Boosting"]
)

# Select forecast horizon
forecast_view = dashboard_forecast.head(horizon)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

latest_hhs = daily_data[
    "Children in HHS Care"
].iloc[-1]

avg_hhs_forecast = forecast_view[
    "Predicted_HHS_Care"
].mean()

avg_discharge_forecast = forecast_view[
    "Predicted_Discharge_Demand"
].mean()

latest_date = daily_data["Date"].max()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest HHS Care",
    f"{latest_hhs:,.0f}"
)

col2.metric(
    f"{horizon}-Day Avg HHS Care",
    f"{avg_hhs_forecast:,.0f}"
)

col3.metric(
    f"{horizon}-Day Avg Discharge",
    f"{avg_discharge_forecast:.2f}"
)

col4.metric(
    "Latest Data Date",
    latest_date.strftime("%d %b %Y")
)

# --------------------------------------------------
# HHS CARE FORECAST
# --------------------------------------------------

st.header("📈 HHS Care Load Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

if model_choice == "Random Forest":

    ax.plot(
        forecast_view["Date"],
        forecast_view["Predicted_HHS_Care"],
        label="Random Forest Forecast"
    )

    ax.fill_between(
        forecast_view["Date"],
        forecast_view["Lower_Bound_HHS"],
        forecast_view["Upper_Bound_HHS"],
        alpha=0.2,
        label="Approximate Prediction Interval"
    )

else:

    gb_view = future_model_comparison.head(horizon)

    ax.plot(
        gb_view["Date"],
        gb_view["Predicted_HHS_Care_GB"],
        label="Gradient Boosting Forecast"
    )

ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Future HHS Care Load")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

st.caption(
    "The Random Forest uncertainty band is an approximate "
    "residual-based prediction interval, not a formal statistical "
    "confidence interval."
)

# --------------------------------------------------
# DISCHARGE FORECAST
# --------------------------------------------------

st.header("🚪 Discharge Demand Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast_view["Date"],
    forecast_view["Predicted_Discharge_Demand"],
    label="Discharge Forecast"
)

ax.fill_between(
    forecast_view["Date"],
    forecast_view["Lower_Bound_Discharge"],
    forecast_view["Upper_Bound_Discharge"],
    alpha=0.2,
    label="Approximate Prediction Interval"
)

ax.set_xlabel("Date")
ax.set_ylabel("Expected Discharges")
ax.set_title("Future Discharge Demand")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.header("🤖 Model Performance Comparison")

model_metrics = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        39.2935,
        54.2463
    ],
    "RMSE": [
        61.2575,
        76.3687
    ],
    "MAPE (%)": [
        1.8619,
        2.5702
    ]
})

st.dataframe(
    model_metrics,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DISCHARGE MODEL COMPARISON
# --------------------------------------------------

st.subheader("Discharge Model Performance")

discharge_metrics = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        3.4749,
        4.4279
    ],
    "RMSE": [
        4.6743,
        5.4477
    ],
    "MAPE (%)": [
        30.7892,
        54.6668
    ]
})

st.dataframe(
    discharge_metrics,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# SCENARIO ANALYSIS
# --------------------------------------------------

st.header("🔄 Intake vs Exit Scenario Analysis")

st.dataframe(
    scenario_summary,
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    scenario_summary["Scenario"],
    scenario_summary["Cumulative 30-Day Imbalance"]
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_ylabel("Cumulative 30-Day Imbalance")
ax.set_title("Intake vs Exit Scenario Comparison")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)

st.caption(
    "Positive imbalance means assumed intake exceeds predicted "
    "discharge demand. Negative imbalance means predicted "
    "discharge demand exceeds assumed intake."
)

# --------------------------------------------------
# RELATIVE LOAD INDICATOR
# --------------------------------------------------

st.header("⚠️ Relative Load Monitoring")

historical_threshold = daily_data[
    "Children in HHS Care"
].quantile(0.90)

forecast_max = forecast_view[
    "Predicted_HHS_Care"
].max()

st.write(
    f"Historical 90th-percentile HHS Care level: "
    f"**{historical_threshold:,.0f}**"
)

st.write(
    f"Maximum forecasted HHS Care over selected horizon: "
    f"**{forecast_max:,.0f}**"
)

if forecast_max >= historical_threshold:
    st.warning(
        "Forecast reaches the historical high-load reference level."
    )
else:
    st.success(
        "Forecast remains below the historical high-load reference level."
    )

st.caption(
    "This is a historical relative-load indicator. "
    "The dataset does not contain an actual facility capacity value, "
    "so this should not be interpreted as an actual capacity limit."
)

# --------------------------------------------------
# HISTORICAL DATA
# --------------------------------------------------

st.header("📚 Historical HHS Care Data")

historical_view = daily_data[
    ["Date", "Children in HHS Care"]
].tail(180)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    historical_view["Date"],
    historical_view["Children in HHS Care"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Recent Historical HHS Care Load")
ax.grid(alpha=0.3)

st.pyplot(fig)

# --------------------------------------------------
# FORECAST TABLE
# --------------------------------------------------

st.header("📋 Forecast Details")

st.dataframe(
    forecast_view,
    use_container_width=True,
    hide_index=True
)

st.success("Dashboard loaded successfully!")
