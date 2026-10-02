# Predictive Forecasting of Care Load & Placement Demand

## Project Overview

This project develops a predictive forecasting system for estimating **Children in HHS Care** and **short-term discharge demand** using time-series analysis and machine learning.

The project compares statistical forecasting approaches with machine learning models and provides an interactive Streamlit dashboard for exploring future forecasts, uncertainty, model performance, and intake-versus-exit scenarios.

## Objectives

- Forecast the number of children in HHS care.
- Predict short-term discharge demand.
- Identify intake-versus-exit imbalance scenarios.
- Provide uncertainty estimates around forecasts.
- Compare baseline, statistical, and machine learning forecasting approaches.
- Develop an interactive Streamlit dashboard for decision support.

## Dataset

The dataset contains daily observations covering:

- Children apprehended and placed in CBP custody
- Children in CBP custody
- Children transferred out of CBP custody
- Children in HHS Care
- Children discharged from HHS Care

The original data spans **January 2023 to December 2025**.

Missing calendar dates were identified and a continuous daily time series was created using time-based interpolation.

## Methodology

### Data Preparation

- Converted dates to datetime format.
- Sorted observations chronologically.
- Created a continuous daily date range.
- Identified missing calendar dates.
- Applied time-based interpolation.
- Preserved a missing-date indicator.

### Feature Engineering

The forecasting models use:

- Lag 1 day
- Lag 7 days
- Lag 14 days
- 7-day rolling mean
- 14-day rolling mean
- 7-day rolling variance
- 14-day rolling variance
- Flow signal: Transfers − Discharges
- Day of week
- Month

Rolling features were calculated using previous observations to reduce target leakage.

### Train/Test Strategy

An **80% training and 20% testing split** was performed chronologically.

Random sampling was not used because this is a time-series forecasting problem.

## Models

The project evaluates:

### Baseline Models
- Naive Persistence
- 7-Day Moving Average

### Statistical Models
- Exponential Smoothing
- ARIMA

### Machine Learning Models
- Random Forest Regressor
- Gradient Boosting Regressor

A separate forecasting model was developed for discharge demand.

## HHS Care Forecasting Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive Persistence | 6.65 | 8.25 | 0.30% |
| 7-Day Moving Average | 21.56 | 26.58 | 0.96% |
| Exponential Smoothing | 211.56 | 264.82 | 9.29% |
| ARIMA | 233.32 | 286.88 | 11.00% |
| Random Forest | 39.29 | 61.26 | 1.86% |
| Gradient Boosting | 54.25 | 76.37 | 2.57% |

## Discharge Demand Results

Random Forest was also used for short-term discharge demand forecasting.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Random Forest | 3.47 | 4.67 | 30.79% |
| Gradient Boosting | 4.43 | 5.45 | 54.67% |

MAPE for discharge demand can be high when actual discharge values are small or zero, so MAE and RMSE are also considered when interpreting performance.

## Future Forecast

The dashboard provides forecasts for:

- 7 days
- 14 days
- 30 days

The latest recorded HHS Care value is **2,484** on **21 December 2025**.

The 30-day forecast has an average predicted HHS Care level of approximately **2,480**.

The 30-day average predicted discharge demand is approximately **9 discharges per day**.

## Scenario Analysis

The dashboard provides three intake scenarios:

- Low Intake
- Expected Intake
- High Intake

The scenarios compare assumed intake against predicted discharge demand to identify potential intake-versus-exit imbalance.

These scenarios represent assumptions rather than direct predictions of future transfers.

## Uncertainty

Forecast uncertainty is represented using residual-based prediction intervals.

These intervals are approximate and should not be interpreted as formal statistical confidence intervals.

## Streamlit Dashboard

The interactive dashboard provides:

- Future HHS Care load forecast
- Discharge demand forecast
- Forecast horizon selection
- HHS Care model selection
- Model comparison
- Prediction interval visualization
- Intake-versus-exit scenario analysis
- Historical relative-load monitoring

The dashboard uses the historical 90th percentile of HHS Care as a **relative-load indicator**. The dataset does not contain an actual facility capacity value, so this should not be interpreted as a direct capacity-exceedance measure.

## Project Files

- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies
- `dashboard_forecast.csv` — HHS Care and discharge forecasts
- `future_model_comparison.csv` — Future model predictions
- `scenario_summary.csv` — Scenario analysis results
- `discharge_forecast_summary.csv` — Discharge forecast summary
- `daily_data.csv` — Prepared daily dataset
- `horizon_df.csv` — Forecast horizon results

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Streamlit

## Limitations

- The dataset contains missing calendar dates that required interpolation.
- The dataset does not provide an explicit operational capacity value.
- Future scenario analysis depends on assumed intake levels.
- Prediction intervals are approximate residual-based intervals.
- Model evaluation approaches differ across baseline, statistical, and ML models and should be interpreted accordingly.

## Conclusion

This project demonstrates how time-series analysis and machine learning can be combined to forecast HHS care load and discharge demand. The resulting Streamlit dashboard provides an accessible interface for reviewing forecasts, uncertainty, model comparisons, and intake-versus-exit scenarios.
