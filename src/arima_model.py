from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def apply_arima_forecast_last_year(df, system_col="System", river_col="River", time_col="Year", target_col="Total_Returns"):
    """
    Trains an ARIMA(1,1,0) model for each (System, River) using all years except the last,
    and forecasts the Total_Returns for the last year (e.g., 2024).

    Returns:
        List of dictionaries with prediction, actual, MSE, and metadata per river.
    """
    results = []

    for (system, river), group in df.groupby([system_col, river_col]):
        if len(group) < 4:
            continue  # Not enough data to train + test

        group = group.sort_values(by=time_col)
        train_data = group.iloc[:-1]
        test_data = group.iloc[-1:]

        train_series = train_data[target_col].values
        actual = test_data[target_col].values[0]
        forecast_year = test_data[time_col].values[0]

        try:
            model = ARIMA(train_series, order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            mse = mean_squared_error([actual], [forecast])

            results.append({
                "System": system,
                "River": river,
                "Year": forecast_year,
                "Predicted": forecast,
                "Actual": actual,
                "MSE": mse
            })

            print(f"{system} - {river} ({forecast_year}): Predicted={forecast:.0f}, Actual={actual:.0f}, MSE={mse:.2f}")

        except Exception as e:
            print(f"{system} - {river}: ARIMA failed due to {e}")

    return results