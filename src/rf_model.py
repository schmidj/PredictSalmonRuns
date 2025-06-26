import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def train_and_apply_rf_with_tuning(train_df, test_df, target_col="Total_Returns_NextYear"):
    """
    Tunes and applies RandomForestRegressor using GridSearchCV.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        target_col (str): Target variable to predict.

    Returns:
        dict: MSE, predictions, actuals, feature importances, best params, timeline dataframe.
    """
    exclude_cols = ["System", "Year", target_col, "River_Name"]
    features = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[features]
    y_train = train_df[target_col]

    X_test = test_df[features]
    y_test = test_df[target_col]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    base_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions_train = best_model.predict(X_train)
    predictions = best_model.predict(X_test)

    # Performance metrics
    r2_train = r2_score(y_train, predictions_train)
    mse_train = mean_squared_error(y_train, predictions_train)
    mape_train = np.mean(np.abs((y_train - predictions_train) / y_test)) * 100

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Random Forest R2: {r2:.2f}")
    print(f"Random Forest MSE: {mse:.2f}")
    print(f"Random Forest MAPE: {mape:.2f}")

    # Metrics per System and River_Name
    results_df = test_df.copy()
    results_df["Predicted"] = predictions
    results_df["Actual"] = y_test.values

    def compute_group_metrics(df):
        r2 = r2_score(df["Actual"], df["Predicted"])
        mse = mean_squared_error(df["Actual"], df["Predicted"])
        mape = np.mean(np.abs((df["Actual"] - df["Predicted"]) / df["Actual"])) * 100
        return pd.Series({"R2": r2, "MSE": mse, "MAPE": mape})
    
    system_metrics = results_df.groupby("System").apply(compute_group_metrics).reset_index()
    river_metrics = results_df.groupby("River_Name").apply(compute_group_metrics).reset_index()
    
    # Create timeline DataFrame using preserved river names
    timeline_df = test_df[["River_Name", "Year"]].copy()
    timeline_df["Predicted"] = predictions
    timeline_df["Actual"] = y_test.values

    return {
        "R2_train": r2_train,
        "MSE_train": mse_train,
        "MAPE_train": mape_train,
        "R2": r2,
        "MSE": mse,
        "MAPE": mape,
        "Predicted": predictions,
        "Actual": y_test.values,
        "Feature_Importances": dict(zip(features, best_model.feature_importances_)),
        "Best_Params": grid_search.best_params_,
        "Timeline": timeline_df,
        "Metrics_by_System": system_metrics,
        "Metrics_by_River": river_metrics
    }