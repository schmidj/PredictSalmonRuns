import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Random Forest MSE: {mse:.2f}")
    
    # Create timeline DataFrame using preserved river names
    timeline_df = test_df[["River_Name", "Year"]].copy()
    timeline_df["Predicted"] = predictions
    timeline_df["Actual"] = y_test.values

    return {
        "MSE": mse,
        "Predicted": predictions,
        "Actual": y_test.values,
        "Feature_Importances": dict(zip(features, best_model.feature_importances_)),
        "Best_Params": grid_search.best_params_,
        "Timeline": timeline_df
    }