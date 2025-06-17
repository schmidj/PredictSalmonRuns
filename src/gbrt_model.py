import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

def train_and_apply_gbrt_with_tuning(train_df, test_df, target_col="Total_Returns_NextYear"):
    """
    Tunes and applies HistGradientBoostingRegressor using GridSearchCV.

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
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_iter': [100, 200]
    }

    base_model = HistGradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Get permutation importances
    perm_importances = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = dict(zip(features, perm_importances.importances_mean))

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Gradient Boosting MSE: {mse:.2f}")
    
    timeline_df = test_df[["River_Name", "Year"]].copy()
    timeline_df["Predicted"] = predictions
    timeline_df["Actual"] = y_test.values

    return {
        "MSE": mse,
        "Predicted": predictions,
        "Actual": y_test.values,
        "Feature_Importances": importances,
        "Best_Params": grid_search.best_params_,
        "Timeline": timeline_df
    }
