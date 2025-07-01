import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor



def train_and_apply_with_tuning(model, train_df, test_df, topk_feat = 0, target_col="Total_Returns_NextYear"):
    """
    Tunes and applies RandomForestRegressor using GridSearchCV and optionally selects a subset of featues.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        topk_feat (int): Number of features to select for model.
        target_col (str): Target variable to predict.

    Returns:
        dict: R2, MSE, MAPE, predictions, actuals, feature importances, best params, timeline dataframe,
        Metrics_by_System, Metrics_by_River.
    """

    exclude_cols = ["System", target_col, "River_Name", "Year"]
    features = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[features]
    y_train = train_df[target_col]

    X_test = test_df[features]
    y_test = test_df[target_col]

    # Optional: Select top k features with higheset scores
    if (topk_feat > 0):
        selector = SelectKBest(score_func=f_regression, k=topk_feat)
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()]
        X_train = pd.DataFrame(X_train_new, columns=selected_features)
        X_test = pd.DataFrame(X_test_new, columns=selected_features)

    print("Selected features:")
    print(X_train.columns)

    best_params = {} 

    if (model == "RF"): # RandomForestRegressor
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        base_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

    elif (model == "GBRT"): # GradientBoostingRegressor
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 10],
            'min_samples_leaf': [1, 2],
            'max_iter': [100, 200]
        }
        base_model = HistGradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

    elif model == "XGB": # XGBRegressor
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        }
        base_model = XGBRegressor(random_state=42, verbosity=0)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

    elif model == "LR": # LinearRegression
        param_grid = {
            'fit_intercept': [True, False],
            'positive': [False]
        }
        base_model = LinearRegression()
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

    elif model == "PR": # Polynomial Regression
        pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        predictions_train = best_model.predict(X_train)
        predictions = best_model.predict(X_test)
        grid_search = None

    
    if grid_search is not None:
        best_model = grid_search.best_estimator_
        predictions_train = best_model.predict(X_train)
        predictions = best_model.predict(X_test)


    # Performance metrics
    r2_train = r2_score(y_train, predictions_train)
    mse_train = mean_squared_error(y_train, predictions_train)
    mape_train = np.mean(np.abs((y_train - predictions_train) / y_train)) * 100

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    if grid_search is not None:
        print(f"Best Parameters: {grid_search.best_params_}")
    else:
        print("No parameter tuning for this model.")

    print(f"{model} R2: {r2:.2f}")
    print(f"{model} MSE: {mse:.2f}")
    print(f"{model} MAPE: {mape:.2f}")

    # Metrics per System and River_Name
    results_df = test_df.copy()
    results_df["Predicted"] = predictions
    results_df["Actual"] = y_test.values

    # Train predictions for group metrics
    train_results_df = train_df.copy()
    train_results_df["Predicted"] = predictions_train
    train_results_df["Actual"] = y_train.values


    def compute_group_metrics(df):
        r2 = r2_score(df["Actual"], df["Predicted"])
        mse = mean_squared_error(df["Actual"], df["Predicted"])
        mape = np.mean(np.abs((df["Actual"] - df["Predicted"]) / df["Actual"])) * 100
        return pd.Series({"R2": r2, "MSE": mse, "MAPE": mape})

    system_metrics_test = results_df.groupby("System", group_keys=False).apply(
        lambda g: compute_group_metrics(g.drop(columns="System"))
    ).reset_index()

    river_metrics_test = results_df.groupby("River_Name", group_keys=False).apply(
        lambda g: compute_group_metrics(g.drop(columns="River_Name"))
    ).reset_index()

    system_metrics_train = train_results_df.groupby("System", group_keys=False).apply(
        lambda g: compute_group_metrics(g.drop(columns="System"))
    ).reset_index()

    river_metrics_train = train_results_df.groupby("River_Name", group_keys=False).apply(
        lambda g: compute_group_metrics(g.drop(columns="River_Name"))
    ).reset_index()


    # Create timeline DataFrames for plots
    timeline_train_df = train_df[["River_Name", "Year"]].copy()
    timeline_train_df["Predicted"] = predictions_train
    timeline_train_df["Actual"] = y_train.values

    timeline_test_df = test_df[["River_Name", "Year"]].copy()
    timeline_test_df["Predicted"] = predictions
    timeline_test_df["Actual"] = y_test.values

    return {
        "R2_train": r2_train,
        "MSE_train": mse_train,
        "MAPE_train": mape_train,
        "R2": r2,
        "MSE": mse,
        "MAPE": mape,
        "Predicted": predictions,
        "Actual": y_test.values,
        "Best_Params": grid_search.best_params_ if grid_search is not None else None,
        "Timeline_train": timeline_train_df,
        "Timeline_test": timeline_test_df,
        "Metrics_by_System_Test": system_metrics_test,
        "Metrics_by_River_Test": river_metrics_test,
        "Metrics_by_System_Train": system_metrics_train,
        "Metrics_by_River_Train": river_metrics_train,
        "Selected_Feature_Names": list(X_train.columns),
        "Hyperparameters": best_params,
    }

def retrain_model_with_best_params(model_type, best_params, full_train_df, prediction_df, selected_features, river_name, target_col="Total_Returns_NextYear"):
    X_train = full_train_df[selected_features]
    y_train = full_train_df[target_col]
    X_test = prediction_df[selected_features]

    if model_type == "RF":
        model = RandomForestRegressor(random_state=42, **best_params)
    elif model_type == "GBRT":
        model = HistGradientBoostingRegressor(random_state=42, **best_params)
    elif model_type == "XGB":
        model = XGBRegressor(random_state=42, verbosity=0, **best_params)
    elif model_type == "LR":
        model = LinearRegression(**best_params)
    elif model_type == "PR":
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    else:
        raise ValueError(f"Unsupported model type for retraining: {model_type}")

    model.fit(X_train, y_train)
    predictions_retro = model.predict(X_train)
    predictions = model.predict(X_test)

    # Extract retro predictions for years 2020 to 2024
    if "Year" in full_train_df.columns:
        retro_df = full_train_df.copy()
        retro_df["Predicted_Returns"] = predictions_retro

        # Filter for years and river name
        retro_df_filtered = retro_df[retro_df["Year"].between(2019, 2024)]
        retro_df_filtered = retro_df_filtered[retro_df_filtered["River_Name"] == river_name]

        retro_df_filtered = retro_df_filtered[["Year", "Predicted_Returns"]].reset_index(drop=True)
    else:
        retro_df_filtered = pd.DataFrame()
    return predictions, retro_df_filtered