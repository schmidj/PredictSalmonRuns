import pandas as pd
from typing import Tuple, List

def split_time_series_by_river(
    df: pd.DataFrame,
    time_column: str = "Year",
    group_columns: List[str] = ["System", "River"],
    test_fraction: float = 0.2,
    gap_years: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-series dataframe into train and test sets by river system,
    maintaining time order and grouping.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        time_column (str): The name of the column representing time (e.g., 'Year').
        group_columns (list): Columns to group by (e.g., ['System', 'River']).
        test_fraction (float): Fraction of data to allocate to the test set for each group.
        gap_years (int): Optional number of years to leave as a gap between training and testing.

    Returns:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Testing set.
    """
    train_list = []
    test_list = []

    grouped = df.groupby(group_columns)

    for group_key, group_df in grouped:
        group_df = group_df.sort_values(by=time_column)
        n_samples = len(group_df)

        n_test = int(round(n_samples * test_fraction))
        n_train = n_samples - n_test - gap_years

        if n_train < 1 or n_test < 1:
            print(f"Skipping group {group_key} due to insufficient samples.")
            continue

        train_part = group_df.iloc[:n_train]
        test_part = group_df.iloc[n_train + gap_years:]

        train_list.append(train_part)
        test_list.append(test_part)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df