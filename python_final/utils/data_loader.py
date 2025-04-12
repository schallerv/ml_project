import pandas as pd
from setup.config import DATA_PARAMS


def load_games_data(file_path):
    """
    Loads the game data CSV file.
    Assumes that the first DATA_PARAMS['continuous_columns_end_index'] columns are continuous,
    and the remaining columns are binary flags.
    """
    df = pd.read_csv(file_path)

    # split column names: continuous features and binary flags.
    continuous_cols = df.columns[:DATA_PARAMS['continuous_columns_end_index']]
    binary_cols = df.columns[DATA_PARAMS['continuous_columns_end_index']:]

    # ensure binary columns are integer type.
    df[continuous_cols] = df[continuous_cols].apply(pd.to_numeric, errors='coerce')
    df[binary_cols] = df[binary_cols].astype('int')

    return df, list(continuous_cols), list(binary_cols)
