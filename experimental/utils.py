import pandas as pd

def series_to_Xy(data: pd.Series):
    return (data.index.to_numpy().astype(int)/10**9).astype(int).reshape(-1,1), data.to_numpy()