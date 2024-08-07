import pandas as pd

DATASET_FILEPATH = "datasets/database.csv"

def load_dataset():
    df = pd.read_csv(DATASET_FILEPATH, header=0, sep=";", index_col=0)
    df.index = pd.to_datetime(df.index, unit='s')
    df.index.names = ['Datetime']
    df = df.rename(columns={"X1": "Production", "X2": "Demand", "RCE": "Price"})
    return df


"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data = load_dataset()
    print(data.head())