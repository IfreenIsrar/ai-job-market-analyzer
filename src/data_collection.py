import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    print("Initial shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df