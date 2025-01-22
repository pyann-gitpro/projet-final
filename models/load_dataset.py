import pandas as pd

def load_data():
    """
    Load the data from the clean_data.csv file and return it as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv("./data/raw/data_fraud_calls.csv")
       
    return data