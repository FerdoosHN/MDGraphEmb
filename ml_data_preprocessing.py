import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

report_dir = "report"
os.makedirs(report_dir, exist_ok=True)

def load_and_prepare_data(filepath):
    """
    Load and prepare dataset.
    """
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    print(df.head())
    print(df.info())
    print(df.describe())

    if 'state' not in df.columns:
        raise ValueError("Target column 'state' is missing from the dataset.")
    else:
        print("Target column 'state' is present.")

    x = df.drop(['state'], axis='columns')
    y = df.state

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return df, x, y_encoded, label_encoder

def split_data(x, y):
    """
    Split data into training and testing sets.
    """
    indices = np.arange(x.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=50)
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print("Data split into training and testing sets.")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test, test_indices




