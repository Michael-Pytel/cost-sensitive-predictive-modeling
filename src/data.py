from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(source = 'original'):
    """Load train/test arrays and return scaled versions plus the fitted scaler.
    
    Args:
        source: 'original' or 'scaled_reduced_mi'

    NOTE: if source is 'scaled_reduced_mi', the data is returned as pandas DataFrames and not numpy arrays.
    """
    if source == 'original':
        X_train = np.loadtxt(DATA_DIR / "x_train.txt", skiprows=1)
        y_train = np.loadtxt(DATA_DIR / "y_train.txt", skiprows=1).astype(int)
        X_test  = np.loadtxt(DATA_DIR / "x_test.txt",  skiprows=1)

        scaler  = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        return X_train, y_train, X_test, X_train_s, X_test_s, scaler
    
    elif source == 'scaled_reduced_mi':
        X_train = pd.read_csv(DATA_DIR / "processed" /"x_train_scaled_reduced_after_mi.csv")
        y_train = np.loadtxt(DATA_DIR / "y_train.txt", skiprows=1).astype(int)
        X_test = pd.read_csv(DATA_DIR / "processed" / "x_test_scaled_reduced_after_mi.csv")

        return X_train, y_train, X_test, None, None, None
