from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    """Load train/test arrays and return scaled versions plus the fitted scaler."""
    X_train = np.loadtxt(DATA_DIR / "x_train.txt", skiprows=1)
    y_train = np.loadtxt(DATA_DIR / "y_train.txt", skiprows=1).astype(int)
    X_test  = np.loadtxt(DATA_DIR / "x_test.txt",  skiprows=1)

    scaler  = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train, y_train, X_test, X_train_s, X_test_s, scaler
