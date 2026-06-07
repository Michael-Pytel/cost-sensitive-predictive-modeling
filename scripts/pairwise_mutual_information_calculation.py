"""Compute pairwise mutual information for all feature pairs (parallelized)."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from joblib import Parallel, delayed

_, y_train, _, X_train, _, _ = load_data()

def compute_mi(pair, X):
    i, j = pair
    mi_score = mutual_info_regression(X[:, [i]], X[:, j])[0]
    print(f"Feature {i=}, {j=} {mi_score=}")
    return {'feature_i': i, 'feature_j': j, 'mi_score': mi_score}

def calculate_all_mi_pairs_parallel(X, output_file='mi_results.csv', n_jobs=8):
    """Calculates all pairwise Mutual Information using 8 CPU cores."""
    n_features = X.shape[1]
    pairs = list(combinations(range(n_features), 2))
    
    print(f"Starting parallel calculation for {len(pairs)} pairs using {n_jobs} cores...")
    
    mi_data = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_mi)(pair, X) for pair in pairs
    )
    
    df = pd.DataFrame(mi_data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df

calculate_all_mi_pairs_parallel(X_train)