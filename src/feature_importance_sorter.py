"""Feature importance ranking via repeated cross-validated tree models."""

import numpy as np
import pandas as pd
from model_generator import ModelGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class FeatureImportanceSorter:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.generator = ModelGenerator(config_type='random', random_seed=random_seed)

    def get_ranking(self, X, y, feature_names, model_type='rf', random_config=False, n_splits=5):
        """
        Returns feature ranking, cross-validated metrics, and model parameters.
        """
        # 1. Prepare model configuration
        self.generator.config_type = 'random' if random_config else 'default'
        model_template, params, config_type = self.generator.generate(model_type)

        # 2. Cross-validation setup
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        
        metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'tp': [], 'fp': []}
        importances = []

        # 3. Training loop
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a fresh model instance for each fold to avoid data leakage
            model = self.generator.model_map[model_type](**params)
            if hasattr(model, 'random_state'):
                model.set_params(random_state=self.random_seed)
            if model_type == 'xgb':
                model.set_params(seed=self.random_seed)
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=-1)
            
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_val)
            cm = confusion_matrix(y_val, preds)
            # Binary classification: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Metrics
            metrics['acc'].append(accuracy_score(y_val, preds))
            metrics['prec'].append(precision_score(y_val, preds, zero_division=0))
            metrics['rec'].append(recall_score(y_val, preds, zero_division=0))
            metrics['f1'].append(f1_score(y_val, preds, zero_division=0))
            metrics['tp'].append(tp)
            metrics['fp'].append(fp)
            
            # Feature Importance
            importances.append(model.feature_importances_)

        # 4. Compile results
        avg_importances = np.asarray(importances).mean(axis=0)

        ranking = pd.DataFrame({
            'feature_name': feature_names,
            'importance': avg_importances,
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        summary = {
            'model_type': model_type,
            'config_type': config_type,
            'random_seed': self.random_seed,
            'n_splits': n_splits,
            **{
                metric: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                for metric, vals in metrics.items()
            },
        }

        return ranking, summary, params