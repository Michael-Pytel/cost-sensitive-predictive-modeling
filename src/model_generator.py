"""Random hyperparameter sampling for tree-based classifiers."""

import random
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelGenerator:
    """
    Generates random model configurations from defined parameter grids.
    """
    def __init__(self, config_type='random', random_seed=42):
        self.config_type = config_type
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Define the parameter grids
        self.param_grids = {
            'rf': {
                'n_estimators': stats.randint(50, 500),
                'max_depth': list(range(4, 12)),
                'min_samples_leaf': stats.randint(1, 5),
                'min_samples_split': stats.randint(2, 10),
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
            },
            'extra': {
                'n_estimators': stats.randint(50, 500),
                'max_depth': [None] + list(range(3, 15)),
                'min_samples_split': stats.randint(2, 10),
                'min_samples_leaf': stats.randint(1, 10),
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
            },
            'xgb': {
                'n_estimators': stats.randint(100, 400),
                'learning_rate': stats.uniform(0.03, 0.15),
                'max_depth': stats.randint(3, 10),
                'subsample': stats.uniform(0.5, 0.5),
                'colsample_bytree': stats.uniform(0.5, 0.5),
                'min_child_weight': stats.randint(1, 10),
                'reg_alpha': stats.uniform(0, 1),
                'reg_lambda': stats.uniform(0.5, 2),
            },
            'lgbm': {
                'n_estimators': stats.randint(100, 400),
                'learning_rate': stats.uniform(0.03, 0.15),
                'num_leaves': stats.randint(20, 100),
                'min_child_samples': stats.randint(10, 50),
                'colsample_bytree': stats.uniform(0.5, 0.5),
                'reg_alpha': stats.uniform(0, 1),
                'reg_lambda': stats.uniform(0.5, 2),
            },
        }

        self.model_map = {
            'rf': RandomForestClassifier,
            'extra': ExtraTreesClassifier,
            'xgb': XGBClassifier,
            'lgbm': lambda **kwargs: LGBMClassifier(verbose=-1, **kwargs)
        }

    def generate(self, model_type='rf'):
        """
        Generates a model instance with random hyperparameters.
        If random_seed=None, returns a base model with default parameters.
        """
        if model_type not in self.model_map:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Choose from: {list(self.model_map.keys())}"
            )

        # random_seed=None -> use default parameters
        if self.config_type == 'default':
            model = self.model_map[model_type]()
            return model, {}, "default"

        params = {
            k: (v.rvs() if hasattr(v, 'rvs') else random.choice(v))
            for k, v in self.param_grids[model_type].items()
        }

        if model_type == 'xgb':
            model = self.model_map[model_type](**params, seed=self.random_seed)
        else:
            model = self.model_map[model_type](**params, random_state=self.random_seed)

        return model, params, "random"