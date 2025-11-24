import xgboost as xgb
from .base_model import ModelBase

class XGBoostModel(ModelBase):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,
                 subsample=1.0, colsample_bytree=1.0, random_state=42):
        super().__init__()
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

    @staticmethod
    def get_param_grid():
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

    @staticmethod
    def get_random_param_distributions():
        from scipy.stats import uniform, randint
        return {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }
