from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# 1. Lasso Regression
base_lasso = Lasso(max_iter=10000, random_state=42)
lasso_space = {
    'alpha': Real(1e-4, 10.0, prior='log-uniform')
}
opt_lasso = BayesSearchCV(
    base_lasso,
    lasso_space,
    n_iter=30,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# 2. Random Forest
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_params = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(5, 50),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', 0.1, 0.3, 0.5, 1.0]),
    'criterion': Categorical(['squared_error', 'absolute_error'])
}
bayesearch_rf = BayesSearchCV(
    base_rf,
    rf_params,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# 3. XGBoost
base_xgb = XGBRegressor(
    tree_method='hist',  # Use 'gpu_hist' if GPU is available
    predictor='auto',  # Automatically choose the best predictor
    random_state=42,
    n_jobs=-1
)
gpu_xgb = XGBRegressor(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    random_state=42,
    n_jobs=-1
)
xgb_params = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5)
}

bayesearch_xgb = BayesSearchCV(
    base_xgb,
    xgb_params,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# 4. LightGBM
base_lgb = LGBMRegressor(
    device='cpu',  # Use 'gpu' if GPU is available
    random_state=42,
    n_jobs=-1
)
gpu_lgb = LGBMRegressor(
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=42,
    n_jobs=-1
)
lgb_params = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'num_leaves': Integer(20, 150),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_alpha': Real(0, 5),
    'reg_lambda': Real(0, 5)
}
bayessearch_lgb = BayesSearchCV(
    base_lgb,
    lgb_params,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# 5. Gradient Boosting
base_gb = GradientBoostingRegressor(random_state=42)
gb_params = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'min_samples_leaf': Integer(1, 10)
}
bayessearch_gb = BayesSearchCV(
    base_gb,
    gb_params,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)