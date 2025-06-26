"""
preprocess_train
evaluate_model
plot_actual_vs_pred

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr


def get_cols_inf(df):
    """
    Returns a list of column names that contain positive or negative infinity.
    """
    return df.columns[np.isinf(df.values).any(axis=0)].tolist()

def get_cols_zerostd(df):
    """
    Returns a list of column names with zero standard deviation (excluding NaNs).
    """
    nunique_non_nan = df.nunique(dropna=True)
    return nunique_non_nan[nunique_non_nan <= 1].index.tolist()

def get_nan_columns(df):
    """
    Returns a list of column names that contain NaN values.
    """
    return df.columns[df.isna().any()].tolist()

# def preprocess_train(train, columns_to_drop=[]):
#     df = train.copy()
    
#     #### Preprocessing
#     # Identify once at the start
#     cols_inf = get_cols_inf(df)
#     print("Columns with infinite values:", cols_inf)
#     cols_nan = get_nan_columns(df)
#     print("Columns with NaN values:", cols_nan)
#     cols_zerostd = get_cols_zerostd(df)
#     print("Columns with zero standard deviation:", cols_zerostd)
#     # Drop all at once
#     cols_to_drop = set(cols_inf) | set(cols_nan) | set(cols_zerostd)

#     df = df.drop(columns=cols_to_drop)

#     #### Feature Engineering

#     df.loc[:, 'bidask_ratio'] = df['bid_qty'] / df['ask_qty']
#     df.loc[:, 'buysell_ratio'] = np.where(df['volume'] == 0, 0, df['buy_qty'] / df['sell_qty'])

#     # df.loc[:, 'buysell_ratio_shift1'] = df['buysell_ratio'].shift(-1)

#     df.loc[:, 'bidask_delta'] = df['bid_qty'] - df['ask_qty']
#     df.loc[:, 'buysell_delta'] = df['buy_qty'] - df['sell_qty']

#     df.loc[:, 'buysell_size'] = df['buy_qty'] + df['sell_qty']
#     df.loc[:, 'bidask_size'] = df['bid_qty'] + df['ask_qty']

#     # Final Drop
#     df = df.drop(columns=columns_to_drop)
#     return df

def evaluate_model_kaggle(y_pred, y_test):
    """
    Returns the Pearson correlation coefficient between predictions and true values.
    """
    corr, _ = pearsonr(y_pred, y_test)
    return corr

def evaluate_model(y_true, y_pred, X=None, linear=False, verbose=True):
    """
    General evaluation of regression models.
    Inputs:
        y_true: True target values
        y_pred: Predicted target values
        X: Feature matrix (optional, for adj_r2 and n_features)
        linear: If True, AIC and BIC will be computed (meaningful for linear models only)
        verbose: Print the results

    Outputs (dict):
        n_obs: Number of observations
        n_features: Number of features (if X provided)
        r2: R^2 score
        adj_r2: Adjusted R^2 (only if X is provided)
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        medae: Median Absolute Error
        pearson_corr, pearson_pvalue
        spearman_corr, spearman_pvalue
        aic: Akaike Information Criterion (only if linear=True and X provided)
        bic: Bayesian Information Criterion (only if linear=True and X provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_obs = len(y_true)
    n_features = X.shape[1] if X is not None else None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    adj_r2 = (
        1 - (1 - r2) * (n_obs - 1) / (n_obs - n_features - 1)
        if X is not None and n_obs > n_features + 1 else np.nan
    )

    if linear and X is not None:
        n_params = n_features + 1  # +1 for intercept
        rss = np.sum((y_true - y_pred)**2)
        aic = n_obs * np.log(rss / n_obs) + 2 * n_params
        bic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)
    else:
        aic = np.nan
        bic = np.nan

    results = {
        "n_obs": n_obs,
        "n_features": n_features,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "pearson_corr": pearson_corr,
        "pearson_pvalue": pearson_p,
        # "spearman_corr": spearman_corr,
        # "spearman_pvalue": spearman_p,
        "aic": aic,
        "bic": bic,
    }

    if verbose:
        print(f"Observations:            {n_obs}")
        if n_features is not None:
            print(f"Features:                {n_features}")
        print(f"R^2:                     {r2:.5f}")
        print(f"Adjusted R^2:            {adj_r2:.5f}")
        print(f"RMSE:                    {rmse:.5f}")
        print(f"MAE:                     {mae:.5f}")
        print(f"Median Absolute Error:   {medae:.5f}")
        print(f"Pearson Corr:            {pearson_corr:.5f} (p={pearson_p:.3g})")
        print(f"Spearman Corr:           {spearman_corr:.5f} (p={spearman_p:.3g})")
        if linear and X is not None:
            print(f"AIC:                     {aic:.2f}")
            print(f"BIC:                     {bic:.2f}")
    return results

def plot_actual_vs_pred(y_train, y_test, y_pred, figsize=(15, 5)):
    """
    Plots actual train, test and predicted test values on a time series plot.
    Assumes all inputs are pandas Series with datetime index.
    """
    # Rename for clarity
    df_y_train = pd.DataFrame(y_train).reset_index().rename(columns={'label': "Train"})
    df_y_test = pd.DataFrame(y_test).reset_index().rename(columns={'label': "Test"})
    df_y_pred = pd.DataFrame(y_pred, index=y_test.index).reset_index().rename(columns={ 0: "Predicted"})
    # Create a combined DataFrame (aligns on index)
    df_plot = df_y_train.merge(df_y_test, on="timestamp", how = 'outer').merge(df_y_pred, on="timestamp", how='outer')
    # return df_plot
    # Plot
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df_plot, x='timestamp', y='Train', label='Train', ax=ax)
    sns.lineplot(data=df_plot, x='timestamp', y='Test', label='Test', ax=ax)
    # sns.lineplot(data=df_plot, x='timestamp', y='Predicted', label='Predicted', ax=ax)
    ax.axvline(x=df_y_train['timestamp'].iloc[-1], color='gray', linestyle='--', label='Train/Test Split')
    ax.set_xlabel("Timestamp")
    ax.set_title("Actual vs Predicted Time Series")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.show()