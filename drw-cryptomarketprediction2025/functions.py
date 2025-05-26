"""
preprocess_train
evaluate_model
plot_actual_vs_pred

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def preprocess_train(train, columns_to_drop=[]):
    df = train.copy()
    
    #### Preprocessing
    # Identify once at the start
    cols_inf = get_cols_inf(df)
    print("Columns with infinite values:", cols_inf)
    cols_nan = get_nan_columns(df)
    print("Columns with NaN values:", cols_nan)
    cols_zerostd = get_cols_zerostd(df)
    print("Columns with zero standard deviation:", cols_zerostd)
    # Drop all at once
    cols_to_drop = set(cols_inf) | set(cols_nan) | set(cols_zerostd)

    df = df.drop(columns=cols_to_drop)

    #### Feature Engineering

    df.loc[:, 'bidask_ratio'] = df['bid_qty'] / df['ask_qty']
    df.loc[:, 'buysell_ratio'] = np.where(df['volume'] == 0, 0, df['buy_qty'] / df['sell_qty'])

    # df.loc[:, 'buysell_ratio_shift1'] = df['buysell_ratio'].shift(-1)

    df.loc[:, 'bidask_delta'] = df['bid_qty'] - df['ask_qty']
    df.loc[:, 'buysell_delta'] = df['buy_qty'] - df['sell_qty']

    df.loc[:, 'buysell_size'] = df['buy_qty'] + df['sell_qty']
    df.loc[:, 'bidask_size'] = df['bid_qty'] + df['ask_qty']

    # Final Drop
    df = df.drop(columns=columns_to_drop)
    return df


from scipy.stats import pearsonr
def evaluate_model(y_pred, y_test):
    """
    Returns the Pearson correlation coefficient between predictions and true values.
    """
    corr, _ = pearsonr(y_pred, y_test)
    return corr


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