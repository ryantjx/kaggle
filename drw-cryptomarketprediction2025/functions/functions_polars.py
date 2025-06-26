import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import polars as pl
import numpy as np
from tqdm import tqdm

def get_cols_inf(df: pl.DataFrame) -> list[str]:
    """
    Returns a list of column names that contain any positive or negative infinity.
    """
    cols = []
    for col in df.columns:
        # df[col] is a Series; .is_infinite() → Boolean Series; .any() → Python bool
        try:
            if df[col].is_infinite().any():
                cols.append(col)
        except Exception:
            # if the column isn’t numeric, .is_infinite() might error—just skip it
            continue
    return cols

def get_nan_columns(df: pl.DataFrame) -> list[str]:
    """
    Returns a list of column names with any NaN/null values.
    """
    cols = []
    for col in df.columns:
        if df.select(pl.col(col).is_null().any()).item():
            cols.append(col)
    return cols

def get_cols_zerostd(df: pl.DataFrame) -> list[str]:
    """
    Returns a list of column names whose standard deviation is zero
    (or whose std returns None because all values are null).
    Non-numeric columns (e.g. datetime) are skipped.
    """
    cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        # Only attempt std() on numeric dtypes
        if dtype.is_numeric():  
            # df[col] is a Series; .std() returns a Python float or None
            std_val = df[col].std()
            if std_val == 0.0 or std_val is None:
                cols.append(col)
    return cols


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    # Feature engineering
    df = df.with_columns([
        # bidask_ratio = bid_qty / ask_qty
        (pl.col("bid_qty") / pl.col("ask_qty")).alias("bidask_ratio"),

        # buysell_ratio = 0 if volume == 0 else buy_qty / sell_qty
        pl.when(pl.col("volume") == 0)
        .then(0)
        .otherwise(pl.col("buy_qty") / pl.col("sell_qty"))
        .alias("buysell_ratio"),

        # bidask_delta = bid_qty - ask_qty
        (pl.col("bid_qty") - pl.col("ask_qty")).alias("bidask_delta"),

        # buysell_delta = buy_qty - sell_qty
        (pl.col("buy_qty") - pl.col("sell_qty")).alias("buysell_delta"),

        # buysell_size = buy_qty + sell_qty
        (pl.col("buy_qty") + pl.col("sell_qty")).alias("buysell_size"),

        # bidask_size = bid_qty + ask_qty
        (pl.col("bid_qty") + pl.col("ask_qty")).alias("bidask_size"),
    ])
    return df
def preprocess_train(train: pl.DataFrame, columns_to_drop: list[str] = []) -> pl.DataFrame:
    """
    Mirror of the original pandas workflow, but using polars.
    1. Identify columns with infinite, NaN, or zero‐std and drop them.
    2. Drop any user‐specified columns (e.g. label or order‐book columns).
    3. (You can add normalized/scaling steps here if needed.)
    """
    df = train.clone()

    df = feature_engineering(df)
    
    #### Preprocessing
    cols_inf = get_cols_inf(df)
    print("Columns with infinite values:", cols_inf)

    cols_nan = get_nan_columns(df)
    print("Columns with NaN values:", cols_nan)

    cols_zerostd = get_cols_zerostd(df)
    print("Columns with zero standard deviation:", cols_zerostd)
    # Drop columns with infinite, NaN, or zero‐std values
    drop_columns = list(set(cols_inf) | set(cols_nan) | set(cols_zerostd) | set(columns_to_drop))
    if drop_columns:
        df = df.drop(drop_columns)
    # df = df.sort("timestamp", descending=False)
    return df, drop_columns

def preprocess_test(test: pl.DataFrame, columns_to_drop: list[str] = []) -> pl.DataFrame:
    df = test.clone()
    df = feature_engineering(df)
    df = df.drop(columns_to_drop)
    print("Columns dropped from test set:", columns_to_drop)
    return df