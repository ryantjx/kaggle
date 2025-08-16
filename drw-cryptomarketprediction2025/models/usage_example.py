"""
Usage example for integrating filter methods with FeatureEngineeringPipeline
"""

import polars as pl
import numpy as np
from sklearn.decomposition import IncrementalPCA
from filter_methods_implementation import (
    add_filter_methods_to_pipeline, 
    EXAMPLE_CONFIGS, 
    get_available_filter_methods
)

# Define the FeatureEngineeringPipeline class (copy from your notebook)
class FeatureEngineeringPipeline:
    def __init__(
        self,
        data: pl.DataFrame,
        y_col: str = "label",
        drop_columns: list[str] = None,
        config: dict = None
    ):
        self.data = data.lazy()
        self.y_col = y_col
        self.drop_columns = drop_columns or []
        self.config = config or {}

    def run_model_agnostic_selection(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        all_drop_cols = []
        stages = [
            self._data_preprocessing_1,
            self._scaling_or_norm_2,
            self._addfeaturesandcategorical_3,
            self._featuretransformandgenreation_4,
            self._unsupervisedextraction_5,
            self._filtermethods_6,
        ]
        for fn in stages:
            lazy_df, drop_list = fn(lazy_df)
            all_drop_cols.extend(drop_list)
        return lazy_df, all_drop_cols

    def _data_preprocessing_1(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        df_collected = lazy_df.collect()
        cols = df_collected.drop("timestamp").columns
        
        inf_flags = df_collected.select([pl.col(c).is_infinite().any().alias(c) for c in cols]).row(0)
        inf_cols = [c for c, flag in zip(cols, inf_flags) if flag]

        nan_flags = df_collected.select([pl.col(c).is_null().any().alias(c) for c in cols]).row(0)
        nan_cols = [c for c, flag in zip(cols, nan_flags) if flag]

        numeric_cols = [c for c, dt in zip(df_collected.columns, df_collected.dtypes) if dt.is_numeric()]
        std_flags = (
            df_collected
            .select([pl.col(c).std().alias(c) for c in numeric_cols])
            .row(0)
        )
        zerostd_cols = [c for c, std in zip(numeric_cols, std_flags) if std == 0 or std is None]

        drop_cols = inf_cols + nan_cols + zerostd_cols + self.drop_columns
        return lazy_df.drop(drop_cols), drop_cols
    
    def _scaling_or_norm_2(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        return lazy_df, []
    
    def _addfeaturesandcategorical_3(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        lazy_df = lazy_df.with_columns([
            (pl.col("bid_qty") / pl.col("ask_qty")).alias("bidask_ratio"),
            pl.when(pl.col("volume") == 0)
            .then(0)
            .otherwise(pl.col("buy_qty") / pl.col("sell_qty"))
            .alias("buysell_ratio"),
            (pl.col("bid_qty") - pl.col("ask_qty")).alias("bidask_delta"),
            (pl.col("buy_qty") - pl.col("sell_qty")).alias("buysell_delta"),
            (pl.col("buy_qty") + pl.col("sell_qty")).alias("buysell_size"),
            (pl.col("bid_qty") + pl.col("ask_qty")).alias("bidask_size"),
        ])
        drop_cols = ["bid_qty", "ask_qty", "buy_qty", "sell_qty"]
        return lazy_df.drop(drop_cols), drop_cols

    def _featuretransformandgenreation_4(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        return lazy_df, []

    def _unsupervisedextraction_5(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        df = lazy_df.collect()
        ipca: IncrementalPCA = IncrementalPCA(n_components=5)
        ipca.fit(df).components_.T
        load = pl.DataFrame(data=ipca.fit(df).components_.T, schema=df.columns)
        thresh = load.select(pl.all().abs().mean()).unpivot().select(pl.col("value")).mean().item()
        features = (
            load
            .select(pl.all().abs().max())
            .unpivot()
            .filter(pl.col("value") >= thresh)
            .select("variable")
            .to_series()
            .to_list()
        )
        all_cols = df.columns
        keep_cols = ["timestamp"] + features
        drop_cols = [col for col in all_cols if col not in keep_cols]
        return lazy_df.select(pl.col("timestamp"), *features), drop_cols
        
    def _filtermethods_6(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        # This will be replaced by the filter methods implementation
        return lazy_df, []

    def __get_config(self, key: str, default=None):
        return self.config.get(key, default)

# Add filter methods to the pipeline
add_filter_methods_to_pipeline(FeatureEngineeringPipeline)

def demonstrate_filter_methods():
    """Demonstrate different filter methods with example data"""
    
    # Create sample data (you would replace this with your actual data)
    print("Creating sample data...")
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Create sample features
    X = np.random.randn(n_samples, n_features)
    
    # Create target variable with some relationship to features
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + 
         np.random.randn(n_samples) * 0.1)
    
    # Create polars DataFrame
    feature_cols = [f"X{i}" for i in range(n_features)]
    data_dict = {"timestamp": pl.date_range(
        start=pl.datetime(2023, 1, 1), 
        end=pl.datetime(2023, 1, 1, 23, 59), 
        interval="1m"
    )[:n_samples]}
    
    for i, col in enumerate(feature_cols):
        data_dict[col] = X[:, i]
    
    data_dict["label"] = y
    sample_data = pl.DataFrame(data_dict)
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test different filter methods
    filter_configs = {
        "Variance Threshold": {"filter": {"method": "variance_threshold", "threshold": 0.01}},
        "SelectKBest F-regression": {"filter": {"method": "selectkbest_freg", "k": 20}},
        "SelectPercentile Mutual Info": {"filter": {"method": "selectpercentile_mutualinfo", "percentile": 30}},
        "Correlation Threshold": {"filter": {"method": "correlation_threshold", "threshold": 0.8}},
        "ANOVA F-test": {"filter": {"method": "anova_f_test", "alpha": 0.05}}
    }
    
    results = {}
    
    for method_name, config in filter_configs.items():
        print(f"\nTesting {method_name}...")
        
        pipeline = FeatureEngineeringPipeline(
            sample_data, y_col="label", config=config
        )
        
        try:
            # Run the pipeline
            result_df, dropped_cols = pipeline.run_model_agnostic_selection(pipeline.data)
            final_df = result_df.collect()
            
            # Count remaining features
            remaining_features = len(final_df.columns) - 2  # exclude timestamp and label
            results[method_name] = {
                "remaining_features": remaining_features,
                "dropped_features": len(dropped_cols),
                "shape": final_df.shape
            }
            
            print(f"  - Remaining features: {remaining_features}")
            print(f"  - Dropped features: {len(dropped_cols)}")
            print(f"  - Final shape: {final_df.shape}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            results[method_name] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    print("Filter Methods Demonstration")
    print("=" * 50)
    
    # Show available methods
    print("Available filter methods:")
    methods = get_available_filter_methods()
    for method, description in methods.items():
        print(f"- {method}: {description}")
    
    print("\n" + "=" * 50)
    print("Running demonstrations...")
    
    # Run demonstrations
    results = demonstrate_filter_methods()
    
    print("\n" + "=" * 50)
    print("Summary of results:")
    for method, result in results.items():
        if "error" in result:
            print(f"{method}: ERROR - {result['error']}")
        else:
            print(f"{method}: {result['remaining_features']} features remaining "
                  f"({result['dropped_features']} dropped)") 