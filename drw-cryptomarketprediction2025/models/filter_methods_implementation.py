import numpy as np
import polars as pl
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, 
    SelectFpr, SelectFdr, SelectFwe, f_regression, 
    mutual_info_regression, chi2
)
import pandas as pd

def add_filter_methods_to_pipeline(pipeline_class):
    """
    Add comprehensive filter methods to the FeatureEngineeringPipeline class.
    This function extends the existing class with filter functionality.
    """
    
    def _filtermethods_6(self, lazy_df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        """
        Filter methods implementation with configurable parameters.
        Supports various univariate feature selection methods.
        """
        # Get filter configuration from self.config
        filter_config = self.__get_config("filter", {})
        method = filter_config.get("method", "none")
        
        if method == "none":
            return lazy_df, []
        
        # Collect the lazy frame for processing
        df = lazy_df.collect()
        
        # Prepare X and y for sklearn compatibility
        X = df.drop(["timestamp", self.y_col]).to_numpy()
        y = df.select(self.y_col).to_numpy().ravel()
        feature_names = df.drop(["timestamp", self.y_col]).columns
        
        # Apply filter method based on configuration
        selected_features = self._apply_filter_method(X, y, feature_names, filter_config)
        
        # Get columns to drop (all columns except timestamp, y_col, and selected features)
        all_cols = df.columns
        keep_cols = ["timestamp", self.y_col] + selected_features
        drop_cols = [col for col in all_cols if col not in keep_cols]
        
        return lazy_df.select(pl.col("timestamp"), pl.col(self.y_col), *selected_features), drop_cols
    
    def _apply_filter_method(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], config: dict) -> list[str]:
        """
        Apply filter method based on configuration using match cases.
        """
        method = config.get("method", "none")
        
        match method.lower():
            case "variance_threshold":
                threshold = config.get("threshold", 0.0)
                return self._variance_threshold_filter(X, feature_names, threshold)
            
            case "selectkbest_freg":
                k = config.get("k", 10)
                return self._select_kbest_freg_filter(X, y, feature_names, k)
            
            case "selectkbest_mutualinfo":
                k = config.get("k", 10)
                return self._select_kbest_mutualinfo_filter(X, y, feature_names, k)
            
            case "selectpercentile_freg":
                percentile = config.get("percentile", 10)
                return self._select_percentile_freg_filter(X, y, feature_names, percentile)
            
            case "selectpercentile_mutualinfo":
                percentile = config.get("percentile", 10)
                return self._select_percentile_mutualinfo_filter(X, y, feature_names, percentile)
            
            case "selectfpr_freg":
                alpha = config.get("alpha", 0.05)
                return self._select_fpr_freg_filter(X, y, feature_names, alpha)
            
            case "selectfdr_freg":
                alpha = config.get("alpha", 0.05)
                return self._select_fdr_freg_filter(X, y, feature_names, alpha)
            
            case "selectfwe_freg":
                alpha = config.get("alpha", 0.05)
                return self._select_fwe_freg_filter(X, y, feature_names, alpha)
            
            case "correlation_threshold":
                threshold = config.get("threshold", 0.95)
                return self._correlation_threshold_filter(X, feature_names, threshold)
            
            case "mutual_info_threshold":
                threshold = config.get("threshold", 0.01)
                return self._mutual_info_threshold_filter(X, y, feature_names, threshold)
            
            case "anova_f_test":
                alpha = config.get("alpha", 0.05)
                return self._anova_f_test_filter(X, y, feature_names, alpha)
            
            case "chi2_test":
                alpha = config.get("alpha", 0.05)
                return self._chi2_test_filter(X, y, feature_names, alpha)
            
            case _:
                print(f"Unknown filter method: {method}. No filtering applied.")
                return feature_names
    
    def _variance_threshold_filter(self, X: np.ndarray, feature_names: list[str], threshold: float) -> list[str]:
        """Remove features with variance below threshold."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_kbest_freg_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int) -> list[str]:
        """Select k best features using F-regression."""
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_kbest_mutualinfo_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int) -> list[str]:
        """Select k best features using mutual information."""
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_percentile_freg_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], percentile: float) -> list[str]:
        """Select top percentile features using F-regression."""
        percentile = min(percentile, 100)
        selector = SelectPercentile(score_func=f_regression, percentile=percentile)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_percentile_mutualinfo_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], percentile: float) -> list[str]:
        """Select top percentile features using mutual information."""
        percentile = min(percentile, 100)
        selector = SelectPercentile(score_func=mutual_info_regression, percentile=percentile)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_fpr_freg_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float) -> list[str]:
        """Select features using False Positive Rate test with F-regression."""
        selector = SelectFpr(score_func=f_regression, alpha=alpha)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_fdr_freg_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float) -> list[str]:
        """Select features using False Discovery Rate test with F-regression."""
        selector = SelectFdr(score_func=f_regression, alpha=alpha)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_fwe_freg_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float) -> list[str]:
        """Select features using Family-wise Error test with F-regression."""
        selector = SelectFwe(score_func=f_regression, alpha=alpha)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _correlation_threshold_filter(self, X: np.ndarray, feature_names: list[str], threshold: float) -> list[str]:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Return features that are not highly correlated
        return [name for name in feature_names if name not in to_drop]
    
    def _mutual_info_threshold_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], threshold: float) -> list[str]:
        """Select features with mutual information above threshold."""
        mi_scores = mutual_info_regression(X, y)
        selected_mask = mi_scores > threshold
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _anova_f_test_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float) -> list[str]:
        """Select features using ANOVA F-test."""
        f_scores, p_values = f_regression(X, y)
        selected_mask = p_values < alpha
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _chi2_test_filter(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float) -> list[str]:
        """Select features using Chi-squared test."""
        # Chi2 requires non-negative features
        X_non_negative = X - X.min(axis=0)
        chi2_scores, p_values = chi2(X_non_negative, y)
        selected_mask = p_values < alpha
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    # Add all methods to the pipeline class
    pipeline_class._filtermethods_6 = _filtermethods_6
    pipeline_class._apply_filter_method = _apply_filter_method
    pipeline_class._variance_threshold_filter = _variance_threshold_filter
    pipeline_class._select_kbest_freg_filter = _select_kbest_freg_filter
    pipeline_class._select_kbest_mutualinfo_filter = _select_kbest_mutualinfo_filter
    pipeline_class._select_percentile_freg_filter = _select_percentile_freg_filter
    pipeline_class._select_percentile_mutualinfo_filter = _select_percentile_mutualinfo_filter
    pipeline_class._select_fpr_freg_filter = _select_fpr_freg_filter
    pipeline_class._select_fdr_freg_filter = _select_fdr_freg_filter
    pipeline_class._select_fwe_freg_filter = _select_fwe_freg_filter
    pipeline_class._correlation_threshold_filter = _correlation_threshold_filter
    pipeline_class._mutual_info_threshold_filter = _mutual_info_threshold_filter
    pipeline_class._anova_f_test_filter = _anova_f_test_filter
    pipeline_class._chi2_test_filter = _chi2_test_filter

# Example usage configurations
EXAMPLE_CONFIGS = {
    "variance_threshold": {
        "filter": {"method": "variance_threshold", "threshold": 0.01}
    },
    "selectkbest_freg": {
        "filter": {"method": "selectkbest_freg", "k": 50}
    },
    "selectpercentile_mutualinfo": {
        "filter": {"method": "selectpercentile_mutualinfo", "percentile": 20}
    },
    "correlation_threshold": {
        "filter": {"method": "correlation_threshold", "threshold": 0.95}
    },
    "anova_f_test": {
        "filter": {"method": "anova_f_test", "alpha": 0.05}
    },
    "selectfpr_freg": {
        "filter": {"method": "selectfpr_freg", "alpha": 0.05}
    },
    "selectfdr_freg": {
        "filter": {"method": "selectfdr_freg", "alpha": 0.05}
    },
    "selectfwe_freg": {
        "filter": {"method": "selectfwe_freg", "alpha": 0.05}
    },
    "mutual_info_threshold": {
        "filter": {"method": "mutual_info_threshold", "threshold": 0.01}
    },
    "chi2_test": {
        "filter": {"method": "chi2_test", "alpha": 0.05}
    }
}

def get_available_filter_methods():
    """Return a list of available filter methods with their descriptions."""
    return {
        "variance_threshold": "Remove features with variance below threshold",
        "selectkbest_freg": "Select k best features using F-regression",
        "selectkbest_mutualinfo": "Select k best features using mutual information",
        "selectpercentile_freg": "Select top percentile features using F-regression",
        "selectpercentile_mutualinfo": "Select top percentile features using mutual information",
        "selectfpr_freg": "Select features using False Positive Rate test with F-regression",
        "selectfdr_freg": "Select features using False Discovery Rate test with F-regression",
        "selectfwe_freg": "Select features using Family-wise Error test with F-regression",
        "correlation_threshold": "Remove highly correlated features",
        "mutual_info_threshold": "Select features with mutual information above threshold",
        "anova_f_test": "Select features using ANOVA F-test",
        "chi2_test": "Select features using Chi-squared test"
    } 