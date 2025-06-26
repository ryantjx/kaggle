import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

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

# def regression_diagnostics_sklearn(model, X, y, alpha=0.05, verbose=True):
#     """
#     Computes extended regression diagnostics for scikit-learn linear models,
#     including standard errors, t-stats, p-values, confidence intervals, and
#     classic regression metrics.
#     """
#     n, k = X.shape
#     X_ = np.column_stack([np.ones(n), X])  # Add intercept
#     y_pred = model.predict(X)
#     residuals = y - y_pred
#     rss = np.sum(residuals ** 2)
#     tss = np.sum((y - np.mean(y)) ** 2)
#     r2 = r2_score(y, y_pred)
#     df_model = k
#     df_resid = n - k - 1
#     mse_resid = rss / df_resid

#     # Coefficient stats
#     if hasattr(model, "intercept_"):
#         coefs = np.concatenate([[model.intercept_], model.coef_])
#     else:
#         coefs = model.coef_

#     # Variance-Covariance matrix
#     try:
#         xtx_inv = np.linalg.inv(np.dot(X_.T, X_))
#         se = np.sqrt(np.diag(mse_resid * xtx_inv))
#     except np.linalg.LinAlgError:
#         se = np.full_like(coefs, np.nan)

#     t_stats = coefs / se
#     p_values = 2 * t.sf(np.abs(t_stats), df_resid)

#     # Confidence intervals
#     ci = t.ppf(1 - alpha/2, df_resid) * se
#     ci_low = coefs - ci
#     ci_high = coefs + ci

#     # Model-level metrics
#     aic = n * np.log(rss / n) + 2 * (k + 1)
#     bic = n * np.log(rss / n) + np.log(n) * (k + 1)

#     # F-statistic (overall regression significance)
#     ms_model = (tss - rss) / df_model
#     ms_resid = rss / df_resid
#     f_stat = ms_model / ms_resid
#     from scipy.stats import f
#     f_pval = 1 - f.cdf(f_stat, df_model, df_resid)

#     # Durbin-Watson
#     dw = np.sum(np.diff(residuals)**2) / rss

#     # Correlation metrics
#     pearson_val, pearson_p = pearsonr(y, y_pred)
#     spearman_val, spearman_p = spearmanr(y, y_pred)

#     # Compile results
#     results = {
#         "n_obs": n,
#         "df_model": df_model,
#         "df_resid": df_resid,
#         "aic": aic,
#         "bic": bic,
#         "f_stat": f_stat,
#         "f_pval": f_pval,
#         "r2": r2,
#         "rmse": np.sqrt(mean_squared_error(y, y_pred)),
#         "mae": mean_absolute_error(y, y_pred),
#         "medae": median_absolute_error(y, y_pred),
#         "pearson_corr": pearson_val,
#         "pearson_pvalue": pearson_p,
#         "spearman_corr": spearman_val,
#         "spearman_pvalue": spearman_p,
#         "durbin_watson": dw,
#         "coefs": coefs,
#         "std_err": se,
#         "t_stats": t_stats,
#         "p_values": p_values,
#         "ci_low": ci_low,
#         "ci_high": ci_high,
#         "residuals": residuals,
#     }

#     if verbose:
#         print(f"Observations: {n}")
#         print(f"Degrees of Freedom (Model): {df_model}")
#         print(f"Degrees of Freedom (Residuals): {df_resid}")
#         print(f"R^2: {r2:.4f}")
#         print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
#         print(f"F-statistic: {f_stat:.2f}, p-value: {f_pval:.3g}")
#         print(f"Durbin-Watson: {dw:.2f}")
#         print(f"Pearson r: {pearson_val:.4f} (p={pearson_p:.3g})")
#         print(f"Spearman rho: {spearman_val:.4f} (p={spearman_p:.3g})")
#         print(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}, Median AE: {results['medae']:.4f}")
#         print("\nCoefficients:")
#         header = ["coef", "std err", "t", "P>|t|", f"[{100*alpha/2:.1f}%", f"{100*(1-alpha/2):.1f}%]"]
#         print("{:12s} {:12s} {:12s} {:12s} {:12s} {:12s}".format(*header))
#         for idx, (coef, s, tstat, pval, l, h) in enumerate(zip(coefs, se, t_stats, p_values, ci_low, ci_high)):
#             name = "const" if idx == 0 else f"x{idx}"
#             print(f"{name:12s} {coef:12.4f} {s:12.4f} {tstat:12.3f} {pval:12.3g} {l:12.4f} {h:12.4f}")

#     return results