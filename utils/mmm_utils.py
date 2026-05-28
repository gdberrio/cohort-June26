"""
Marketing Mix Modeling utility functions.

Refactored from Workshop Modelling Code.py and adstock_shiny_app.py
for use in Jupyter notebooks throughout the bootcamp.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, weibull_min
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import re
import math


# ---------------------------------------------------------------------------
# Adstock transformations
# ---------------------------------------------------------------------------

def adstock_geometric(x, theta):
    """Apply geometric (carry-over) adstock transformation.

    Parameters
    ----------
    x : array-like
        Input media series (spend, GRP, impressions, etc.).
    theta : float
        Decay rate in [0, 1]. Higher = slower decay.

    Returns
    -------
    dict with keys:
        x           – original input
        x_decayed   – adstocked series
        thetaVecCum – cumulative decay vector
        inflation_total – ratio of total adstocked to total original
    """
    if isinstance(x, pd.Series):
        x = x.values.astype(float)
    x = np.asarray(x, dtype=float)

    if not np.isscalar(theta):
        raise ValueError("theta must be a single value")

    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for i in range(1, len(x)):
        x_decayed[i] = x[i] + theta * x_decayed[i - 1]

    theta_vec_cum = np.zeros_like(x)
    theta_vec_cum[0] = theta
    for t in range(1, len(x)):
        theta_vec_cum[t] = theta_vec_cum[t - 1] * theta

    inflation_total = np.sum(x_decayed) / max(np.sum(x), 1e-10)

    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": theta_vec_cum,
        "inflation_total": inflation_total,
    }


def normalize(x):
    """Min-max normalize an array to [0, 1]."""
    rng = np.ptp(x)
    if rng == 0:
        return np.concatenate(([1], np.zeros(len(x) - 1)))
    return (x - np.min(x)) / rng


def adstock_weibull(x, shape, scale, windlen=None, adstock_type="pdf"):
    """Apply Weibull adstock transformation (CDF or PDF variant).

    Parameters
    ----------
    x : array-like
        Input media series.
    shape : float
        Weibull shape parameter.
    scale : float
        Weibull scale parameter (quantile-based).
    windlen : int or None
        Window length (defaults to len(x)).
    adstock_type : str
        'pdf' or 'cdf'.

    Returns
    -------
    dict with keys: x, x_decayed, thetaVecCum, inflation_total
    """
    if isinstance(x, pd.Series):
        x_vals = x.values.astype(float)
        x_index = x.index
    else:
        x_vals = np.asarray(x, dtype=float)
        x_index = range(len(x_vals))

    windlen = len(x_vals) if windlen is None else windlen
    x_bin = np.arange(1, windlen + 1)
    scale_trans = int(np.quantile(x_bin, scale))

    if shape == 0 or scale == 0:
        x_decayed = x_vals.copy()
        theta_vec_cum = np.zeros(windlen)
    else:
        if adstock_type == "cdf":
            theta_vec = np.concatenate(
                [[1], 1 - weibull_min.cdf(x_bin[:-1], shape, scale=scale_trans)]
            )
            theta_vec_cum = np.cumprod(theta_vec)
        else:  # pdf
            theta_vec_cum = normalize(
                weibull_min.pdf(x_bin, shape, scale=scale_trans)
            )

        x_decayed = np.zeros(windlen, dtype=float)
        for i in range(len(x_vals)):
            x_vec = np.concatenate((np.zeros(i), np.full(windlen - i, x_vals[i])))
            theta_lag = np.concatenate((np.zeros(i), theta_vec_cum[: windlen - i]))
            x_decayed += (x_vec[:windlen] * theta_lag[:windlen])

    inflation_total = np.sum(x_decayed) / max(np.sum(x_vals), 1e-10)

    return {
        "x": x_vals,
        "x_decayed": x_decayed,
        "thetaVecCum": theta_vec_cum,
        "inflation_total": inflation_total,
    }


# ---------------------------------------------------------------------------
# Saturation transformations
# ---------------------------------------------------------------------------

def saturation_hill(x, alpha, gamma):
    """Hill (S-curve) saturation transformation.

    Parameters
    ----------
    x : array-like
        Input series (typically adstocked).
    alpha : float
        Shape parameter controlling curve steepness.
    gamma : float
        Parameter in [0, 1] controlling inflection point position
        between min(x) and max(x).

    Returns
    -------
    np.ndarray – saturated series in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    inflexion = np.min(x) * (1 - gamma) + np.max(x) * gamma
    return x**alpha / (x**alpha + inflexion**alpha)


def saturation_power(x, n):
    """Power saturation transformation: x^n where 0 < n < 1 gives diminishing returns.

    Parameters
    ----------
    x : array-like
        Input series.
    n : float
        Power exponent (typically 0 < n < 1).

    Returns
    -------
    np.ndarray – transformed series.
    """
    x = np.asarray(x, dtype=float)
    return x**n


# ---------------------------------------------------------------------------
# Grid-search transformation pipeline
# ---------------------------------------------------------------------------

def geometric_hill_transform(x, varname, theta_range, alpha_range, gamma_range):
    """Apply grid search over adstock (theta) and saturation (alpha, gamma) params.

    Parameters
    ----------
    x : array-like
        Raw media series.
    varname : str
        Variable name for labeling columns.
    theta_range : array-like of float
        Decay rates to try.
    alpha_range : array-like of float
        Hill alpha values to try.
    gamma_range : array-like of float
        Hill gamma values to try.

    Returns
    -------
    pd.DataFrame – columns named '{varname}_{theta}_{alpha}_{gamma}'
    """
    if isinstance(x, pd.Series):
        x_arr = x.values.astype(float)
    else:
        x_arr = np.asarray(x, dtype=float)

    transformed = pd.DataFrame({varname: x_arr})

    for theta in theta_range:
        adstocked = adstock_geometric(x_arr, theta)["x_decayed"]
        for alpha in alpha_range:
            for gamma in gamma_range:
                col = f"{varname}_{theta}_{alpha}_{gamma}"
                transformed[col] = saturation_hill(adstocked, alpha, gamma)

    return transformed


# Default parameter ranges per channel type
PARAM_RANGES = {
    "tv": {
        "theta": np.round(np.linspace(0.3, 0.8, num=6), 1),
        "alpha": np.round(np.arange(0.5, 3.1, 0.1), 1),
        "gamma": np.round(np.arange(0.3, 1.1, 0.1), 1),
    },
    "digital": {
        "theta": np.round(np.arange(0.0, 0.4, 0.1), 1),
        "alpha": np.round(np.arange(0.5, 3.1, 0.1), 1),
        "gamma": np.round(np.arange(0.3, 1.1, 0.1), 1),
    },
    "traditional": {
        "theta": np.round(np.arange(0.1, 0.5, 0.1), 1),
        "alpha": np.round(np.arange(0.5, 3.1, 0.1), 1),
        "gamma": np.round(np.arange(0.3, 1.1, 0.1), 1),
    },
    "organic": {
        "theta": np.round(np.arange(0.1, 0.5, 0.1), 1),
        "alpha": np.round(np.arange(0.5, 3.1, 0.1), 1),
        "gamma": np.round(np.arange(0.3, 1.1, 0.1), 1),
    },
    "atl": {
        "theta": np.round(np.arange(0.1, 0.9, 0.1), 1),
        "alpha": np.round(np.arange(0.5, 3.1, 0.1), 1),
        "gamma": np.round(np.arange(0.3, 1.1, 0.1), 1),
    },
}


def best_transformation(transformed_df, y, varname, ascending=False):
    """Select the transformed column with the highest (or lowest) correlation to y.

    Parameters
    ----------
    transformed_df : pd.DataFrame
        Output of geometric_hill_transform.
    y : array-like
        Dependent variable.
    varname : str
        Original variable name (excluded from correlation search).
    ascending : bool
        If True, pick the most negative correlation (e.g., competition vars).

    Returns
    -------
    tuple (best_col_name, correlation, p_value, pd.DataFrame of all correlations)
    """
    cors = []
    for col in transformed_df.columns:
        if col == varname:
            continue
        r, p = pearsonr(transformed_df[col], y)
        cors.append({"column": col, "correlation": r, "p_value": p})

    cor_df = pd.DataFrame(cors)
    cor_df = cor_df.sort_values("correlation", ascending=ascending)
    best = cor_df.iloc[0]
    return best["column"], best["correlation"], best["p_value"], cor_df


# ---------------------------------------------------------------------------
# OLS model building
# ---------------------------------------------------------------------------

def build_ols_model(y, X, add_constant=True):
    """Fit OLS regression and return model + diagnostics.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    add_constant : bool
        Whether to add an intercept.

    Returns
    -------
    statsmodels OLS results object.
    """
    if add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def model_diagnostics(model, y):
    """Compute a diagnostics summary for a fitted OLS model.

    Returns
    -------
    pd.DataFrame with one row per variable containing: coefficient, std error,
    t-value, p-value, VIF, CI_lower, CI_upper, plus model-level R-sq, adj R-sq,
    MAPE, NRMSE, AIC, Durbin-Watson.
    """
    y_pred = model.fittedvalues
    y_arr = np.asarray(y)
    residuals = y_arr - y_pred

    results = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "StdError": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "VIF": [
            variance_inflation_factor(model.model.exog, i)
            for i in range(model.model.exog.shape[1])
        ],
        "CI_lower": model.conf_int().iloc[:, 0].values,
        "CI_upper": model.conf_int().iloc[:, 1].values,
        "R_squared": model.rsquared,
        "Adj_R_squared": model.rsquared_adj,
        "MAPE": np.mean(np.abs(residuals / y_arr)),
        "NRMSE": np.sqrt(np.mean(residuals**2)) / (y_arr.max() - y_arr.min()),
        "AIC": model.aic,
        "Durbin_Watson": durbin_watson(model.resid),
    })
    return results


# ---------------------------------------------------------------------------
# Contribution analysis
# ---------------------------------------------------------------------------

def _clean_varname(text):
    """Remove numeric suffixes like _0.8_0.5_0.3 from transformed variable names."""
    cleaned = re.sub(r"_\d+\.\d+", "", text)
    cleaned = re.sub(r"_+$", "", cleaned)
    return cleaned


def compute_contributions(model):
    """Compute Beta * mean(X) contribution decomposition.

    Parameters
    ----------
    model : statsmodels OLS results object.

    Returns
    -------
    pd.DataFrame with columns: Variable, Coefficient, Average, Beta_x_Mean,
    Abs_Beta_x_Mean, Contribution_pct, Contribution_signed, Variable_clean.
    """
    coef = model.params
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    avg = exog.mean()

    df = pd.DataFrame({
        "Variable": exog.columns,
        "Coefficient": coef.values,
        "Average": avg.values,
        "Beta_x_Mean": (coef * avg).values,
        "Abs_Beta_x_Mean": np.abs((coef * avg).values),
    })

    total_abs = df["Abs_Beta_x_Mean"].sum()
    total_signed = df["Beta_x_Mean"].sum()

    df["Contribution_pct"] = df["Abs_Beta_x_Mean"] / total_abs * 100
    df["Contribution_signed_pct"] = df["Beta_x_Mean"] / total_signed * 100
    df["Sign"] = np.sign(df["Coefficient"])
    df["Contribution_with_sign"] = df["Contribution_pct"] * df["Sign"]
    df["Variable_clean"] = df["Variable"].apply(_clean_varname)

    return df


def compute_decomp_rssd(media_spend_vars, contrib_df, data):
    """Compute DECOMP.RSSD (root sum of squared differences between spend share
    and effect share). Lower = better alignment.

    Parameters
    ----------
    media_spend_vars : list of str
        Column names of original spend variables in *data*.
    contrib_df : pd.DataFrame
        Output of compute_contributions().
    data : pd.DataFrame
        Original dataset containing raw spend columns.

    Returns
    -------
    float – DECOMP.RSSD value.
    """
    spend_totals = data[media_spend_vars].sum()
    spend_share = spend_totals / spend_totals.sum()

    effect = (
        contrib_df[contrib_df["Variable_clean"].isin(media_spend_vars)]
        .set_index("Variable_clean")["Beta_x_Mean"]
    )
    effect_share = effect / effect.sum()

    merged = pd.DataFrame({
        "Spend_Share": spend_share,
        "Effect_Share": effect_share,
    }).fillna(0)

    return np.sqrt(((merged["Effect_Share"] - merged["Spend_Share"]) ** 2).sum())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_contribution_plot(contrib_df, col="Contribution_with_sign",
                             title="Contribution Chart", figsize=None):
    """Horizontal bar chart of variable contributions.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Output of compute_contributions().
    col : str
        Column to plot.
    title : str
        Chart title.
    figsize : tuple or None
        Figure size; auto-calculated if None.

    Returns
    -------
    matplotlib Figure.
    """
    df_sorted = contrib_df.sort_values(col, ascending=True)
    n = len(df_sorted)

    if figsize is None:
        w = 12 + 2 * math.ceil(n / 3)
        h = max(4, 2 * math.ceil(n / 3))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df_sorted["Variable"], df_sorted[col], color="#002060", height=0.9)

    for idx, (val, var) in enumerate(
        zip(df_sorted[col], df_sorted["Variable"])
    ):
        ha = "left" if val > 0 else "right"
        offset = 0.1 if val > 0 else -0.1
        ax.text(val + offset, idx, f"{val:.2f}", va="center", ha=ha,
                fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Contribution (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.grid(False)
    fig.tight_layout()
    return fig
