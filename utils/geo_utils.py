"""
GeoLift / Geo-experiment utility functions.

Provides Python-native helpers plus an optional R bridge via rpy2
for students who have GeoLift installed.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure-Python geo-experiment helpers
# ---------------------------------------------------------------------------


def load_geo_data(path):
    """Load a GeoLift-formatted CSV (location, Y, date).

    Parameters
    ----------
    path : str or Path
        Path to CSV file.

    Returns
    -------
    pd.DataFrame with columns: location, Y, date (datetime).
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def pivot_geo_data(df, value_col="Y", location_col="location", date_col="date"):
    """Pivot geo data from long to wide format (dates as rows, locations as columns).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format geo data.

    Returns
    -------
    pd.DataFrame with date index and one column per location.
    """
    return df.pivot(index=date_col, columns=location_col, values=value_col)


def plot_geo_timeseries(df, locations=None, title="Geo Time Series", figsize=(14, 6)):
    """Plot time series for selected geo locations.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format geo data with columns: location, Y, date.
    locations : list of str or None
        Locations to plot; None plots all.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if locations is None:
        locations = df["location"].unique()

    fig, ax = plt.subplots(figsize=figsize)
    for loc in locations:
        subset = df[df["location"] == loc]
        ax.plot(subset["date"], subset["Y"], label=loc, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def compute_treatment_effect(
    pre_df,
    post_df,
    treatment_locations,
    control_locations=None,
    value_col="Y",
    location_col="location",
    date_col="date",
):
    """Simple difference-in-differences treatment effect estimate.

    Parameters
    ----------
    pre_df, post_df : pd.DataFrame
        Pre-test and post-test geo data in long format.
    treatment_locations : str or list of str
        Treatment location(s). If a single string is passed and
        *control_locations* is None, all other locations are used as controls.
    control_locations : list of str or None
        Control location(s). If None, all non-treatment locations are used.

    Returns
    -------
    dict with keys: post_treatment_avg, post_control_avg, pre_treatment_avg,
    pre_control_avg, did_estimate, lift_pct.
    """
    # Convenience: accept a single string for treatment_locations
    if isinstance(treatment_locations, str):
        treatment_locations = [treatment_locations]

    # Default: all other locations become controls
    if control_locations is None:
        all_locs = pre_df[location_col].unique()
        control_locations = [loc for loc in all_locs if loc not in treatment_locations]

    pre_treat = pre_df[pre_df[location_col].isin(treatment_locations)][value_col].mean()
    pre_ctrl = pre_df[pre_df[location_col].isin(control_locations)][value_col].mean()
    post_treat = post_df[post_df[location_col].isin(treatment_locations)][
        value_col
    ].mean()
    post_ctrl = post_df[post_df[location_col].isin(control_locations)][value_col].mean()

    did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
    counterfactual = pre_treat + (post_ctrl - pre_ctrl)
    lift_pct = did / counterfactual * 100 if counterfactual != 0 else np.nan

    return {
        "post_treatment_avg": post_treat,
        "post_control_avg": post_ctrl,
        "pre_treatment_avg": pre_treat,
        "pre_control_avg": pre_ctrl,
        "did_estimate": did,
        "lift_pct": lift_pct,
    }


# ---------------------------------------------------------------------------
# R bridge (optional – requires rpy2 + GeoLift R package)
# ---------------------------------------------------------------------------


def try_import_geolift():
    """Attempt to import GeoLift via rpy2. Returns (GeoLift_module, available_flag)."""
    try:
        from rpy2.robjects.packages import importr

        GeoLift = importr("GeoLift")
        return GeoLift, True
    except Exception:
        return None, False


def r_to_pandas(r_df):
    """Convert an R dataframe to pandas (requires rpy2)."""
    from rpy2.robjects import conversion, default_converter
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        return conversion.rpy2py(r_df)


def pandas_to_r(pd_df):
    """Convert a pandas DataFrame to R (requires rpy2)."""
    from rpy2.robjects import conversion, default_converter
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(pd_df)
