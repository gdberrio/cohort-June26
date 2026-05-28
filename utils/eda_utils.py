"""
Exploratory Data Analysis utility functions for Marketing Mix Modeling.

Refactored from ccf_plot.py, correlation_heatmap_code.py, and
Dualaxis_line_chart_updated_axis.py into clean, notebook-ready functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.stattools import ccf
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_workshop_data(path, sheet_name="Data"):
    """Load the MMM workshop Excel dataset.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.
    sheet_name : str
        Sheet name to read.

    Returns
    -------
    pd.DataFrame
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Ensure numeric columns
    for col in df.columns:
        if col not in ("Month", "Date", "date", "month"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(0, inplace=True)
    return df


def load_config(path):
    """Load and parse the MMM config CSV file.

    Returns
    -------
    dict with keys: dependent_var, time_column, paid_media_spends,
    competition_spend_vars, untransformed_vars, tv_vars, traditional_vars,
    atl_vars, digital_vars.
    """
    config = pd.read_csv(path)

    def _extract(col):
        vals = config[col].dropna()
        if vals.empty:
            return []
        return [v.strip() for v in ",".join(vals.astype(str)).split(",") if v.strip()]

    result = {
        "dependent_var": config.loc[0, "dependent_var"],
        "time_column": config.loc[0, "time_column"],
        "paid_media_spends": _extract("paid_media_spends"),
        "competition_spend_vars": _extract("competition_spend_vars"),
        "untransformed_vars": _extract("untransformed_vars"),
        "tv_vars": _extract("tv_vars"),
        "traditional_vars": _extract("traditional_vars"),
        "atl_vars": _extract("atl_vars"),
    }

    # Derive digital vars = paid_media_spends - tv - traditional - atl
    non_digital = set(
        result["tv_vars"] + result["traditional_vars"] + result["atl_vars"]
    )
    result["digital_vars"] = [
        v for v in result["paid_media_spends"] if v not in non_digital
    ]

    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summary_statistics(df, time_col="Month"):
    """Produce summary statistics for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        Column to exclude from numeric summary.

    Returns
    -------
    pd.DataFrame with descriptive statistics.
    """
    numeric_cols = [
        c
        for c in df.columns
        if c != time_col and df[c].dtype in ("float64", "int64", "float32")
    ]
    return df[numeric_cols].describe().T.round(2)


# ---------------------------------------------------------------------------
# CCF (Cross-Correlation Function) plots
# ---------------------------------------------------------------------------


def ccf_plot(y, x, y_name="KPI", x_name="Variable", max_lag=4, ax=None):
    """Plot the cross-correlation function between two series.

    Parameters
    ----------
    y : array-like or pd.DataFrame
        Reference series (e.g., Sales). If a DataFrame is passed, *x* is
        treated as the KPI column name and *y_name* as the variable column
        name (convenience shortcut used in notebooks).
    x : array-like or str
        Series to compare, or KPI column name when *y* is a DataFrame.
    y_name : str
        Label for y series (or variable column name when *y* is a DataFrame).
    x_name : str
        Label for x series (unused when *y* is a DataFrame).
    max_lag : int
        Maximum lag in each direction.
    ax : matplotlib Axes or None
        If None, creates a new figure.

    Returns
    -------
    matplotlib Figure (or None if ax was provided).
    """
    # Support convenience call: ccf_plot(df, kpi_col, var_col)
    if isinstance(y, pd.DataFrame):
        df = y
        kpi_col = x
        var_col = y_name
        y_arr = np.asarray(df[kpi_col], dtype=float)
        x_arr = np.asarray(df[var_col], dtype=float)
        y_name = kpi_col
        x_name = var_col
    else:
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x, dtype=float)

    backwards = ccf(y_arr[::-1], x_arr[::-1], adjusted=False)[::-1]
    forwards = ccf(y_arr, x_arr, adjusted=False)
    ccf_vals = np.r_[backwards[:-1], forwards]

    mid = len(ccf_vals) // 2
    start = mid - max_lag
    end = mid + max_lag + 1
    ccf_lagged = ccf_vals[start:end]
    lags = np.arange(-max_lag, max_lag + 1)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True

    ax.stem(lags, ccf_lagged, linefmt="k", markerfmt="", basefmt="w")
    n = len(forwards)
    ax.axhline(-1.96 / np.sqrt(n), color="b", ls="--", alpha=0.7, label="95% CI")
    ax.axhline(1.96 / np.sqrt(n), color="b", ls="--", alpha=0.7)
    ax.axhline(0, color="r", ls="-", alpha=0.5)
    ax.set_title(f"{y_name} vs {x_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lag", fontsize=11, fontweight="bold")
    ax.set_ylabel("CCF", fontsize=11, fontweight="bold")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def ccf_plot_all(
    df,
    kpi_col="Sales_Volume_Total",
    time_col="Month",
    max_lag=4,
    cols_per_row=3,
    figsize_per_plot=(4, 3),
):
    """Generate a grid of CCF plots for all variables vs the KPI.

    Parameters
    ----------
    df : pd.DataFrame
    kpi_col : str
        Name of the KPI (dependent variable) column.
    time_col : str
        Column to skip.
    max_lag : int
    cols_per_row : int
    figsize_per_plot : tuple

    Returns
    -------
    matplotlib Figure.
    """
    var_cols = [c for c in df.columns if c not in (kpi_col, time_col)]
    n_vars = len(var_cols)
    n_rows = int(np.ceil(n_vars / cols_per_row))

    fig, axes = plt.subplots(
        n_rows,
        cols_per_row,
        figsize=(figsize_per_plot[0] * cols_per_row, figsize_per_plot[1] * n_rows),
    )
    axes = np.atleast_2d(axes)

    for idx, col in enumerate(var_cols):
        r, c = divmod(idx, cols_per_row)
        ccf_plot(
            df[kpi_col],
            df[col],
            y_name=kpi_col,
            x_name=col,
            max_lag=max_lag,
            ax=axes[r, c],
        )

    # Hide unused subplots
    for idx in range(n_vars, n_rows * cols_per_row):
        r, c = divmod(idx, cols_per_row)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Cross-Correlation with {kpi_col}", fontsize=16, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def correlation_heatmap(df, cols=None, figsize=(14, 10), annot=True, cmap="coolwarm_r"):
    """Create a correlation heatmap for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str or None
        Columns to include; if None, uses all numeric columns.
    figsize : tuple
    annot : bool
    cmap : str

    Returns
    -------
    matplotlib Figure.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dual-axis line charts
# ---------------------------------------------------------------------------


def dual_axis_chart(
    df, kpi_col, var_col=None, time_col="Month", ax=None, spend_col=None
):
    """Create a dual-axis line chart comparing KPI with another variable over time.

    Parameters
    ----------
    df : pd.DataFrame
    kpi_col : str
        KPI column (left y-axis, green).
    var_col : str
        Comparison column (right y-axis, blue).
    time_col : str
        Time/x-axis column.
    ax : matplotlib Axes or None
    spend_col : str or None
        Alias for *var_col* (for notebook convenience).

    Returns
    -------
    matplotlib Figure (or None if ax was provided).
    """
    # Accept spend_col as alias for var_col
    if var_col is None:
        var_col = spend_col
    if var_col is None:
        raise ValueError("Must provide var_col (or spend_col)")
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    kpi_label = kpi_col.replace("_", " ")
    var_label = var_col.replace("_", " ")

    ax.plot(df[time_col], df[kpi_col], color="g", linewidth=1.5, label=kpi_label)
    ax.set_ylabel(kpi_label, color="g", fontsize=12)
    ax.set_ylim(0, df[kpi_col].max() * 1.1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(df[time_col], df[var_col], color="b", linewidth=1.5, label=var_label)
    ax2.set_ylabel(var_label, color="b", fontsize=12)
    ax2.set_ylim(0, df[var_col].max() * 1.1)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.set_title(f"{kpi_label} vs {var_label}", fontsize=14, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def dual_axis_chart_all(
    df,
    kpi_col,
    time_col="Month",
    cols_per_row=2,
    figsize_per_plot=(6, 4),
    spend_cols=None,
):
    """Generate a grid of dual-axis charts for all variables vs the KPI.

    Parameters
    ----------
    df : pd.DataFrame
    kpi_col : str
    time_col : str
    cols_per_row : int
    figsize_per_plot : tuple
    spend_cols : list of str or None
        If provided, only plot these variables. Otherwise plot all
        numeric columns except kpi_col and time_col.

    Returns
    -------
    matplotlib Figure.
    """
    if spend_cols is not None:
        var_cols = spend_cols
    else:
        var_cols = [c for c in df.columns if c not in (kpi_col, time_col)]
    n_vars = len(var_cols)
    n_rows = int(np.ceil(n_vars / cols_per_row))

    fig, axes = plt.subplots(
        n_rows,
        cols_per_row,
        figsize=(figsize_per_plot[0] * cols_per_row, figsize_per_plot[1] * n_rows),
    )
    axes = np.atleast_2d(axes)

    for idx, col in enumerate(var_cols):
        r, c = divmod(idx, cols_per_row)
        dual_axis_chart(df, kpi_col, col, time_col, ax=axes[r, c])

    for idx in range(n_vars, n_rows * cols_per_row):
        r, c = divmod(idx, cols_per_row)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Variables vs {kpi_col.replace('_', ' ')}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig
