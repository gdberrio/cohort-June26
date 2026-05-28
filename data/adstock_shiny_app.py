# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:19:30 2025

@author: admin
"""

from shiny import App, render, ui, reactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import nest_asyncio
# from io import BytesIO
# import base64


# Geometric Adstock Function
def adstock_geometric(x, theta):
    if isinstance(x, pd.Series):
        x = x.values
    if not np.isscalar(theta):
        raise ValueError("theta must be a single value")

    if len(x) > 1:
        x_decayed = np.zeros_like(x)
        x_decayed[0] = x[0]
        for i in range(1, len(x_decayed)):
            x_decayed[i] = x[i] + theta * x_decayed[i - 1]

        theta_vec_cum = np.zeros_like(x)
        theta_vec_cum[0] = theta
        for t in range(1, len(x)):
            theta_vec_cum[t] = theta_vec_cum[t - 1] * theta
    else:
        x_decayed = x
        theta_vec_cum = np.array([theta])

    x_sum = np.sum(x)
    inflation_total = np.sum(x_decayed) / x_sum if x_sum != 0 else 1.0
    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": theta_vec_cum,
        "inflation_total": inflation_total,
    }


def normalize(x):
    if np.ptp(x) == 0:  # ptp calculates the range of an array
        return np.concatenate(([1], np.zeros(len(x) - 1)))
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


# Weibull Adstock Function
def adstock_weibull(x, shape, scale, windlen=None, adstockType="pdf"):
    if not (np.isscalar(shape) and np.isscalar(scale)):
        raise ValueError("shape and scale must be single values")

    windlen = len(x) if windlen is None else windlen
    x_bin = pd.Series(range(1, windlen + 1))
    scaleTrans = int(x_bin.quantile(scale))

    if shape == 0 or scale == 0:
        x_decayed = x
        thetaVecCum = thetaVec = pd.Series(np.zeros(windlen), index=x_bin)
        x_imme = x
        inflation_total = 1.0  # No transformation applied
    else:
        if adstockType == "cdf":
            thetaVec = pd.Series(
                [1] + list(1 - weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans)),
                index=x_bin,
            )
            thetaVecCum = thetaVec.cumprod()
        elif adstockType == "pdf":
            thetaVecCum = normalize(
                pd.Series(
                    weibull_min.pdf(x_bin, shape, scale=scaleTrans),
                    index=x_bin,
                )
            )

        x_decayed = pd.Series(0, index=x.index, dtype=float)
        for i, xi in enumerate(x):
            x_vec = np.concatenate((np.zeros(i), np.repeat(xi, windlen - i)))
            thetaVecCumLag = np.concatenate(
                (np.zeros(i), thetaVecCum[: len(thetaVecCum) - i])
            )
            x_prod = x_vec[:windlen] * thetaVecCumLag[:windlen]
            x_decayed += pd.Series(x_prod, index=x.index)

        x_imme = (
            x
            if adstockType == "cdf"
            else pd.Series(np.convolve(x, thetaVecCum)[: len(x)], index=x.index)
        )

        inflation_total = x_decayed.sum() / x.sum()

    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": thetaVecCum,
        "inflation_total": inflation_total,
        "x_imme": x_imme,
    }


# Power Transformation Function
def saturation_power(x, n, theta):
    if not (np.isscalar(n) and np.isscalar(theta)):
        raise ValueError("n and theta must be single values")
    x_saturated = np.zeros_like(x)
    x_saturated[0] = x[0] ** n
    for i in range(1, len(x)):
        x_saturated[i] = x[i] ** n + theta * x_saturated[i - 1]
    return x_saturated


# Hill Tranformation Function
def saturation_hill(x, alpha, inflexion):
    if not (np.isscalar(alpha) and np.isscalar(inflexion)):
        raise ValueError("alpha and gamma must be single values")
    return x**alpha / (x**alpha + inflexion**alpha)


# Data
spends_data = pd.Series(
    [
        0,
        24403.91,
        18494.72,
        18123.32,
        15391.17,
        11225.17,
        28180.1,
        17515.13,
        13030.23,
        0,
        0,
        115362.15,
        51133.38,
        51010.74,
        47596.98,
        15874.76,
        46225.58,
        44021.76,
        9065,
        0,
        0,
        0,
        0,
        0,
        0,
        11974.76,
        10829.9,
        18856.38,
        7594.04,
        39583.59,
        33650.23,
        35338.35,
    ]
)
time = np.arange(len(spends_data))

# UI
app_ui = ui.page_fluid(
    ui.input_dark_mode(),
    ui.tags.div(
        ui.tags.img(
            src="https://www.arymalabs.com/wp-content/uploads/2022/04/logo1-1.jpg",
            height="60px",
        ),
        style="text-align: left;margin-top: 10px",
    ),
    ui.h1(
        "Adstock Transformation in MMM",
        style="text-align: center;margin-bottom: 20px;font-weight: bold;",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "dist",
                "Select Adstock:",
                {
                    "Geometric": "Geometric",
                    "Weibull CDF": "Weibull CDF",
                    "Weibull PDF": "Weibull PDF",
                },
                selected="Geometric",
            ),
            ui.output_ui("param_sliders"),
            ui.input_select(
                "trans_type",
                "Select Method for Saturation:",
                {"Hill": "Hill Transformation", "Power": "Power Transformation"},
                selected="Hill",
            ),
            ui.output_ui("trans_params"),
            width=400,
        ),
        ui.navset_pill(
            ui.nav_panel("Decay Curve", ui.output_plot("decay_plot", height="500px")),
            ui.nav_panel(
                "Diminishing Returns Curve", ui.output_plot("sat_plot", height="500px")
            ),
        ),
    ),
)


# Server
def server(input, output, session):
    CHOICES = {
        "Geom": {"Hill": "Hill Transformation", "Power": "Power Transformation"},
        "Weib": {"Hill": "Hill Transformation"},
    }

    @reactive.effect
    def _():
        choices = "Geom" if input.dist() == "Geometric" else "Weib"
        ui.update_select("trans_type", choices=CHOICES[choices])

    @output
    @render.ui
    def param_sliders():
        if input.dist() == "Geometric":
            return ui.input_slider("theta", "Theta:", 0, 1, value=0.3, step=0.01)
        elif input.dist() == "Weibull CDF":
            return ui.div(
                ui.input_slider("shape_cdf", "Shape:", 0, 2, value=0.5, step=0.01),
                ui.input_slider("scale_cdf", "Scale:", 0, 0.1, value=0.05, step=0.01),
            )
        elif input.dist() == "Weibull PDF":
            return ui.div(
                ui.input_slider("shape_pdf", "Shape:", 0, 10, value=0.5, step=0.1),
                ui.input_slider("scale_pdf", "Scale:", 0, 0.1, value=0.05, step=0.01),
            )

    @output
    @render.ui
    def trans_params():
        if input.trans_type() == "Hill":
            return ui.div(
                ui.input_slider("alpha", "Alpha:", 0.5, 3, value=1.2, step=0.1),
                ui.input_slider("gamma", "Gamma:", 0.3, 1, value=0.5, step=0.1),
            )
        elif input.trans_type() == "Power":
            return ui.input_slider("n", "n:", 0.1, 0.9, value=0.5, step=0.1)

    @output
    @render.plot
    def decay_plot():
        # Determine decay curve based on distribution type
        if input.dist() == "Geometric":
            decay = np.append(
                1, adstock_geometric(spends_data, input.theta())["thetaVecCum"][:-1]
            )
        elif input.dist() == "Weibull CDF":
            decay = adstock_weibull(
                spends_data,
                shape=input.shape_cdf(),
                scale=input.scale_cdf(),
                adstockType="cdf",
            )["thetaVecCum"]
        elif input.dist() == "Weibull PDF":
            decay = adstock_weibull(
                spends_data,
                shape=input.shape_pdf(),
                scale=input.scale_pdf(),
                adstockType="pdf",
            )["thetaVecCum"]
        else:
            raise ValueError("Invalid distribution type")

        fig, ax = plt.subplots()
        ax.plot(time, decay, color="blue", lw=2)
        ax.set_title(f"{input.dist()} Decay Curve", fontsize=18, fontweight="bold")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Decay", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12)

        return fig

    @output
    @render.plot
    def sat_plot():
        # Determine decayed series based on distribution type
        if input.dist() == "Geometric":
            x_decayed = [1] + adstock_geometric(spends_data, input.theta())[
                "x_decayed"
            ][:-1]
        elif input.dist() == "Weibull CDF":
            x_decayed = adstock_weibull(
                spends_data,
                shape=input.shape_cdf(),
                scale=input.scale_cdf(),
                adstockType="cdf",
            )["x_decayed"]
        elif input.dist() == "Weibull PDF":
            x_decayed = adstock_weibull(
                spends_data,
                shape=input.shape_pdf(),
                scale=input.scale_pdf(),
                adstockType="pdf",
            )["x_decayed"]
        else:
            raise ValueError("Invalid distribution type")

        simulated_x = np.linspace(0, 1.2 * spends_data.max(), 100)
        if input.trans_type() == "Hill":
            gamma = input.gamma()
            alpha = input.alpha()
            inflexion = x_decayed.min() * (1 - gamma) + x_decayed.max() * gamma
            y = saturation_hill(simulated_x, alpha, inflexion)
        elif input.trans_type() == "Power":
            n = input.n()
            theta = input.theta()
            y = saturation_power(simulated_x, n, theta)

        fig, ax = plt.subplots()
        ax.plot(simulated_x, y, color="red", lw=2)
        ax.set_title(
            f"{input.trans_type()} Transformation", fontsize=18, fontweight="bold"
        )
        ax.set_xlabel("Spends", fontsize=14, fontweight="bold")
        ax.set_ylabel("Saturated Series", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12)

        return fig


# Run the app
app = App(app_ui, server)

nest_asyncio.apply()

app.run()
