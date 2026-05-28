"""
Synthetic Data Generator for Marketing Science Bootcamp
========================================================

Generates two datasets used across the bootcamp curriculum:

1. Multi-Geo Marketing Dataset  -- weekly spend/revenue across 5 US regions
   (designed for Meridian-style Marketing Mix Modeling exercises).

2. Synthetic Customer Journey Dataset  -- 10,000 multi-touch journeys
   (designed for attribution modeling exercises: Shapley, Markov, heuristic).

Usage:
    python data/generate_synthetic_data.py

Outputs (written to the same directory as this script):
    - synthetic_multi_geo_data.csv
    - synthetic_journeys.csv

Author:  Marketing Science Bootcamp -- Cohort June-26
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_adstock(series: np.ndarray, theta: float) -> np.ndarray:
    """Apply geometric (Koyck) adstock transformation to a 1-D spend array.

    Parameters
    ----------
    series : np.ndarray
        Raw weekly spend values (length T).
    theta : float
        Decay / carry-over rate in [0, 1].
        - 0  = no carry-over (immediate effect only)
        - 1  = full carry-over (effect never decays)

    Returns
    -------
    np.ndarray
        Adstocked series of the same length.
    """
    adstocked = np.zeros_like(series, dtype=float)
    adstocked[0] = series[0]
    for t in range(1, len(series)):
        adstocked[t] = series[t] + theta * adstocked[t - 1]
    return adstocked


def diminishing_returns(x: np.ndarray, half_saturation: float = 1.0) -> np.ndarray:
    """Hill-function saturation curve (diminishing returns).

    Parameters
    ----------
    x : np.ndarray
        Adstocked spend (or any non-negative input).
    half_saturation : float
        The input level at which the output reaches 50 % of its maximum.

    Returns
    -------
    np.ndarray
        Saturated values in [0, 1).
    """
    return x / (x + half_saturation)


# ---------------------------------------------------------------------------
# Dataset 1 -- Multi-Geo Marketing Data
# ---------------------------------------------------------------------------

def generate_multi_geo_data(seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic multi-geo weekly marketing dataset.

    The dataset mimics what a brand might feed into Google Meridian or a
    similar Bayesian MMM framework: weekly revenue alongside media spend
    across five US geographic regions over two years (104 weeks).

    Revenue is driven by:
        - A geo-specific base level
        - Seasonal patterns (sinusoidal + Q4 holiday bump)
        - Media effects with channel-specific adstock decay and saturation
          **applied to impressions** (not spend), mirroring the real causal
          chain: Spend → Impressions → Consumer Response → Revenue.
        - A price index (negative effect on revenue)
        - Competitor spend (negative effect on revenue)
        - Random noise

    Each channel also gets a time-varying CPM (cost-per-mille) so that the
    relationship between spend and impressions is realistic but not constant.

    Parameters
    ----------
    seed : int
        Numpy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 520 rows (5 geos x 104 weeks) and columns:
        date, geo, revenue, {ch}_spend, {ch}_impressions, {ch}_cpm,
        price_index, competitor_spend  (for each media channel).
    """
    rng = np.random.RandomState(seed)

    geos = ["northeast", "southeast", "midwest", "west", "southwest"]
    n_weeks = 104
    start_date = pd.Timestamp("2022-01-03")  # first Monday of 2022
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

    # Geo-level base revenue (weekly, in thousands of dollars)
    geo_base = {
        "northeast": 850,
        "southeast": 720,
        "midwest": 600,
        "west": 900,
        "southwest": 550,
    }

    # Geo-level spend scaling -- larger geos get proportionally more spend
    geo_spend_scale = {
        "northeast": 1.10,
        "southeast": 0.95,
        "midwest": 0.80,
        "west": 1.20,
        "southwest": 0.70,
    }

    # Adstock decay rates per channel
    adstock_theta = {
        "tv": 0.70,
        "digital": 0.10,
        "social": 0.30,
        "search": 0.15,
        "display": 0.25,
    }

    # Half-saturation points (in units of *impressions* after adstock)
    half_sat = {
        "tv": 500_000,
        "digital": 300_000,
        "social": 200_000,
        "search": 250_000,
        "display": 150_000,
    }

    # Base CPM per channel ($ per 1,000 impressions) — will vary over time
    base_cpm = {
        "tv": 25.0,     # TV is expensive (GRP-equivalent)
        "digital": 8.0,
        "social": 6.0,
        "search": 15.0,  # search clicks are pricier
        "display": 4.0,
    }

    # Revenue coefficients for saturated media (per geo-week)
    media_coef = {
        "tv": 0.30,
        "digital": 0.20,
        "social": 0.15,
        "search": 0.25,
        "display": 0.10,
    }

    week_idx = np.arange(n_weeks)

    rows = []
    for geo in geos:
        scale = geo_spend_scale[geo]

        # ---- Spend generation (realistic patterns) ----
        # TV: flighted -- runs ~60 % of weeks, heavier in Q4
        tv_on = rng.binomial(1, 0.60, size=n_weeks).astype(float)
        # Boost probability in Q4 (weeks 44-52 of each year)
        q4_mask = (week_idx % 52 >= 44)
        tv_on[q4_mask] = rng.binomial(1, 0.90, size=q4_mask.sum()).astype(float)
        tv_spend = tv_on * rng.uniform(80, 250, size=n_weeks) * scale

        # Digital: always-on with weekly variation
        digital_spend = rng.uniform(40, 150, size=n_weeks) * scale
        # Slight Q4 uplift
        digital_spend[q4_mask] *= rng.uniform(1.1, 1.4, size=q4_mask.sum())

        # Social: moderate frequency, some zero weeks
        social_on = rng.binomial(1, 0.75, size=n_weeks).astype(float)
        social_spend = social_on * rng.uniform(20, 100, size=n_weeks) * scale

        # Search: always-on, correlated with demand (seasonal)
        search_base = rng.uniform(50, 130, size=n_weeks) * scale
        seasonal_mult = 1 + 0.15 * np.sin(2 * np.pi * week_idx / 52)
        search_spend = search_base * seasonal_mult

        # Display: runs ~50 % of weeks
        display_on = rng.binomial(1, 0.50, size=n_weeks).astype(float)
        display_spend = display_on * rng.uniform(15, 70, size=n_weeks) * scale

        # ---- External factors ----
        # Price index: centered around 100 with slow drift
        price_index = 100 + np.cumsum(rng.normal(0, 0.5, size=n_weeks))
        price_index = np.clip(price_index, 85, 115)

        # Competitor spend: moderate noise, slight upward trend
        competitor_spend = (
            rng.uniform(100, 300, size=n_weeks)
            + np.linspace(0, 30, n_weeks)
            + 20 * np.sin(2 * np.pi * week_idx / 26)
        ) * scale

        # ---- Spend → Impressions via time-varying CPM ----
        spends = {
            "tv": tv_spend,
            "digital": digital_spend,
            "social": social_spend,
            "search": search_spend,
            "display": display_spend,
        }

        cpms = {}
        impressions = {}
        for ch, spend_arr in spends.items():
            # CPM varies with seasonality (Q4 inflation) + random walk
            cpm_seasonal = 1 + 0.25 * np.sin(2 * np.pi * week_idx / 52) \
                           + 0.15 * (week_idx % 52 >= 44).astype(float)
            cpm_noise = np.exp(np.cumsum(rng.normal(0, 0.03, size=n_weeks)))
            cpm_noise = cpm_noise / cpm_noise[0]  # start at 1.0
            cpms[ch] = base_cpm[ch] * cpm_seasonal * cpm_noise
            # Impressions = Spend / (CPM / 1000)
            impressions[ch] = np.where(
                spend_arr > 0,
                spend_arr / (cpms[ch] / 1000),
                0.0,
            )

        # ---- Adstock + saturation on IMPRESSIONS (not spend) ----
        adstocked = {}
        saturated = {}
        for ch in spends:
            adstocked[ch] = apply_adstock(impressions[ch], adstock_theta[ch])
            saturated[ch] = diminishing_returns(adstocked[ch], half_sat[ch])

        # ---- Revenue assembly ----
        base = geo_base[geo]
        seasonal = (
            1
            + 0.15 * np.sin(2 * np.pi * week_idx / 52)
            + 0.10 * (week_idx % 52 > 44).astype(float)
        )

        media_effect = sum(
            media_coef[ch] * saturated[ch] * base  # scale media effect to base
            for ch in spends
        )

        noise = rng.normal(0, base * 0.04, size=n_weeks)

        revenue = (
            base * seasonal
            + media_effect
            - 0.5 * (price_index - 100)  # deviation from baseline price
            - 0.1 * (competitor_spend - competitor_spend.mean())
            + noise
        )
        # Floor at zero (shouldn't happen with realistic params, but safety)
        revenue = np.maximum(revenue, 0)

        for w in range(n_weeks):
            row = {
                "date": dates[w].strftime("%Y-%m-%d"),
                "geo": geo,
                "revenue": round(float(revenue[w]), 2),
            }
            for ch in spends:
                row[f"{ch}_spend"] = round(float(spends[ch][w]), 2)
                row[f"{ch}_impressions"] = round(float(impressions[ch][w]), 0)
                row[f"{ch}_cpm"] = round(float(cpms[ch][w]), 2)
            row["price_index"] = round(float(price_index[w]), 2)
            row["competitor_spend"] = round(float(competitor_spend[w]), 2)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Dataset 2 -- Synthetic Customer Journey Data
# ---------------------------------------------------------------------------

def generate_journey_data(n_journeys: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic multi-touch customer journey data.

    Each row represents one customer journey with a sequence of marketing
    touchpoints, whether the journey ended in a conversion, and (if
    converted) the revenue generated.

    Realistic behavioural patterns are embedded:
        - Organic Search is a common first touch.
        - Paid Search and Direct are common last touches.
        - Email tends to appear mid-journey.
        - More touchpoints generally increase conversion probability.
        - Channel mix influences conversion likelihood.
        - Revenue follows a log-normal distribution (~$50 mean).

    Parameters
    ----------
    n_journeys : int
        Number of customer journeys to simulate.
    seed : int
        Numpy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: journey_id, touchpoint_sequence,
        num_touchpoints, converted, revenue.
    """
    rng = np.random.RandomState(seed)

    channels = [
        "Paid Search",
        "Display",
        "Social Media",
        "Email",
        "Organic Search",
        "Direct",
    ]

    # --- Position-dependent channel probabilities ---
    # Probabilities for first, middle, and last position (will be normalised)
    first_weights = np.array([0.18, 0.12, 0.15, 0.05, 0.35, 0.15])
    mid_weights   = np.array([0.15, 0.20, 0.22, 0.25, 0.10, 0.08])
    last_weights  = np.array([0.30, 0.05, 0.08, 0.07, 0.15, 0.35])

    first_probs = first_weights / first_weights.sum()
    mid_probs   = mid_weights / mid_weights.sum()
    last_probs  = last_weights / last_weights.sum()

    # Channel-level conversion "lift" (additive contribution to logit)
    channel_conv_lift = {
        "Paid Search":    0.25,
        "Display":       -0.10,
        "Social Media":   0.05,
        "Email":          0.30,
        "Organic Search": 0.15,
        "Direct":         0.40,
    }

    rows = []
    for jid in range(1, n_journeys + 1):
        # Number of touchpoints: geometric-ish, mostly 1-6, tail up to ~10
        n_tp = int(np.clip(rng.geometric(p=0.35), 1, 12))

        # Build touchpoint sequence
        if n_tp == 1:
            # Single touchpoint -- blend of first and last weights
            blend = (first_weights + last_weights)
            blend /= blend.sum()
            seq = [channels[rng.choice(len(channels), p=blend)]]
        elif n_tp == 2:
            first_ch = channels[rng.choice(len(channels), p=first_probs)]
            last_ch  = channels[rng.choice(len(channels), p=last_probs)]
            seq = [first_ch, last_ch]
        else:
            first_ch = channels[rng.choice(len(channels), p=first_probs)]
            last_ch  = channels[rng.choice(len(channels), p=last_probs)]
            mid_chs  = [
                channels[rng.choice(len(channels), p=mid_probs)]
                for _ in range(n_tp - 2)
            ]
            seq = [first_ch] + mid_chs + [last_ch]

        # --- Conversion probability ---
        # Base logit (tuned so average conversion ~ 12 %)
        logit = -2.5

        # More touchpoints -> higher conversion (diminishing)
        logit += 0.25 * np.log1p(n_tp)

        # Channel mix contribution
        unique_channels_in_journey = set(seq)
        for ch in unique_channels_in_journey:
            logit += channel_conv_lift[ch]

        # Diversity bonus: seeing more distinct channels helps
        logit += 0.08 * (len(unique_channels_in_journey) - 1)

        # Small random noise per journey
        logit += rng.normal(0, 0.15)

        conv_prob = 1 / (1 + np.exp(-logit))
        converted = int(rng.binomial(1, conv_prob))

        # Revenue (log-normal, mean ~ $50 for converted customers)
        if converted:
            # lognormal params: mu=3.6, sigma=0.7 -> mean ~ exp(3.6+0.7^2/2) ~ 48.4
            rev = round(float(rng.lognormal(mean=3.6, sigma=0.7)), 2)
        else:
            rev = 0.0

        rows.append({
            "journey_id": jid,
            "touchpoint_sequence": ";".join(seq),
            "num_touchpoints": n_tp,
            "converted": converted,
            "revenue": rev,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, name: str) -> None:
    """Print descriptive summary statistics for a generated dataset."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    if "geo" in df.columns:
        # Multi-geo dataset summaries
        print("--- Revenue by geo ---")
        rev_by_geo = df.groupby("geo")["revenue"].agg(["mean", "std", "min", "max"])
        print(rev_by_geo.round(2).to_string())

        print("\n--- Spend summary (mean per week across all geos) ---")
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
        print(df[spend_cols].mean().round(2).to_string())

        print(f"\n--- Date range ---")
        print(f"  First date: {df['date'].min()}")
        print(f"  Last date:  {df['date'].max()}")

    if "converted" in df.columns:
        # Journey dataset summaries
        n = len(df)
        n_conv = df["converted"].sum()
        print(f"Total journeys:      {n}")
        print(f"Conversions:         {n_conv}  ({100*n_conv/n:.1f} %)")
        print(f"Avg touchpoints:     {df['num_touchpoints'].mean():.2f}")
        print(f"Median touchpoints:  {df['num_touchpoints'].median():.0f}")
        print(f"Max touchpoints:     {df['num_touchpoints'].max()}")
        print(f"\nAvg revenue (converted): ${df.loc[df['converted']==1, 'revenue'].mean():.2f}")
        print(f"Median revenue (conv.):  ${df.loc[df['converted']==1, 'revenue'].median():.2f}")

        print("\n--- Conversion rate by number of touchpoints ---")
        conv_by_tp = (
            df.groupby("num_touchpoints")["converted"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_journeys", "mean": "conv_rate"})
        )
        print(conv_by_tp.round(3).to_string())

        # Most common first / last touches
        df_temp = df.copy()
        df_temp["first_touch"] = df_temp["touchpoint_sequence"].str.split(";").str[0]
        df_temp["last_touch"]  = df_temp["touchpoint_sequence"].str.split(";").str[-1]
        print("\n--- Top first-touch channels ---")
        print(df_temp["first_touch"].value_counts().head(6).to_string())
        print("\n--- Top last-touch channels ---")
        print(df_temp["last_touch"].value_counts().head(6).to_string())

    print()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Dataset 1: Multi-Geo Marketing Data ----
    geo_df = generate_multi_geo_data(seed=42)
    geo_path = os.path.join(script_dir, "synthetic_multi_geo_data.csv")
    geo_df.to_csv(geo_path, index=False)
    print(f"Saved multi-geo dataset to: {geo_path}")
    print_summary(geo_df, "Multi-Geo Marketing Dataset (Meridian Practice)")

    # ---- Dataset 2: Synthetic Customer Journeys ----
    journey_df = generate_journey_data(n_journeys=10_000, seed=42)
    journey_path = os.path.join(script_dir, "synthetic_journeys.csv")
    journey_df.to_csv(journey_path, index=False)
    print(f"Saved journey dataset to: {journey_path}")
    print_summary(journey_df, "Synthetic Customer Journey Data (Attribution Exercises)")

    print("Done -- both datasets generated successfully.")
