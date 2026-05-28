# Reading 10: How PyMC-Marketing Handles Time Effects

A comparison of PyMC-Marketing's approach to seasonality, controls, events,
and optional time-varying parameters.

This reading answers a practical question:

> If Meridian uses knots for a shared time-varying intercept, how does
> PyMC-Marketing solve the same modeling problem?

Short answer: PyMC-Marketing usually handles known time patterns through
explicit controls and Fourier seasonality. It also supports time-varying
intercepts, but they are optional and off by default.

**Sources:**
[PyMC-Marketing MMM API](https://www.pymc-marketing.io/en/stable/api/generated/pymc_marketing.mmm.mmm.MMM.html) |
[PyMC-Marketing latest MMM examples](https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mmm.mmm.html) |
[Time-varying parameters notebook](https://www.pymc-marketing.io/en/0.11.0/notebooks/mmm/mmm_tvp_example.html) |
[Fourier seasonality API](https://www.pymc-marketing.io/en/stable/api/generated/pymc_marketing.mmm.fourier.html)

---

## 1. The Baseline PyMC-Marketing MMM

The standard PyMC-Marketing MMM is a Bayesian regression with transformed media
inputs, optional controls, and an intercept.

Conceptually:

```text
y_t =
    intercept
  + media effects
  + control effects
  + error
```

The media effects are built from adstock and saturation transformations:

```text
media effect = beta_m * saturation(adstock(media_m))
```

Known non-media drivers can be passed as `control_columns`.

Examples:

- holidays
- special events
- price
- promotions
- distribution
- macro variables
- competitor variables
- trend columns

This is the first big difference from a knot-first discussion: in
PyMC-Marketing, if you know why the KPI moved, you usually put that reason into
the design matrix as a control, event, or seasonality term.

---

## 2. How Regular Seasonality Is Usually Modeled

PyMC-Marketing has built-in Fourier seasonality support. In the core `MMM`
constructor, the `yearly_seasonality` argument controls the number of Fourier
modes to include.

Example:

```python
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM

mmm = MMM(
    date_column="date",
    channel_columns=["paid_search", "tv", "social"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    control_columns=["price_index", "promotion_flag"],
    yearly_seasonality=4,
)
```

Fourier terms model smooth repeating cycles using sine and cosine waves.

The modeling assumption is:

```text
There is a seasonal pattern that repeats with a known period.
```

For weekly data, yearly Fourier terms can capture patterns such as:

- higher demand every summer
- lower demand every January
- regular holiday-season shape, if it is smooth enough
- recurring annual demand cycles

PyMC-Marketing also exposes Fourier utilities such as `YearlyFourier`,
`MonthlyFourier`, and `WeeklyFourier` for custom model building.

### Why Fourier Terms Are Useful

Fourier terms are parsimonious. A small number of sine/cosine pairs can
represent a smooth annual pattern without giving each week its own parameter.

This helps with:

- extrapolation into future periods
- stable seasonality estimates
- fewer parameters
- easier prior specification

This is different from a highly flexible time-varying intercept, which may fit
in-sample time movement well but may not extrapolate a regular seasonal pattern
as cleanly.

---

## 3. How Known Events Are Modeled

PyMC-Marketing supports ordinary control columns and a newer events interface.

For simple event controls, you can create columns yourself:

```python
df["black_friday_week"] = (df["date"].isin(black_friday_weeks)).astype(int)
df["site_outage"] = (df["date"].between("2025-03-10", "2025-03-12")).astype(int)
```

Then include them:

```python
mmm = MMM(
    date_column="date",
    channel_columns=media_cols,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    control_columns=["black_friday_week", "site_outage"],
)
```

The latest docs also show an `add_events(...)` workflow using event start and
end dates with event-effect basis functions.

The modeling assumption is:

```text
This named event has an explicit effect on the outcome.
```

This is useful when the event has a business interpretation you want to keep
visible. For example, if stakeholders want to know whether a promotion week
lifted revenue, an explicit event term is more interpretable than asking a
generic time-varying intercept to absorb the spike.

---

## 4. Does PyMC-Marketing Use Time-Varying Intercepts?

Yes, but not by default.

The current PyMC-Marketing `MMM` API includes:

```python
time_varying_intercept=False
time_varying_media=False
```

So the default model does not use a time-varying intercept. But you can turn it
on:

```python
mmm = MMM(
    date_column="date",
    channel_columns=media_cols,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    time_varying_intercept=True,
)
```

PyMC-Marketing's documentation describes this time-varying intercept as a
Gaussian Process based component, implemented efficiently with a Hilbert Space
Gaussian Process approximation.

Conceptually:

```text
y_t =
    baseline_intercept * time_varying_multiplier_t
  + media effects
  + controls
  + error
```

The time-varying component is centered around 1 and acts as a multiplier on the
baseline intercept. If the baseline intercept is 1,000 units, the GP describes
how the baseline moves above or below that average over time.

This is different from Meridian's knot implementation.

| Framework | Time-Varying Intercept Mechanism | Default? | Main Tuning Lever |
|---|---|---:|---|
| Meridian | Spline-like shared time effect controlled by `knots` | Yes for geo models, one knot for national models | Number/location of knots or AKS |
| PyMC-Marketing | Optional GP/HSGP time-varying multiplier | No | GP length-scale, covariance, basis functions |

The concepts are related, but the implementation and defaults differ.

---

## 5. What PyMC-Marketing Recommends for Seasonality and Trend

The time-varying parameter notebook makes an important point: a GP-based
time-varying intercept can technically capture seasonality and trend, but that
does not mean it is the best tool for those patterns.

For regular seasonality, prefer Fourier terms.

For steady growth or decline, prefer an explicit trend variable.

For known events, prefer event indicators, event effects, or control variables.

For unexplained irregular temporal variation, consider a time-varying
intercept.

The logic is:

| Pattern | Better First Tool | Why |
|---|---|---|
| Smooth annual seasonality | Fourier terms | Repeating pattern with known period |
| Linear growth | Trend control | Simple, interpretable shape |
| Holiday or launch | Event/control variable | Named business event |
| Price or distribution change | Control variable | Measured driver |
| Unexplained irregular shocks | Time-varying intercept | Catch-all temporal variation |

This is a useful modeling principle:

> Use structure when you know the structure. Use flexible time variation when
> you do not.

---

## 6. PyMC-Marketing Versus Meridian

Both frameworks need to solve the same basic problem:

```text
How do we stop media from getting credit for time patterns that are not caused
by media?
```

But they encourage different workflows.

### Meridian Workflow

Meridian has a built-in time-varying intercept controlled by `knots`.

In a geo model, Meridian can support a flexible shared time effect because each
time period has multiple geo observations. The shared time effect helps absorb
common shocks across geos.

The analyst's question becomes:

```text
How flexible should the shared time-varying intercept be?
```

That is why Meridian users need to think carefully about `knots`.

### PyMC-Marketing Workflow

PyMC-Marketing starts from a more explicit regression-style specification:

```text
y_t =
    intercept
  + transformed media
  + controls
  + Fourier seasonality
  + events
  + error
```

The analyst's first question is:

```text
What time patterns can I explain with explicit variables?
```

Only after that would you add:

```python
time_varying_intercept=True
```

for residual temporal variation that the explicit variables do not capture.

---

## 7. When to Use PyMC-Marketing's Time-Varying Intercept

Use it when:

- posterior predictive checks show unexplained time structure
- residuals have irregular temporal spikes or dips
- you suspect omitted shocks but do not have measured controls
- the baseline genuinely changes over time in a non-periodic way
- Fourier terms and explicit controls are not enough

Be cautious when:

- your main issue is ordinary seasonality
- you have a simple trend that could be a trend column
- media spend is highly correlated with time
- you have a short time series
- the GP baseline starts absorbing media contribution
- out-of-sample extrapolation matters a lot

The PyMC-Marketing TVP notebook is especially clear on this point: GP-based
time-varying intercepts are powerful for unexplained events, but regular
patterns such as seasonality or constant growth are usually better modeled with
Fourier or linear terms.

---

## 8. Minimal Comparison Code

### Regular Seasonality With Fourier Terms

```python
mmm = MMM(
    date_column="date",
    channel_columns=media_cols,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    control_columns=["price_index", "promotion_flag"],
    yearly_seasonality=4,
)
```

Use this when seasonality is regular and repeating.

### Explicit Event Effects

```python
mmm = MMM(
    date_column="date",
    channel_columns=media_cols,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    control_columns=["black_friday", "supply_outage"],
)
```

Use this when the time pattern is tied to known events.

### Optional Time-Varying Intercept

```python
mmm = MMM(
    date_column="date",
    channel_columns=media_cols,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    control_columns=["price_index", "promotion_flag"],
    yearly_seasonality=4,
    time_varying_intercept=True,
)
```

Use this when explicit controls and Fourier seasonality still leave meaningful
unexplained time movement.

---

## 9. Practical Recommendation for This Course

For a PyMC-Marketing model, teach the sequence this way:

1. Start with media transformations and a fixed intercept.
2. Add measured controls for known non-media drivers.
3. Add Fourier seasonality for regular cycles.
4. Add event indicators or event effects for named shocks.
5. Check posterior predictive fit and residual time structure.
6. Add `time_varying_intercept=True` only if meaningful unexplained time
   variation remains.

This sequence makes the model easier to explain.

It also reduces the risk that a flexible baseline becomes a black box that
absorbs effects we should have modeled explicitly.

---

## 10. Analyst Checklist

Before adding a PyMC-Marketing time-varying intercept, answer:

- Have we included known event and promotion controls?
- Have we included price, distribution, or macro controls where relevant?
- Have we modeled regular annual seasonality with Fourier terms?
- Is the remaining time movement irregular rather than repeating?
- Are media effects stable before and after adding the time-varying intercept?
- Does the TVP improve posterior predictive checks without swallowing media?
- Do we understand the GP length-scale assumption?
- Is the goal in-sample explanation or future extrapolation?

If the pattern is known, name it with a control. If it repeats, model the
cycle. If it is unexplained and irregular, then a PyMC-Marketing time-varying
intercept can be the right tool.
