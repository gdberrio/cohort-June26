# Reading 8: Understanding Knots in Google Meridian

A practical guide to Meridian's `knots` parameter, time-varying intercepts,
and the choices analysts make when modeling trend and seasonality.

For the broader Bayesian modeling concept, read
`[reading_08_time_varying_intercepts.md](reading_08_time_varying_intercepts.md)`
first. This reading focuses on how Meridian implements that idea.
For a comparison with PyMC-Marketing's approach, see
`[reading_09_pymc_marketing_time_effects.md](reading_09_pymc_marketing_time_effects.md)`.

Knots are one of the easiest Meridian settings to skip past, because they look
like a technical spline detail. They are not just a technical detail. In an MMM,
the way we model time can change how much outcome is attributed to media versus
baseline demand, seasonality, promotions, macro factors, and other slow-moving
forces.

**Sources:**
[Set knots](https://developers.google.com/meridian/docs/advanced-modeling/setting-knots) |
[Configure the model](https://developers.google.com/meridian/docs/user-guide/configure-model) |
[Model specification](https://developers.google.com/meridian/docs/advanced-modeling/model-spec) |
[Input data scaling](https://developers.google.com/meridian/docs/advanced-modeling/input-data) |
[Knots API](https://developers.google.com/meridian/reference/api/meridian/model/knots/get_knot_info) |
[FAQs](https://developers.google.com/meridian/docs/faqs)

---

## 1. The Short Version

In Meridian, `knots` control how flexible the model's shared time pattern is.
That time pattern is called the **time-varying intercept**.

You can think of it as Meridian asking:

> After accounting for media, controls, geo differences, and noise, what common
> time pattern is still left in the KPI?

That common time pattern might include:

- broad trend
- seasonality
- holidays that are not explicitly controlled
- category demand cycles
- macro shocks
- product or business changes shared across all geos
- any omitted time-varying factor that affects the KPI

The `knots` setting determines how much freedom Meridian gives that time
pattern.

- **More knots**: more flexible time pattern, lower bias, higher variance.
- **Fewer knots**: smoother time pattern, higher bias, lower variance.
- **One knot**: effectively one common intercept across all time periods.
- **A knot at every time point**: each modeled time period can have its own
time effect.

This is a bias-variance decision. It is also a causal decision, because the
time pattern can compete with media variables for explaining the KPI.

---

## 2. Where Time-Varying Intercepts Sit in the Model

A simplified Meridian model can be written as:

```text
KPI_geo,time =
    shared_time_effect_time
  + geo_intercept_geo
  + media_effects_geo,time
  + control_effects_geo,time
  + error_geo,time
```

In Meridian notation, the shared time effect is usually written as:

```text
mu_t
```

The model includes a `mu_t` value for each modeled time period. These `mu_t`
parameters are the **time-varying intercepts**.

Why "intercept"? Because they shift the expected KPI up or down before media
and control effects are added.

Why "time-varying"? Because the shift can be different in week 1, week 2,
week 3, and so on.

Why "shared"? Because Meridian's knots model a time effect that is common
across geos. If all regions tend to be high in December and low in January,
the time-varying intercept can capture that shared movement.

This matters for interpretation:

- `mu_t` is not a media channel.
- `mu_t` is not a causal treatment.
- `mu_t` is not a time-varying ROI.
- `mu_t` is a shared time adjustment that helps define the baseline.

If `mu_t` absorbs too much movement, media effects may look too small. If it
absorbs too little movement, media may get credit for trend or seasonality that
does not belong to media.

---

## 3. What a Knot Is

Meridian does not always estimate a separate free parameter for every `mu_t`.
Instead, it can estimate a smaller number of underlying values called
`knot_values`, then interpolate between them.

The core relationship is:

```text
mu = W * b
```

Where:

- `mu` is the vector of time effects, one value for each modeled time period.
- `b` is the vector of `knot_values`, one value for each knot.
- `W` is a deterministic weight matrix that maps knot values into time effects.

If there are `T` modeled time periods and `K` knots:

- `mu` has length `T`.
- `b` has length `K`.
- `K` is less than or equal to `T`.

The model samples the knot values, not every possible time effect directly.
Then Meridian uses the weights to construct the `mu_t` values.

### A Small Example

Suppose time period 9 sits between two knots:

- one knot at time 6
- one knot at time 11

Time 9 is closer to 11 than to 6, so the knot at 11 gets more weight. The
resulting `mu_9` is a weighted average of the neighboring knot values.

Conceptually:

```text
mu_9 = 0.4 * knot_value_at_6 + 0.6 * knot_value_at_11
```

That is the spline intuition: the model estimates values at selected time
locations, then fills in the in-between periods smoothly.

### Knots Are Not Change Points

A knot is not the same thing as saying "the business changed on this date."
It is a support point for a smooth time function.

If you place a knot at Black Friday, that knot can help the time function bend
around Black Friday, but it does not mean the model has discovered a causal
Black Friday effect. If Black Friday is a known event and you need a direct
effect estimate, a holiday or event control may be more appropriate.

---

## 4. How the `knots` Argument Works

In `ModelSpec`, `knots` can be:

- `None`
- an integer
- a list or array of integer locations

### `knots=None`

This is the default.

For geo-level models, Meridian's default is effectively one knot per modeled
time period. This gives the shared time pattern maximum flexibility.

For national-level models, Meridian's default is one knot. This gives the
model a common intercept across time and avoids trying to estimate a separate
time effect from only one observation per time period.

Important: Meridian's documentation notes that the relevant `n_times` for
knots is the number of modeled time periods after accounting for `max_lag`.
The lag window reduces the time periods available to the likelihood.

### `knots=1`

This uses a single common coefficient for all time periods.

In plain English: there is no flexible time pattern. The model still has an
intercept, but it does not automatically bend up and down across time.

If you use `knots=1`, you are telling Meridian:

> Do not use a spline-like baseline to explain time movement. I will handle
> important time effects through controls, periodic functions, or accept that
> they are not modeled.

### `knots=10`

This asks Meridian to use 10 knots spread evenly across the modeled time
periods, including endpoints.

This is usually a good way to run a first sensitivity check because it is easy
to compare:

```python
from meridian.model import spec

model_spec = spec.ModelSpec(knots=10)
```

### `knots=[,0 13, 26, 39, 52, 78, 103]`

This manually places knots at specific zero-indexed time locations.

Meridian uses zero-based indexing:

- `0` means the first modeled time period
- `1` means the second modeled time period
- `n_times - 1` means the last modeled time period

Manual placement is useful when you want denser knots in important time
regions and sparser knots elsewhere.

For example, a retailer might place more knots around holiday demand periods
and fewer knots in calmer periods.

```python
from meridian.model import spec

holiday_sensitive_knots = [0, 20, 40, 46, 47, 48, 49, 50, 51, 70, 90, 103]
model_spec = spec.ModelSpec(knots=holiday_sensitive_knots)
```

Do this only when the placement reflects real business context. Otherwise,
an evenly spaced integer is easier to explain and audit.

---

## 5. What Knots Do in Practice

The `knots` choice changes how Meridian splits outcome movement among:

- baseline time effects
- controls
- media effects
- residual noise

That split affects the story you tell after modeling.

### With Many Knots

The baseline can follow the KPI closely.

This can be helpful when:

- you have many geos per time period
- there are strong shared time shocks
- seasonality is irregular
- you want the model to avoid attributing every seasonal rise to media

But it can be risky when:

- there are few geos
- media also follows the same time pattern
- national media or national controls move only over time
- the baseline becomes so flexible that it absorbs media signal

Many knots reduce bias in the time effect but increase variance. The model has
more freedom, and that freedom must be supported by data.

### With Few Knots

The baseline is smoother and less flexible.

This can be helpful when:

- the model is national-level
- there are few geos
- time movement is gradual
- you have strong explicit controls for holidays, price, distribution, macro,
or other demand drivers

But it can be risky when:

- important time effects are omitted
- media is planned around seasonal peaks
- the KPI has sharp demand events
- the model starts assigning baseline seasonality to media

Few knots reduce variance but increase bias. The model becomes more stable,
but it may be stably wrong if the true time pattern is more complex.

---

## 6. Geo Models Versus National Models

The most important distinction is whether you have multiple observations per
time period.

### Geo-Level Models

In a geo model, each time period has multiple geo observations. For example,
with 50 geos and 104 weeks, each week contributes 50 observations.

That makes flexible time effects more identifiable. Meridian's default for
geo-level data is therefore to start with one knot per modeled time period.

Practical guidance:

- Start with the default when you have a healthy number of geos.
- Reduce knots if the model overfits, media effects look implausible, or the
time-varying intercept is doing too much work.
- Be more conservative when the number of geos is small.
- Be careful combining full knots with many event controls; both are trying to
explain time.

### National-Level Models

In a national model, each time period has only one observation. If you gave
every week its own time effect, the model could spend nearly all its degrees of
freedom explaining time and leave little information for media effects.

Meridian's default for national data is therefore one knot.

Practical guidance:

- Start with `knots=1`.
- Add flexibility gradually.
- Prefer explicit, interpretable controls for known events.
- Use periodic functions for smooth seasonality when that assumption is
reasonable.
- Watch for unstable or unrealistic media estimates as knots increase.

National models are especially sensitive to the knots decision because time,
media, and demand often move together.

---

## 7. A Practical Workflow for Choosing Knots

There is no universal "correct" number of knots. Use a sensitivity workflow.

### Step 1: Identify the Modeling Situation

Ask:

- Is this geo-level or national-level data?
- How many geos do we have?
- How many modeled time periods remain after `max_lag`?
- Are there national-level media or control variables?
- Does media planning depend heavily on seasonality or events?
- Do we have explicit controls for major time-varying confounders?

If media is planned around time, then time can be a confounder. For example,
a travel brand may spend more before holidays because demand is also higher
then. If time is not modeled well, the media estimate may be biased.

### Step 2: Start From Meridian's Default

Use the default as a reference model:

```python
from meridian.model import model
from meridian.model import spec

model_spec = spec.ModelSpec()
```

Then record what Meridian used:

```python
mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
knot_info = mmm.knot_info

print(knot_info.n_knots)
print(knot_info.knot_locations)
```

### Step 3: Fit a Small Sensitivity Grid

For a national weekly model with about two years of data, a reasonable first
grid might be:

```python
candidate_knots = [1, 4, 8, 13, 26]
```

For a geo model with several geos, a reasonable first grid might compare the
default against smoother alternatives:

```python
candidate_knots = [None, 52, 26, 13]
```

These are not magic numbers. They are starting points that create meaningfully
different levels of flexibility. Similar values, such as 10 and 11, often give
similar results, so spread the candidates out.

### Step 4: Compare Diagnostics and Business Plausibility

For each candidate, inspect:

- posterior predictive fit
- residual time patterns
- baseline shape
- media contribution shares
- ROI and mROI ranges
- response curves
- convergence diagnostics
- whether results are stable across nearby specifications

Do not choose the model only because it fits best in-sample. A very flexible
time-varying intercept can improve fit while weakening media identification.

### Step 5: Ask the Causal Question

For each candidate, ask:

> Is the time-varying intercept controlling for confounding, or is it absorbing
> signal that should belong to media or explicit controls?

If the time pattern is a proxy for omitted variables, consider adding better
controls instead of only changing knots.

Examples:

- If COVID cases affected both demand and media planning, add a COVID-related
control if available.
- If price promotions drive both media bursts and revenue spikes, add price or
promotion controls.
- If holidays have repeatable effects, add holiday indicators or seasonal
functions.
- If distribution changed, add distribution or availability controls.

Knots can help with unobserved shared time patterns, but they are not a
substitute for thoughtful variable design.

---

## 8. Automatic Knot Selection

Meridian includes Automatic Knot Selection (AKS).

To enable it:

```python
from meridian.model import model
from meridian.model import spec

model_spec = spec.ModelSpec(enable_aks=True)
mmm = model.Meridian(input_data=input_data, model_spec=model_spec)

selected_knots = mmm.knot_info.knot_locations
print(selected_knots)
```

AKS chooses knot locations using an algorithm that balances fit and complexity.
It starts from candidate knots, removes knots that do not improve fit enough,
and uses a penalty that accounts for the number of geos. More geos can support
more time flexibility.

Important rules:

- Do not set `enable_aks=True` and `knots=...` in the same `ModelSpec`.
- Inspect the selected knots before accepting them.
- Treat AKS as a candidate generator, not as a replacement for judgment.

You can also run AKS directly, modify its penalty, and then pass the selected
locations into `ModelSpec(knots=...)`.

```python
import numpy as np
from meridian.model import knots
from meridian.model import spec

aks = knots.AKS(input_data)

knot_locations = aks.automatic_knot_selection(
    base_penalty=np.geomspace(10, 200, 100),
    min_internal_knots=2,
    max_internal_knots=10,
).knots

model_spec = spec.ModelSpec(knots=knot_locations)
```

Increasing the penalty encourages fewer knots. This can be useful when AKS
selects too many knots for a national model or a model with few geos.

---

## 9. Alternatives to Knots

Knots are one way to model time. They are not the only way.

### Meridian Knots Versus Seasonality Controls

In Meridian, knots are not "the seasonality feature." They define the
flexibility of the shared time-varying intercept.

That is different from dummy variables or Fourier terms:


| Approach               | What It Says                                                          | Best Use                                                              | Main Risk                                          |
| ---------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------- |
| Meridian knots         | Let the baseline bend over time using a shared time-varying intercept | Unknown trend, irregular shared shocks, smooth residual time movement | Can absorb media signal or omitted controls        |
| Dummy variables        | Estimate a separate effect for named periods or events                | Holidays, launches, outages, known event windows                      | Too many dummies can overfit or compete with media |
| Fourier/cyclical terms | Estimate smooth repeating seasonal waves                              | Annual or weekly seasonality that repeats predictably                 | Too rigid for one-off shocks or moving holidays    |


The key difference is the assumption being made.

With a holiday dummy, you are saying:

```text
This named event has its own effect.
```

With Fourier terms, you are saying:

```text
The seasonal pattern repeats smoothly with a known period.
```

With Meridian knots, you are saying:

```text
There is a shared baseline time pattern, and I will choose how flexible it is.
```

Those are different modeling choices. They can coexist, but they should not be
treated as interchangeable.

### Why This Matters Specifically in Meridian

Meridian's knot-based time-varying intercept is shared across geos. It is a
common time pattern in the KPI after accounting for media, controls, geo
intercepts, and noise.

This has two important consequences:

1. Knots are good at capturing time movement that affects all geos at roughly
  the same time.
2. Knots are not the best way to represent an event whose effect differs a lot
  by geo, channel, store, or customer segment.

For example:

- A national demand shock may be reasonable for the time-varying intercept.
- A national holiday might be either a dummy control or part of the time
baseline, depending on whether you need to interpret it separately.
- A regional weather event is usually better as a control than as a shared
time spline.
- Smooth annual seasonality may be more parsimonious as Fourier terms,
especially in a national model.

Another practical difference: dummy variables and Fourier terms are explicit
columns in your data. Their coefficients can be inspected as effects of named
features. Knots are not named business events. They are support points used to
construct the baseline path.

So if a stakeholder asks, "How much did Black Friday add?", a Black Friday
control is easier to interpret. If the question is, "Can the baseline flex
enough that media is not forced to explain every seasonal bend?", knots are the
more natural lever.

### How to Choose Among Them

Use this rule of thumb:

- Use **dummy variables** for named, discrete events you want to control for or
explain.
- Use **Fourier/cyclical terms** for smooth, repeating seasonality with a known
period.
- Use **Meridian knots** for residual shared time movement that is not easily
explained by named events or simple repeating cycles.

In practice, a good model often uses a combination:

```text
KPI = media effects
    + explicit controls for known events
    + cyclical terms for regular seasonality
    + time-varying intercept for remaining shared time movement
    + noise
```

Be careful with redundancy. If you include many holiday dummies, rich Fourier
terms, and a very flexible knot structure, all three are trying to explain time.
That can make media effects less stable and make the baseline hard to explain.

### Explicit Control Variables

Use explicit controls when you know the reason the KPI moves.

Examples:

- price
- promotions
- holidays
- distribution
- competitor activity
- macroeconomic indicators
- category demand proxies
- weather
- search interest or other demand signals

Explicit controls are often better than asking knots to absorb everything,
because they make the causal story more inspectable.

Use controls when:

- the variable affects both media execution and the KPI
- the timing is meaningful
- the effect is not just generic seasonality
- you need to explain the driver, not merely absorb it

### Binary Indicators

Binary indicators are useful for events that turn on and off.

Examples:

- holiday weeks
- campaign launch periods
- supply outage periods
- COVID restriction periods
- major site migration periods

In Meridian, binary indicators can be passed as controls. Unlike knots, a
binary indicator can have geo-varying effects in a geo model, because control
coefficients can vary by geo.

This is useful when an event affects regions differently. A local sports event,
regional weather shock, or geo-specific distribution issue may be better as a
control than as part of a shared time spline.

Watch the total degrees of freedom. Full knots plus many indicators can
over-parameterize time.

### Periodic Functions

Periodic functions, such as Fourier terms, model smooth repeating seasonality.

They are useful when:

- seasonality is smooth and cyclical
- you are working with national data
- you want a parsimonious seasonal structure
- you do not need every seasonal bend to be freely estimated

Example weekly seasonality features:

```python
df["sin_annual"] = np.sin(2 * np.pi * df["week_index"] / 52)
df["cos_annual"] = np.cos(2 * np.pi * df["week_index"] / 52)
```

Periodic functions impose a stronger assumption than knots: the pattern repeats
smoothly. That can be stabilizing, especially in national models, but it may be
too rigid for irregular holiday timing or one-off shocks.

### Data Partitioning for Time-Varying Media Effects

Knots do not make media coefficients time-varying.

If you believe a channel worked differently in two periods, Meridian's FAQ
describes a manual data-partitioning approach: split the channel into separate
variables for distinct periods.

For example:

```text
paid_search_2024 = paid_search spend during 2024, otherwise 0
paid_search_2025 = paid_search spend during 2025, otherwise 0
```

This lets the model estimate separate effects for each period-specific channel.
Use this sparingly. Splitting channels multiplies parameters and can create
identifiability problems.

---

## 10. Common Mistakes

### Mistake 1: Treating Knots as Media Effect Change Points

Knots affect the shared time-varying intercept. They do not directly say that
TV, paid search, or Meta became more effective after a given date.

### Mistake 2: Using Full Knots in a National Model

With one observation per time period, a separate time effect for every period
is usually too flexible. Meridian defaults to one knot for national models for
a reason.

### Mistake 3: Letting Knots Replace Controls

Knots can absorb time movement, but they do not tell you why the movement
happened. If a known driver affected both media and the KPI, model it directly
when possible.

### Mistake 4: Optimizing for In-Sample Fit Only

More knots often improve fit. Better fit does not automatically mean better
media incrementality estimates.

### Mistake 5: Forgetting Collinearity With National Variables

National-level variables change over time but not across geo. A model with a
separate parameter for every time period can be redundant or weakly identified
with these variables. In that situation, use fewer knots and stronger explicit
structure.

### Mistake 6: Assuming Knot Priors Can Vary Over Time

Meridian does not support time-varying priors for knot values. You can choose
knot locations and set priors on `knot_values`, but you cannot give different
time periods different knot-value priors through a built-in temporal prior
system.

---

## 11. Recommended Defaults for This Course

For classroom work, we want settings that are teachable and easy to defend.

### If Using National-Level Workshop Data

Start with:

```python
model_spec = spec.ModelSpec(knots=1)
```

Then try:

```python
candidate_knots = [1, 4, 8, 13]
```

Use the smallest number of knots that leaves no obvious residual time pattern
and does not produce implausible media results.

### If Using Geo-Level Synthetic Meridian Data

Start with Meridian's default or AKS:

```python
model_spec_default = spec.ModelSpec()
model_spec_aks = spec.ModelSpec(enable_aks=True)
```

Then compare smoother alternatives if needed:

```python
candidate_knots = [None, 52, 26, 13]
```

If the model has only a few geos, be more conservative than the default.

### Always Record the Decision

In your Meridian model spec artifact, write:

- the knot setting used
- whether the model is geo-level or national-level
- why that setting was chosen
- what alternatives were tested
- how results changed across alternatives
- whether time effects may still be confounded with media

An acceptable explanation:

```text
We used knots=8 for the national model after comparing 1, 4, 8, and 13 knots.
The 1-knot model left clear seasonal residual structure. The 13-knot model had
slightly better in-sample fit but produced unstable paid search ROI. The 8-knot
model reduced residual seasonality while preserving plausible media estimates.
Holiday indicators were included separately because holiday timing is a known
driver of both demand and media execution.
```

---

## 12. Analyst Checklist

Before finalizing a Meridian model, answer:

- Is the model geo-level or national-level?
- What is the effective number of modeled time periods after `max_lag`?
- What knot setting did Meridian actually use?
- Did we compare at least one more flexible and one less flexible setting?
- Does the baseline shape make business sense?
- Do residuals still show obvious time structure?
- Are media estimates stable across reasonable knot choices?
- Are known time-varying confounders modeled as controls where possible?
- Are national-level variables collinear with a highly flexible time effect?
- Did we document the choice in the model spec artifact?

If you cannot answer these, the model may still be useful for exploration, but
it is not ready for a confident budget recommendation.

---

## 13. Mental Model to Keep

Knots decide how bendy Meridian's baseline time pattern is.

Too bendy, and the baseline may eat signal that belongs elsewhere.

Too stiff, and media may inherit seasonality, trend, or omitted shocks.

The right setting is not the one that sounds most sophisticated. It is the one
that creates a credible baseline, stable media estimates, and a causal story
you can defend.