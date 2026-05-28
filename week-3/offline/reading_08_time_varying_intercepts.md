# Reading 9: Time-Varying Intercepts in Bayesian Models

An additional resource for understanding time-varying intercepts before seeing
how Google Meridian implements them with knots.

This reading is not specific to Meridian. The goal is to build the statistical
intuition: what a time-varying intercept is, why a Bayesian model might need
one, how it differs from trend or seasonality features, and what can go wrong
when it is too flexible or too rigid.

After this reading, compare two framework implementations:
`[reading_07_meridian_knots.md](reading_07_meridian_knots.md)` for Google
Meridian and
`[reading_09_pymc_marketing_time_effects.md](reading_09_pymc_marketing_time_effects.md)`
for PyMC-Marketing.

---

## 1. Start With a Regular Intercept

In a simple regression, the intercept is the expected outcome when all
predictors are zero.

```text
y_t = alpha + beta * x_t + error_t
```

Here:

- `y_t` is the outcome at time `t`
- `x_t` is a predictor
- `beta` is the effect of `x_t`
- `alpha` is the intercept

The intercept is constant. It is the same in every time period.

In many business datasets, this is too simple. The baseline level of the KPI
often moves over time even before we account for marketing, price, promotions,
or other measured variables.

Examples:

- a brand grows gradually as awareness accumulates
- demand rises every December
- conversion rates fall during a site migration
- a product becomes less popular over time
- the whole market changes after a macro shock

A single intercept cannot represent those baseline shifts.

---

## 2. What Is a Time-Varying Intercept?

A time-varying intercept replaces one fixed intercept with an intercept that
can change over time.

Instead of:

```text
y_t = alpha + beta * x_t + error_t
```

we write:

```text
y_t = alpha_t + beta * x_t + error_t
```

Now `alpha_t` is the intercept at time `t`.

In plain English:

> The model has a different baseline level for each time period, before adding
> the effects of predictors.

This is useful when the outcome has time structure that is not fully explained
by the observed predictors.

In a marketing model, `alpha_t` can represent the baseline demand level in each
week, month, or day. It is the part of the KPI that shifts over time after the
model accounts for the variables we included.

---

## 3. Why Use One in a Bayesian Model?

Bayesian models are useful here because we can put structure on the time
variation. We do not have to choose between:

- one fixed intercept for all time
- a completely unrelated intercept for every time period

Instead, we can tell the model that nearby time periods should usually have
similar intercepts, while still allowing the baseline to move.

That structure comes from priors.

For example:

```text
alpha_t ~ Normal(alpha_{t-1}, sigma_alpha)
```

This says:

> This week's baseline is probably close to last week's baseline, but it can
> move by some amount.

The parameter `sigma_alpha` controls how much the baseline can move:

- small `sigma_alpha`: smooth baseline
- large `sigma_alpha`: flexible baseline

This is the Bayesian advantage: the model can estimate a time-varying pattern
while regularizing it so it does not chase every wiggle in the data.

---

## 4. A Useful Mental Model

Think of the model as splitting the outcome into pieces:

```text
observed outcome =
    time-varying baseline
  + predictor effects
  + noise
```

The time-varying intercept answers:

> What does the model think the outcome would be doing over time before adding
> the measured predictors?

That does not mean the time-varying intercept is "truth." It is a statistical
component. It captures whatever shared time movement is not otherwise explained
by the model.

That can include real baseline demand. It can also include omitted variables.

This is why time-varying intercepts are powerful and dangerous.

---

## 5. Common Ways to Model Time-Varying Intercepts

There are several common approaches. They all solve the same problem in
different ways: allow the baseline to move over time without making the model
uncontrolled.

### Independent Intercepts by Time

The most flexible version is:

```text
alpha_t ~ Normal(0, sigma_alpha)
```

Each time period gets its own intercept, and the intercepts are independent
given the prior.

This is very flexible but often too flexible. It can absorb real predictor
effects, especially if predictors also move mainly over time.

Use with caution.

### Random Walk

A random walk says each intercept is close to the previous one:

```text
alpha_t ~ Normal(alpha_{t-1}, sigma_alpha)
```

This is useful for gradual baseline drift. It assumes the baseline evolves
smoothly but can wander over time.

Good for:

- brand growth
- gradual demand shifts
- slow changes in conversion rate
- smooth macro movement

Risk:

- it can drift too much if weakly regularized
- it may struggle with sharp one-off events unless those are modeled separately

### Gaussian Process

A Gaussian process is a flexible prior over functions:

```text
alpha(t) ~ GP(mean_function, covariance_function)
```

Instead of specifying each intercept separately, the model learns a smooth
function over time. The covariance function controls how similar nearby time
points should be.

Good for:

- smooth nonlinear trends
- flexible seasonality
- cases where you want a principled smoothness assumption

Risk:

- can be computationally expensive
- requires careful prior choices
- can be harder to explain to stakeholders

### Splines

A spline models the baseline as a smooth curve controlled by a smaller number
of support points or basis functions.

Conceptually:

```text
alpha_t = weighted combination of spline coefficients
```

Good for:

- smooth baseline curves
- long time series
- interpretable flexibility through the number of knots or basis functions

Risk:

- too many knots can overfit
- too few knots can miss important time structure
- knot placement can matter

### easonal Components

Sometimes the intercept is decomposed into trend plus seasonality:

```text
alpha_t = trend_t + seasonality_t
```

Seasonality might be represented with month effects, week-of-year effects, or
Fourier terms.

Good for:

- repeatable seasonal patterns
- retail, travel, subscriptions, and other seasonal businesses
- models where you want a clear trend-versus-seasonality story

Risk:S

- may be too rigid for irregular events
- can still be confounded with marketing if campaigns follow the same seasonal
pattern

---

## 6. Time-Varying Intercepts Versus Other Model Terms

Time-varying intercepts are often confused with other components.

### Not the Same as a Trend Variable

A trend variable is an observed or constructed predictor, such as:

```text
week_index = 1, 2, 3, ..., T
```

Its effect is estimated through a coefficient:

```text
y_t = alpha + beta_trend * week_index_t + error_t
```

A time-varying intercept is more flexible:

```text
y_t = alpha_t + error_t
```

A trend variable says baseline moves in a specific shape. A time-varying
intercept lets the shape be learned.

### Not the Same as Seasonality

Seasonality is a repeating pattern. A time-varying intercept may capture
seasonality, but it can also capture non-repeating changes.

If the seasonal pattern is known and repeatable, explicit seasonal features may
be easier to interpret.

### Not the Same as Time-Varying Coefficients

A time-varying intercept changes the baseline.

A time-varying coefficient changes the effect of a predictor.

Compare:

```text
y_t = alpha_t + beta * x_t + error_t
```

with:

```text
y_t = alpha + beta_t * x_t + error_t
```

The first says the baseline changes over time. The second says the effect of
`x` changes over time.

This distinction matters a lot in MMM. A time-varying baseline does not mean
media ROI changes over time.

### Not the Same as a Fixed Effect for Every Date

A fixed effect for every date is a non-Bayesian way to give every time period
its own intercept.

Bayesian time-varying intercepts usually add regularization or smoothness
through priors. That regularization is what keeps the model from treating every
date as completely unrelated.

---

## 7. Why They Matter in Marketing Measurement

Marketing data is usually observational. Media spend is not randomly assigned
over time. Teams spend more when demand is expected to be higher, when launches
happen, when competitors move, or when seasonal peaks are coming.

That creates a problem:

```text
media is high during high-demand periods
```

If the model does not account for time-varying demand, it may over-credit media.

A time-varying intercept can help by absorbing baseline demand movement that is
shared across the time series.

But the opposite problem can also happen:

```text
time-varying intercept is too flexible
```

Then the baseline can absorb true media signal, and media may look weaker than
it really is.

The model is balancing two risks:

- **Under-control for time**: media gets credit for trend, seasonality, or
omitted demand shocks.
- **Over-control for time**: the baseline absorbs variation that should help
identify media effects.

This is why time-varying intercepts are not just a technical choice. They are
part of the causal design of the model.

---

## 8. Choosing the Right Amount of Flexibility

There is no universal answer. The right choice depends on the data and the
business question.

### Use More Flexibility When

- you have many observations per time period
- you have panel data with multiple geos, stores, products, or customer groups
- there are strong unmeasured time shocks
- you need to protect media estimates from obvious seasonal confounding
- residuals show time structure after fitting a simpler model

### Use Less Flexibility When

- you have one observation per time period
- media and time are highly correlated
- predictors vary mostly over time and not across units
- you have a short time series
- the baseline starts matching the observed KPI too closely
- media effects become unstable across reasonable model choices

### Prefer Explicit Predictors When

Use a measured predictor instead of asking the intercept to absorb the pattern
when you know the driver.

Examples:

- price
- promotion flags
- holidays
- product launches
- distribution changes
- macro indicators
- competitor activity
- weather
- site outages

Explicit predictors make the causal story clearer. A time-varying intercept is
useful for unobserved or hard-to-measure time movement, but it does not explain
why the movement occurred.

---

## 9. Diagnostics and Questions to Ask

After fitting a model with a time-varying intercept, inspect:

- the estimated baseline over time
- posterior uncertainty around the baseline
- residual autocorrelation
- posterior predictive checks
- whether known events line up with baseline movements
- whether predictor effects are stable under simpler or more flexible baselines
- whether the baseline is doing work that should belong to explicit predictors

Ask:

- Does the baseline shape make business sense?
- Is it too smooth to capture obvious time movement?
- Is it too wiggly and chasing noise?
- Are media or treatment effects stable when we change the baseline structure?
- Are we controlling away part of the effect we wanted to estimate?
- Are we leaving omitted time confounding inside the media coefficients?

The best model is not the one with the prettiest baseline. It is the one whose
baseline, predictors, and uncertainty support a defensible interpretation.

---

## 10. A Tiny PyMC-Style Sketch

This is not meant to be production code. It shows the shape of the idea.

```python
import pymc as pm

with pm.Model() as model:
    # A global baseline level.
    alpha_0 = pm.Normal("alpha_0", mu=0, sigma=1)

    # How much the baseline can move from one period to the next.
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.2)

    # A smooth-ish baseline path over time.
    alpha_time = pm.GaussianRandomWalk(
        "alpha_time",
        sigma=sigma_alpha,
        shape=n_time_periods,
    )

    # Predictor effect.
    beta = pm.Normal("beta", mu=0, sigma=1)

    # Expected outcome.
    mu = alpha_0 + alpha_time[time_index] + beta * x

    # Observation noise.
    sigma = pm.HalfNormal("sigma", sigma=1)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

The important part is not the exact PyMC syntax. The important part is the
structure:

```text
outcome = baseline that moves over time + predictor effects + noise
```

The prior on `alpha_time` determines how freely the baseline can move.

---

## 11. Common Mistakes

### Mistake 1: Treating the Baseline as a Causal Explanation

The time-varying intercept captures residual time movement. It does not explain
why that movement happened.

If the baseline rises in November, the model has not discovered "November
causes sales." It has estimated a time pattern.

### Mistake 2: Forgetting That Baseline and Predictors Compete

If a predictor moves at the same time as the baseline, the model has to decide
how much variation belongs to each. Priors and structure matter.

### Mistake 3: Making the Intercept Too Flexible

A very flexible baseline can improve fit while weakening causal identification.
In MMM, this can make media look artificially small.

### Mistake 4: Making the Intercept Too Rigid

A rigid baseline can leave trend or seasonality inside media coefficients. In
MMM, this can make media look artificially strong.

### Mistake 5: Confusing Baseline Change With ROI Change

A moving baseline is not a moving media ROI. To model changing ROI, you need
time-varying media coefficients or another structure that lets media effects
change over time.

---

## 12. How This Connects to Meridian Knots

Meridian's knots are one way to build a time-varying intercept.

In general Bayesian terms:

```text
time-varying intercept = baseline level that can change over time
```

In Meridian terms:

```text
mu_t = shared time-varying intercept
```

The `knots` setting controls how flexible that `mu_t` path can be. Fewer knots
make it smoother and more constrained. More knots make it more flexible.

So when you read the Meridian knots material, keep this larger idea in mind:

> Knots are not the concept. Knots are one implementation of the concept.

The concept is the time-varying intercept: a baseline component that lets the
model separate time movement from predictor effects.

---

## 13. Analyst Checklist

Before using a time-varying intercept in an applied model, answer:

- What time movement do we expect in the outcome?
- Which parts of that movement are measured by explicit predictors?
- Which parts are unmeasured and may need a flexible baseline?
- How many observations support each time period?
- Do predictors vary within time, or only over time?
- How flexible is the baseline allowed to be?
- Are predictor effects stable if we change that flexibility?
- Could the time-varying intercept absorb the effect we care about?
- Could a rigid intercept leave confounding inside the effect we care about?
- How will we explain the baseline to a stakeholder?

The purpose of a time-varying intercept is not to make the model fancy. It is
to make the baseline honest enough that the predictor effects are easier to
interpret.