# Probabilistic Attribution

**Estimated time: 35-45 minutes**

Probabilistic attribution asks a different question from rule-based models:

```text
Given what we observed about this journey, how likely was conversion?
```

Then attribution is derived from how that probability changes when a channel is
included, removed, masked, or perturbed. The emphasis is not only a point
estimate of credit, but uncertainty around that credit.

## Why Use A Probabilistic View?

Marketing journey data is noisy:

- Many customers have only one or two recorded touches.
- Some channels appear frequently but mostly for high-intent users.
- Some path combinations are rare.
- Conversion rates can be unstable for small segments.

A probabilistic model lets us smooth noisy rates and express uncertainty. This
is especially useful in a course setting because students can see when a channel
looks strong simply because it has little data.

## Simple Channel Lift Model

A minimal probabilistic attribution model compares:

```text
P(conversion | channel present)
P(conversion | channel absent)
```

The raw difference is:

```text
lift(channel) = P(conversion | present) - P(conversion | absent)
```

But raw conversion rates can be unstable, so we use a Beta-Binomial model:

```text
conversions_present ~ Binomial(impressions_present, p_present)
p_present ~ Beta(alpha, beta)

conversions_absent ~ Binomial(impressions_absent, p_absent)
p_absent ~ Beta(alpha, beta)
```

With a `Beta(1, 1)` prior, the posterior mean is:

```text
(conversions + 1) / (journeys + 2)
```

This gives a smoothed conversion probability even for low-volume channels.

## Journey-Level Credit

After estimating each channel's probabilistic lift, allocate credit inside a
converted journey:

```text
credit(channel in journey) =
    positive_lift(channel) / sum(positive_lift(channels in journey))
```

This is not a causal proof. It is a transparent way to convert uncertain
channel-level propensities into journey-level credit.

## From Simple Lift To Full Probabilistic Models

The simple model can be extended in several ways:

| Extension | What it adds | Tradeoff |
|---|---|---|
| Bayesian logistic regression | Controls and regularization | More modeling choices |
| Hierarchical channel priors | Better estimates for sparse channels | More complex priors |
| Time-to-conversion survival model | Delay and censoring | Requires timestamps |
| Latent state model | Unobserved intent or funnel stage | Harder to explain and validate |
| Uplift or causal response model | Incrementality target | Requires treatment/control variation |

In production, "probabilistic attribution" often means a propensity model that
predicts conversion from journey features, followed by a perturbation method to
turn predictions into credit.

## Calibration

A probabilistic attribution model should be calibrated. If the model says a
group of journeys has a 20 percent conversion probability, about 20 percent of
those journeys should convert.

Good checks:

- calibration table by predicted probability decile
- Brier score
- log loss
- out-of-time validation split
- sensitivity to priors or smoothing assumptions

## Caveats

Probabilistic attribution estimates association unless the model is trained on
experimental variation or a defensible quasi-experiment. A channel can have a
high conditional probability because it appears late in the funnel, not because
it caused the conversion.

For budget decisions, compare probabilistic attribution with:

- geo experiments or user holdouts
- MMM contribution and ROI estimates
- incrementality tests for retargeting, brand search, and email

## References

- Shao and Li, "Data-Driven Multi-Touch Attribution Models," KDD Workshop on
  Data Mining for Online Advertising, 2011.
  https://doi.org/10.1145/2020408.2020453
- Dalessandro, Stitelman, Perlich, and Provost, "Causally Motivated Attribution
  for Online Advertising," ADKDD, 2012.
  https://doi.org/10.1145/2351356.2351363
- Gelman et al., *Bayesian Data Analysis*, third edition, for Beta-Binomial
  updating and posterior predictive thinking.
  http://www.stat.columbia.edu/~gelman/book/
