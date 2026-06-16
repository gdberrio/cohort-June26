# Markov Chain Attribution

**Estimated time: 35-45 minutes**

Markov chain attribution treats a customer journey as a sequence of states. The
states are usually channels, plus a `Start` state and two absorbing outcomes:
`Conversion` and `Null` or `Drop`. Instead of assigning credit by a fixed rule,
the model estimates how customers move through the journey graph and asks how
the probability of conversion changes when a channel is removed.

This is the most common "sequence-aware" attribution method because it is
simple enough to explain and flexible enough to capture channel order.

## What The Model Learns

Start with journeys like:

```text
Start -> Display -> Paid Search -> Conversion
Start -> Social Media -> Email -> Null
Start -> Organic Search -> Paid Search -> Conversion
```

The model estimates transition probabilities:

```text
P(next state | current state)
```

For example:

| From | To | Probability |
|---|---:|---:|
| Start | Display | 0.25 |
| Start | Paid Search | 0.30 |
| Display | Paid Search | 0.40 |
| Paid Search | Conversion | 0.35 |
| Paid Search | Null | 0.65 |

The full matrix lets us calculate the probability that a journey beginning at
`Start` is eventually absorbed into `Conversion`.

## Absorbing Markov Chain View

Split the transition matrix into:

- transient states: `Start` and the channels
- absorbing states: `Conversion` and `Null`

Then:

```text
Q = transitions among transient states
R = transitions from transient states to absorbing states
N = (I - Q)^-1
B = N R
```

The row in `B` for `Start` gives the probability of ending in each absorbing
state. The first absorbing column is usually `Conversion`.

This makes Markov attribution attractive for teaching because the mechanics are
transparent. The model is not a black box. It is a graph, a transition matrix,
and an explicit removal-effect calculation.

## Removal Effect

The removal effect asks:

```text
How much lower is total conversion probability when this channel is removed
from the journey graph?
```

A common workflow is:

1. Estimate the baseline transition matrix from all journeys.
2. Compute baseline conversion probability from `Start`.
3. Remove one channel from the matrix.
4. Compute the new conversion probability.
5. Attribute credit in proportion to the conversion probability lost.

Example:

| Scenario | Conversion probability |
|---|---:|
| Full journey graph | 0.160 |
| Without Display | 0.145 |
| Without Paid Search | 0.095 |
| Without Email | 0.128 |

Removal effects:

| Channel | Effect |
|---|---:|
| Display | 0.015 |
| Paid Search | 0.065 |
| Email | 0.032 |

Normalize those effects to sum to 100 percent and you have Markov attribution
shares.

## Removal Policy Matters

"Remove a channel" sounds precise, but analysts need to define the policy.
Two common options:

| Policy | Meaning | Typical effect |
|---|---|---|
| Hard removal | If the path would enter the removed channel, route it to `Null` | Larger effects, useful when channel is truly unavailable |
| Bypass removal | Delete the channel from each path and reconnect the surrounding touches | Smaller effects, useful when the channel is unobserved or replaced by the next touch |

Neither policy is automatically correct. The right choice depends on the
business question:

- "What if we stopped buying this channel?" often calls for hard removal.
- "What if this touchpoint was not recorded?" may call for bypass removal.
- "What if spend shifted to another channel?" requires a separate budget or
  experiment model.

## First-Order vs Higher-Order Chains

A first-order Markov chain assumes:

```text
P(next touch | current touch)
```

A second-order chain assumes:

```text
P(next touch | previous two touches)
```

Higher-order chains can capture richer sequences, such as:

```text
Display -> Social -> Paid Search
```

being different from:

```text
Email -> Social -> Paid Search
```

The cost is sparsity. If you have six channels, a first-order model estimates
roughly `6 x 6` channel transitions. A second-order model estimates roughly
`6 x 6 x 6`. Sparse transitions become unstable quickly.

## Practical Checks

Before trusting Markov attribution, inspect:

- path volume by channel and by transition
- conversion and null absorption rates
- channels with very few outgoing transitions
- whether direct, organic, and brand search are acting as intent proxies
- sensitivity to hard removal vs bypass removal
- sensitivity to first-order vs second-order state definitions
- whether results agree with experiments, MMM, or holdout tests

## What It Does Not Solve

Markov attribution is still observational. If retargeting is mostly shown to
people who were already likely to buy, a Markov chain can over-credit
retargeting because it appears near conversion. The model learns the journey
graph it sees. It does not randomize exposure.

Use Markov attribution for tactical journey diagnosis. Use experiments and MMM
for incrementality and budget-level causal claims.

## References

- Anderl, Becker, von Wangenheim, and Schumann, "Mapping the Customer Journey:
  Lessons Learned from Graph-Based Online Attribution Modeling,"
  *International Journal of Research in Marketing*, 2016.
  https://doi.org/10.1016/j.ijresmar.2016.03.001
- Shao and Li, "Data-Driven Multi-Touch Attribution Models," KDD Workshop on
  Data Mining for Online Advertising, 2011.
  https://doi.org/10.1145/2020408.2020453
- Google, "About data-driven attribution."
  https://support.google.com/google-ads/answer/6394265
