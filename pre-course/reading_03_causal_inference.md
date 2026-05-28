# Reading 3: Introduction to Causal Inference for Marketers

**Estimated time: ~60 minutes (text + video)**

---

## Why Causal Inference Matters for Marketing

Marketing measurement isn't just about finding correlations — it's about understanding *what causes what*. When a CMO asks "What happens if I double my TV budget?", they're asking a causal question. Answering it requires causal reasoning, not just statistical associations.

## Directed Acyclic Graphs (DAGs)

A DAG is a visual representation of causal relationships. It consists of:
- **Nodes:** Variables (e.g., TV Spend, Sales, Seasonality)
- **Edges (arrows):** Causal relationships (A → B means "A causes B")
- **Acyclic:** No loops (you can't cause yourself)

### DAG 1: The Naive Model

```
TV Spend → Sales
Meta Spend → Sales
Radio Spend → Sales
```

This is what a naive regression assumes: each channel independently causes sales. But reality is more complex.

### DAG 2: Confounding

```
Seasonality → TV Spend
Seasonality → Sales
TV Spend → Sales
```

**Seasonality** is a *confounder* — it affects both TV Spend (brands spend more during Q4) and Sales (people buy more during Q4). Without controlling for seasonality, the TV coefficient will be biased upward.

**Key insight:** To estimate the causal effect of TV on Sales, we must control for all confounders. This is why MMM includes control variables.

### DAG 3: Mediators

```
TV Spend → Brand Search → Sales
TV Spend → Sales
```

**Brand Search** is a *mediator* — TV drives people to search for the brand, which then drives sales. If we include Brand Search in the model, we "block" part of TV's causal path, underestimating TV's total effect.

**Key insight:** Don't control for mediators! In MMM, this means being careful about including branded search alongside upper-funnel media.

### DAG 4: Colliders

```
TV Spend → Purchase
Online Ad → Purchase
```

If we condition on **Purchase** (e.g., only analyze customers who bought), we introduce a spurious association between TV and Online Ad. This is *collider bias*.

**Key insight:** Don't condition on consequences of the treatment. In MMM, this means modeling total sales, not just "attributed" sales.

## The Three Types of Paths

| Structure | Example | Should you control? |
|---|---|---|
| **Confounder** (X ← Z → Y) | Seasonality affects both spend and sales | Yes — must control |
| **Mediator** (X → Z → Y) | TV drives brand search drives sales | No — blocks causal path |
| **Collider** (X → Z ← Y) | Both channels drive purchase | No — creates bias |

## The Do-Operator (Intuitive)

Judea Pearl's *do-calculus* formalizes causal reasoning:

- **P(Sales | TV = $1M):** "What is sales when we *observe* TV spend is $1M?"
  → This mixes up cases where TV is high because of seasonality

- **P(Sales | do(TV = $1M)):** "What would sales be if we *set* TV spend to $1M?"
  → This is the causal question — what if we intervened?

The difference between "seeing" and "doing" is the essence of causal inference. Experiments directly estimate the "do" version. MMM tries to approximate it by controlling for confounders.

## Why Experiments Are the Gold Standard

Randomized experiments (A/B tests, geo experiments) solve the confounding problem by design:

1. **Randomization** ensures treatment and control groups are comparable on both observed and unobserved confounders
2. The difference in outcomes between groups is the **Average Treatment Effect (ATE)**
3. No need for DAGs, regression, or assumptions about which variables to control

### But experiments have limitations:
- **Expensive:** Running a holdout means lost revenue
- **Slow:** Need enough time to detect effects
- **Limited scope:** Can only test one thing at a time
- **Not always feasible:** Can't easily A/B test TV or OOH

This is why we need both: **MMM for the full picture, experiments for calibration.**

## Causal Inference in MMM

How does this apply to the MMM we'll build?

1. **Include confounders as control variables:** Seasonality (month dummies), trend, price, macroeconomic factors, competitor spend
2. **Don't include mediators:** If TV drives brand search, don't include brand search as a predictor — it will absorb TV's effect
3. **Be aware of reverse causality:** If spend is set based on expected sales (budget follows forecast), the relationship is bidirectional. This is harder to solve and may require instrumental variables or Bayesian structural modeling
4. **Calibrate with experiments:** Use experimental results as informative priors in Bayesian MMM to anchor estimates in causal ground truth

## Practice: Drawing DAGs

Try drawing DAGs for these scenarios:

1. **Scenario A:** A brand runs TV + digital together during product launches. Both channels drive sales, but launches also drive sales independently.

2. **Scenario B:** A retailer adjusts prices weekly based on inventory levels. Both price and inventory affect sales.

3. **Scenario C:** Social media spend drives website visits, which drive conversions. But some people convert directly from social ads without visiting the website.

For each: identify confounders, mediators, and what you should control for in an MMM.

## Key Takeaways

1. **DAGs help you think clearly** about what to include (and not include) in your model
2. **Confounders must be controlled for** — they're the biggest threat to valid MMM results
3. **Mediators should NOT be controlled for** — doing so underestimates the channel's true effect
4. **Experiments provide causal ground truth** — use them to calibrate your MMM
5. **The "do" vs "see" distinction** is fundamental: MMM measures association, experiments measure causation, calibrated MMM bridges the gap

---

*Next reading: "Introduction to Bayesian Thinking"*
