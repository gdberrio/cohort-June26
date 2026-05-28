---
name: marketing-lift-design
description: >
  Design a marketing lift test or geo experiment before launch. Use this when
  students need a disciplined brief covering hypothesis, estimand, assignment
  unit, sample size, runtime, guardrails, and readout plan.
---

# Marketing Lift Design

## Goal

Produce a design brief that makes the test decision-ready before money is spent.

## Use This When

- A team wants to test media incrementality.
- The available units are geos, customers, stores, campaigns, or matched markets.
- The analyst needs to decide whether the test is powered enough to run.

## Skip This When

- The test has already ended and the task is analysis.
- The question is only about MMM calibration after a causal estimate exists.
- The treatment cannot be isolated from the control group.

## Inputs

- Business question and decision the test should inform.
- Treatment and control eligibility.
- Pre-period outcome data.
- Planned spend or intervention.
- Minimum detectable effect, budget, runtime, and risk constraints.

## Workflow

1. Define the hypothesis and estimand.
2. Choose the assignment unit and justify it.
3. Define treatment, control, KPI, guardrails, and exclusion rules.
4. Check pre-period data quality, seasonality, outliers, and missingness.
5. Evaluate power or detectable lift under plausible runtimes.
6. Select treatment and control units while checking parallel trends and
   spillover risk.
7. Specify the analysis model and readout plan before launch.
8. List launch QA checks and stop/go criteria.

## Output

Return a lift-test design brief with:

- hypothesis
- estimand
- assignment unit
- treatment and control plan
- KPI and guardrails
- runtime and power assessment
- spillover and validity risks
- analysis plan
- launch QA checklist

## Domain Rules

- Geo experiments usually have few units, so power constraints must be explicit.
- Parallel pre-period trends are necessary but not sufficient.
- Control contamination can bias lift toward zero.
- The readout should separate statistical uncertainty from business relevance.

## Anti-Patterns

- Choosing markets because they are convenient without checking pre-period fit.
- Promising precise channel ROI from an underpowered test.
- Changing the analysis plan after seeing results without labeling it exploratory.
