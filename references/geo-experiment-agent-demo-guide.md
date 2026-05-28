# Geo-Experiment Agent Demo Guide

This reference ports the relevant geo-experiment lightning-session ideas into a
student-facing Week 4 note.

## Core Lesson

Coding agents can speed up experiment design by profiling data, running power
checks, proposing market assignments, and drafting readout plans. The analyst
still owns the causal design: treatment definition, assignment unit, spillover
risk, power, and interpretation.

## Design Workflow

1. Define the decision, hypothesis, KPI, treatment, and estimand.
2. Inspect pre-period data quality and market-level patterns.
3. Estimate whether the planned test can detect the effect size that matters.
4. Select treatment and control units using pre-period fit plus business
   constraints.
5. Check spillover, contamination, seasonality, and operational feasibility.
6. Write the analysis and readout plan before launch.
7. After the test, translate evidence into a calibration handoff only as far as
   the design supports.

## Agent Prompt Shape

```text
Design a geo experiment for a planned media lift test. Use the pre-period data
to summarize locations, check power for plausible effect sizes, propose
treatment and control markets, flag spillover risk, and write a launch QA and
readout plan. Stop if the test appears underpowered or contaminated.
```

## Failure Modes To Test With Evals

- The agent promises a precise result from an underpowered test.
- The agent drops spillover or contamination caveats.
- The agent changes the analysis plan after seeing outcomes.
- The agent treats a local short-term lift as a universal long-term MMM prior.
- The agent reports lift without uncertainty.

## Verification Questions

- What is the assignment unit?
- What is the minimum detectable effect?
- Are treatment and control markets comparable before launch?
- What could contaminate the control group?
- How should this result constrain MMM: prior, guardrail, sensitivity check, or
  context only?
