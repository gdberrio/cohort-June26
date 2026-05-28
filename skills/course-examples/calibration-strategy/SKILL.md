---
name: calibration-strategy
description: >
  Convert experiment evidence into an MMM calibration handoff. Use this when a
  lift test, geo experiment, holdout, or spend shock should constrain or inform
  model interpretation without overstating what the experiment proves.
---

# Calibration Strategy

## Goal

Create a clear handoff from experimental evidence to MMM modeling decisions.
The handoff should preserve the estimand, uncertainty, population, and
transferability limits.

## Use This When

- A lift test or geo experiment produced an effect estimate.
- A Bayesian MMM needs a prior, guardrail, or sensitivity check.
- A stakeholder wants to compare experiment evidence with model results.

## Skip This When

- There is no causal estimate yet; use `marketing-lift-design`.
- The experiment design is invalid and needs redesign before calibration.
- The request asks for a budget recommendation without diagnostics.

## Inputs

- Experiment design summary.
- Treatment, outcome, assignment unit, dates, and population.
- Effect estimate with uncertainty.
- Known threats to validity and implementation notes.
- Target MMM channel, metric, geography, and time period.

## Workflow

1. State the experiment estimand in plain language.
2. Record the effect estimate and uncertainty interval.
3. Compare the experiment population and timing with the MMM population and
   modeling window.
4. Decide the calibration use:
   - prior
   - guardrail
   - sensitivity check
   - diagnostic comparison
   - context only
5. Translate the estimate only as far as the evidence supports.
6. Preserve caveats that affect transferability.
7. Complete `artifacts/calibration-handoff-template.md`.

## Output

Return a calibration handoff with:

- experiment summary
- causal estimate
- uncertainty
- mapping to MMM channel and metric
- recommended calibration role
- transferability limits
- sensitivity checks
- decisions that require human approval

## Domain Rules

- Do not turn one local test into a universal channel ROI prior without a clear
  transportability argument.
- Do not drop confidence intervals or posterior intervals.
- If the test measured incremental revenue, do not silently convert it to profit
  or long-term LTV.
- If spillover, compliance, or selection problems exist, the calibration role
  should usually be weaker than "prior".

## Anti-Patterns

- Treating observational MMM and experiment results as interchangeable.
- Averaging incompatible tests without a common estimand.
- Hiding uncertainty when the handoff is used for executive decisions.
