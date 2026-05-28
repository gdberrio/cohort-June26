# Agentic MMM Demo Guide

This reference distills the MMM lightning-session material into a student-facing
guide for the June cohort. It is not a separate dependency; use it as context for
Week 2 labs.

## Core Lesson

Coding agents can assemble an MMM workflow quickly, but naive prompting tends to
optimize for statistical convenience rather than marketing-science validity.
Skills and artifacts are useful because they force the workflow to preserve
domain checks that are easy to skip in a live session.

## Common Naive-Agent Failures

- Selecting variables only because they correlate with the KPI.
- Including both aggregate and component versions of the same channel.
- Treating views, visits, clicks, or leads as drivers when they may be
  mediators.
- Fitting before writing down the variable map.
- Reporting fit metrics without coefficient sign checks, VIF, residual
  diagnostics, and causal caveats.
- Treating a plausible decomposition as an identified causal estimate.

## Course Pattern

1. Run the manual notebook first.
2. Ask the agent to produce a variable map before fitting.
3. Review the map against `artifacts/variable-map-template.md`.
4. Fit a first model only after the map is approved.
5. Log each accepted and rejected model in
   `artifacts/model-iteration-log-template.md`.
6. Turn the most severe workflow failure into an eval case.

## Prompt Shape

```text
Before fitting an MMM, inspect the data dictionary and config. Produce a
variable map with KPI, media spend, media exposure, controls, competitor
variables, exclusions, and unresolved fields. Flag aggregate/component
duplication, mediators, post-treatment controls, and outcome leakage.
```

## Verification Questions

- Did the workflow separate variable mapping from model fitting?
- Did it explain why each variable belongs in its role?
- Did it preserve unresolved decisions instead of guessing?
- Did diagnostics change the modeling decision?
- Did the final interpretation state what the model cannot identify?
