---
name: mmm-variable-mapping
description: >
  Create a model-ready variable map for an MMM dataset. Use this when students
  need to classify columns before fitting OLS, Bayesian, or Meridian models and
  need explicit caveats about mediators, controls, leakage, and duplicate
  channel representations.
---

# MMM Variable Mapping

## Goal

Produce an auditable role table before modeling. The output should make it easy
for a reviewer to see what enters the model, what stays out, and what needs a
human decision.

## Use This When

- A dataset has raw marketing, sales, context, competitor, or geo columns.
- A notebook is about to fit an MMM or build transformed media features.
- The analyst needs a role table for `artifacts/variable-map-template.md`.

## Skip This When

- The task is exploratory profiling only; start with EDA first.
- The question is about Meridian array construction; use `meridian-model`.
- The user is asking for causal identification proof from observational data.

## Inputs

- Dataset path and grain.
- Data dictionary or column descriptions, if available.
- Config file that names paid media, KPI, channel groups, or controls.
- Business question and KPI definition.

## Workflow

1. Identify the observation grain: date, geo, customer, journey, or campaign.
2. Name the KPI and verify it is an outcome, not a media or intermediate metric.
3. Classify columns into:
   - KPI
   - media spend
   - media exposure or volume
   - organic or owned activity
   - context controls
   - competitor variables
   - time, geo, or population keys
   - exclusions
   - unresolved
4. Check for duplicate channel representations, such as aggregate and component
   spend in the same model.
5. Flag likely mediators, outcome leakage, post-treatment controls, and
   variables whose timing is ambiguous.
6. Recommend a first-pass modeling set and list the exclusions.
7. Write or update `artifacts/variable-map-template.md`.

## Output

Return a compact table with these columns:

- `variable`
- `role`
- `include_in_first_model`
- `reason`
- `risk_or_question`

Then add a short modeling note with:

- approved KPI
- approved media inputs
- approved controls
- exclusions
- unresolved decisions

## Domain Rules

- Do not include both a channel aggregate and its components in the same model
  unless the model explicitly handles hierarchy.
- Treat views, clicks, visits, and leads as possible mediators when the KPI is
  sales or revenue.
- Treat variables measured after the campaign exposure as possible leakage.
- Controls should generally be external, pre-treatment, or calendar/context
  variables, not consequences of media.
- A variable with a high correlation is not automatically a valid driver.

## Anti-Patterns

- Selecting variables only by correlation.
- Treating every positive coefficient as causal.
- Hiding unresolved role decisions inside prose.
- Ignoring time order when classifying controls.

call `scripts/script.py`

! 
