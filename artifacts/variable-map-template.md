# Variable Map

## Modeling Goal

- Business decision:
- KPI:
- Unit of analysis:
- Time grain:
- Model branch:

## Variable Roles

| Variable | Role | Include? | Transformation Needed? | Source | Notes |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Role Definitions

- KPI: outcome the model explains.
- Media input: brand-controlled marketing activity.
- Media spend: cost associated with media input.
- Control: external or pre-treatment variable that helps reduce confounding.
- Context only: useful for interpretation, not a model feature.
- Exclusion: leakage, outcome decomposition, mediator, collider, duplicate, or
  unusable variable.
- Unresolved: requires business or data-owner clarification.

## Causal Checks

| Question | Notes |
|---|---|
| Could any selected control be post-treatment? |  |
| Are any media volume variables mediators rather than inputs? |  |
| Are any outcome decompositions included as predictors? |  |
| Are competitor variables treated as controls, not owned media? |  |
| Is there overlap between aggregate and component variables? |  |

## Handoff

- Ready for EDA?
- Ready for feature design?
- Needs causal review?
- Needs business clarification?

