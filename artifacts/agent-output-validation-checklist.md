# Agent Output Validation Checklist

Use this after any agent-generated marketing measurement analysis.

## Data Layer

- [ ] Correct dataset loaded.
- [ ] Correct time grain identified.
- [ ] Correct dependent variable selected.
- [ ] Media channels include only owned media unless explicitly modeling
      competitors.
- [ ] Competitor spend is not treated as owned media.
- [ ] Aggregate and component variables are not both included.
- [ ] Missing, zero-inflated, and unusable columns are handled explicitly.

## Transformation Layer

- [ ] Adstock is applied before saturation.
- [ ] Channel type assumptions match the channel.
- [ ] Parameter ranges are plausible for the channel.
- [ ] Competition or negative-effect variables use the correct search direction.
- [ ] Transform choices are saved or reproducible.

## Model Layer

- [ ] Intercept/base demand is included or intentionally excluded.
- [ ] Coefficient signs are plausible.
- [ ] Multicollinearity is reviewed.
- [ ] Autocorrelation is reviewed for time-series models.
- [ ] Number of predictors is reasonable for the number of observations.
- [ ] Bayesian diagnostics are interpreted in Bayesian terms.

## Interpretation Layer

- [ ] Contribution, ROI, and marginal ROI are not conflated.
- [ ] Uncertainty is communicated.
- [ ] Experimental or causal evidence is separated from observational evidence.
- [ ] Budget recommendations match evidence strength.
- [ ] Limitations and next validation steps are explicit.

## Workflow Layer

- [ ] Decisions are captured in an artifact.
- [ ] Open questions are listed.
- [ ] Rejected specs are recorded.
- [ ] A high-risk failure is identified for a future eval case.

