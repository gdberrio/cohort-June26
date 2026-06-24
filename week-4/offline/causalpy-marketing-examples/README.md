# CausalPy Marketing Examples

This folder extends Week 4 experimentation material with CausalPy-inspired
quasi-experimental designs translated into marketing and commercial analytics
settings.

Use these after:

1. `../notebook_07_power_analysis.ipynb`
2. `../../session-7/session_07_experimentation_geolift.ipynb`
3. `../reading_11_calibrating_mmm.md`

The official CausalPy examples are the reference implementation. The local
course notebooks use the June cohort's core pandas, statsmodels, and
scikit-learn stack so students can run them without adding another dependency
or waiting for PyMC sampling. The goal is to teach the design logic and
marketing translation first; teams can then port the same design into CausalPy
or a Bayesian workflow when they need posterior uncertainty.

## Files

| Topic | Reading | Notebook |
|---|---|---|
| Causal design map for marketing analytics | `causal_design_map.md` | - |
| Panel and time-series designs | - | `notebook_panel_methods.ipynb` |
| Cross-sectional commercial designs | - | `notebook_cross_sectional_methods.ipynb` |

## Recommended Sequence

1. Read `causal_design_map.md` to choose the design that matches the business
   question.
2. Run `notebook_panel_methods.ipynb` for national campaign lift, multi-geo
   synthetic control, and regional rollout difference-in-differences.
3. Run `notebook_cross_sectional_methods.ipynb` for threshold rules,
   endogenous exposure, and observational treatment selection.
4. Convert the strongest result into a calibration handoff or capstone
   measurement plan.

## What Students Should Notice

- The causal design is chosen from the assignment mechanism, not from the
  preferred modeling package.
- `causal_design_map.md` includes detailed method notes for every translated
  CausalPy family, including designs that are not implemented in the lightweight
  notebooks.
- A clean chart is not enough. Each method has an identifying assumption that
  must be stated before the estimate is used.
- Marketing lift estimates should become reusable evidence: MMM calibration
  priors, Robyn-style calibration bounds, Meridian priors, or capstone decision
  memos.
- Cross-sectional methods answer local or policy-specific questions. They do
  not automatically justify broad budget reallocation.

## CausalPy References

- Examples index: https://causalpy.readthedocs.io/en/stable/notebooks/index.html
- Lift testing with interrupted time series:
  https://causalpy.readthedocs.io/en/stable/notebooks/its_lift_test.html
- Synthetic control with scikit-learn:
  https://causalpy.readthedocs.io/en/stable/notebooks/sc_skl.html
- Bayesian geolift:
  https://causalpy.readthedocs.io/en/stable/notebooks/geolift1.html
- Multi-cell geolift:
  https://causalpy.readthedocs.io/en/stable/notebooks/multi_cell_geolift.html
- Difference-in-differences:
  https://causalpy.readthedocs.io/en/stable/notebooks/did_skl.html
- Regression discontinuity:
  https://causalpy.readthedocs.io/en/stable/notebooks/rd_skl.html
- Instrumental variables:
  https://causalpy.readthedocs.io/en/stable/notebooks/iv_pymc.html
- Inverse propensity weighting:
  https://causalpy.readthedocs.io/en/stable/notebooks/inv_prop_pymc.html
