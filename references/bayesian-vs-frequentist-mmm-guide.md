# Bayesian vs Frequentist MMM Guide

This reference ports the key ideas from the Bayesian-vs-frequentist lightning
session into a short course note for Week 2.

## Main Takeaway

The practical MMM question is not "Bayesian or frequentist?" The better
question is: how much of the result is coming from data, assumptions,
regularization, priors, and external causal evidence?

## What Actually Changes

- Ridge regression and Bayesian MAP with Gaussian priors can be mathematically
  equivalent at the point-estimate level.
- Full Bayesian workflows add posterior uncertainty, prior sensitivity, and a
  natural way to encode external evidence.
- Both approaches can fail when the data cannot identify channel effects.
- Both approaches depend heavily on variable mapping, transformations,
  regularization, and diagnostics.

## What Students Should Watch For

- Positive-only media priors can make every channel look incrementally positive.
- Small MMM datasets can be prior-dominated.
- Multicollinearity can make individual channel effects unstable.
- Uncertainty intervals are not a substitute for identification.
- Experiments are often needed to calibrate or challenge model claims.

## Agentic Workflow Implication

An agent should not be allowed to choose "Bayesian" as a credibility shortcut.
Ask it to document:

- model assumptions
- prior choices or regularization choices
- prior sensitivity or coefficient stability
- diagnostics
- what causal claim is and is not supported
- whether experimental calibration exists

## Verification Questions

- Did the model report uncertainty and sensitivity, not just point estimates?
- Did the workflow distinguish prediction, decomposition, and causal inference?
- Did the agent preserve limitations when translating results for stakeholders?
- Did it recommend experiments when observational evidence is weak?
