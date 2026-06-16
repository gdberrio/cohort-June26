# Attribution Deep Dives

This folder extends Week 3 attribution material beyond the introductory
reading and notebook. The goal is to help students understand how modern
attribution systems are built, what their code looks like, and where the
measurement caveats remain.

Use these after:

1. `../reading_10_attribution_models.md`
2. `../notebook_06_attribution.ipynb`
3. `../../session-6/session_06_attribution_byod.ipynb`

## Files

| Topic | Reading | Notebook |
|---|---|---|
| Markov chain attribution | `markov_chain_attribution.md` | `notebook_markov_chain_attribution.ipynb` |
| Probabilistic attribution | `probabilistic_attribution.md` | `notebook_probabilistic_attribution.ipynb` |
| Regression-based attribution | `regression_based_attribution.md` | `notebook_regression_based_attribution.ipynb` |
| LSTM and BERT-like attribution | `sequence_models_attribution.md` | `notebook_sequence_model_attribution.ipynb` |

The notebooks use `data/synthetic_journeys.csv` and are intentionally
lightweight. They are built to run with the core June environment rather than
requiring heavyweight deep learning packages.

## Recommended Sequence

1. Run the Markov notebook to see how sequence transitions become removal
   effects.
2. Run the probabilistic notebook to see posterior uncertainty around
   channel-level conversion lift.
3. Run the regression notebook to separate coefficients from marginal effects.
4. Run the sequence-model notebook to see how occlusion or masking can turn a
   trained journey model into channel credit.

## Measurement Guardrails

Attribution is useful for tactical questions, but it is not the same as
incrementality.

- A model can predict conversion well and still over-credit channels that are
  correlated with high-intent users.
- Removal effects in Markov or model-based attribution are model
  counterfactuals, not randomized holdouts.
- Regression coefficients are not budget recommendations by themselves.
- Deep sequence models can learn journey patterns, but their explanations need
  calibration, occlusion checks, and experiment or MMM triangulation.

