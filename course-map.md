# June-26 Course Map

This map organizes the June agentic cohort. The ported March notebooks remain the measurement spine. June adds agentic labs, artifacts, and skill/eval deliverables around those notebooks.

For the recommended reading sequence, use `references/reading-roadmap.md`.

## Week 0: Setup And Agent Readiness

### Source Material

- `README.md`
- `pre-course/00_smoke_test.ipynb`
- `pre-course/notebook_00_explore_dataset.ipynb`
- `AGENTS.md`
- harness-specific instruction files

### New June Layer

- Install and verify Python/Jupyter environment.
- Orient students to coding agents as tools that can read, edit, and run code.
- Explain project instructions (`AGENTS.md`, harness-specific instruction files, or harness equivalent).
- Write a first checkpoint file after setup.
- Inspect one tiny skill and identify what failure it prevents.

### Deliverables

- Smoke test executed.
- Agent answers a constrained repo-orientation prompt.
- `artifacts/checkpoint-template.md` completed for setup status.

## Week 1: Measurement Foundations Plus Context Engineering

### Source Material

- `pre-course/reading_01_what_is_mmm.md`
- `pre-course/reading_03_causal_inference.md`
- `week-1/session-1/session_01_measurement_landscape.ipynb`
- `week-1/session-2/session_02_data_prep_eda.ipynb`
- `week-1/offline/notebook_01_eda.ipynb`
- `references/context-engineering-guide.md`

### New June Layer

- Teach context rot and context engineering before long analytical sessions.
- Add agentic EDA replay after manual EDA.
- Require students to verify agent output against column roles, missingness,
  time grain, correlations, and visual checks.
- Store EDA conclusions in an artifact instead of leaving them only in chat.

### Deliverables

- EDA notebook run.
- `artifacts/eda-findings-template.md` completed.
- `labs/week-1/agentic-eda-lab.md` completed.

## Week 2: MMM From Scratch Plus Skill-Guided Iteration

### Source Material

- `week-2/offline/notebook_02_adstock_saturation.ipynb`
- `week-2/offline/demo_spend_vs_exposure.ipynb`
- `week-2/session-3/session_03_ols_mmm.ipynb`
- `week-2/offline/reading_05_two_stage_funnel_mmm.md`
- `week-2/offline/notebook_03_pymc_basics.ipynb`
- `week-2/session-4/session_04_bayesian_mmm.ipynb`
- `week-2/session-4/nuts-divergences-explainer.md`
- `references/agentic-mmm-demo-guide.md`
- `references/bayesian-vs-frequentist-mmm-guide.md`
- `references/how-to-write-a-skill.md`
- `skills/course-examples/mmm-variable-mapping/SKILL.md`

### New June Layer

- Show a naive agent workflow failing on variable selection or diagnostics.
- Use a variable-mapping skill to create a model-ready role table.
- Compare manual OLS/Bayesian modeling with agent-assisted iteration.
- Capture model decisions, rejected specs, diagnostics, and caveats.

### Deliverables

- Manual model notebook.
- `artifacts/variable-map-template.md` completed.
- `artifacts/model-iteration-log-template.md` completed.
- `labs/week-2/skill-guided-mmm-lab.md` completed.

## Week 3: Meridian, Attribution, And Framework-Specific Skills

### Source Material

- `week-3/session-5/session_05_meridian.ipynb`
- `week-3/offline/notebook_04_meridian_setup.ipynb`
- `week-3/offline/notebook_05_meridian_post_modeling.ipynb`
- `week-3/offline/reading_06_meridian_priors.md`
- `week-3/offline/reading_07_meridian_knots.md`
- `week-3/offline/reading_08_time_varying_intercepts.md`
- `week-3/offline/reading_09_pymc_marketing_time_effects.md`
- `week-3/offline/reading_10_attribution_models.md`
- `week-3/offline/notebook_06_attribution.ipynb`
- `week-3/offline/attribution-deep-dives/README.md`
- `week-3/session-6/session_06_attribution_byod.ipynb`
- `skills/course-examples/meridian-model/SKILL.md`
- `artifacts/meridian-model-spec-template.md`

### New June Layer

- Treat Meridian setup as an artifact-driven workflow.
- Separate model specification, fit record, and diagnostics report.
- Use framework-specific skills for setup pitfalls such as dimensions, media
  vs media spend, controls, NaN handling, and population.
- Keep attribution as a measurement method, but connect it to agent validation:
  agents can compute Shapley or Markov outputs, but students must interpret
  limits.
- Add deeper attribution variants for Markov chains, probabilistic lift,
  regression-based marginal effects, and LSTM/BERT-like sequence models.

### Deliverables

- Meridian model spec or dry-run spec.
- Meridian fit record if fitting is feasible.
- Diagnostics review with explicit caveats.
- `labs/week-3/meridian-skill-lab.md` completed.

## Week 4: Experiments, Calibration, Evals, And Capstone

### Source Material

- `week-4/session-7/session_07_experimentation_geolift.ipynb`
- `week-4/offline/notebook_07_power_analysis.ipynb`
- `week-4/offline/reading_11_calibrating_mmm.md`
- `week-4/session-8/session_08_prior_calibration.ipynb`
- `capstone/capstone-guide.md`
- `capstone/capstone-template.md`
- `references/geo-experiment-agent-demo-guide.md`
- `skills/course-examples/marketing-lift-design/SKILL.md`
- `skills/course-examples/calibration-strategy/SKILL.md`

Optional March reference:

- `capstone/march-legacy/reading_07_capstone_guide.md`
- `capstone/march-legacy/capstone_template.ipynb`

### New June Layer

- Compare manual geo-experiment design prompts with a reusable skill workflow.
- Teach calibration handoff as the bridge from experiments to MMM.
- Add behavior evals for skills and agent workflows.
- Use the step-by-step skill-writing guide before capstone skill patches.
- Require capstone teams to create or improve one reusable workflow asset.

### Deliverables

- Lift-test design brief or experiment analysis report.
- Calibration handoff.
- One skill or skill patch.
- One behavior eval case.
- Updated capstone package.

## Cross-Course Assessment

Students are assessed on both measurement quality and workflow quality.

Measurement quality:

- causal claim is explicit
- variables are correctly classified
- model limitations are stated
- diagnostics and uncertainty are interpreted correctly
- recommendations match evidence strength

Workflow quality:

- project instructions are compact
- checkpoints preserve key decisions
- artifacts are complete and inspectable
- skill boundary is clear
- eval case targets a real high-risk failure
