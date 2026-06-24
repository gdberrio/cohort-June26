# Reading Roadmap

This roadmap gives the recommended order for readings, notebooks, labs, and
skill files in the June cohort. Use it as the canonical sequence when preparing
class, assigning student work, or catching up after a missed session.

The core pattern is:

1. Learn the measurement idea manually.
2. Run or inspect the notebook.
3. Add the agentic workflow layer.
4. Complete the artifact or lab.
5. Convert one severe failure into a skill or eval.

## Instructor Prep Before The Cohort

Read these first to understand the June course shape.

1. `README.md`
   - Course promise, weekly shape, environment setup, and skill progression.
2. `course-map.md`
   - Week-by-week mapping from March measurement material to June agentic
     additions.
3. `agentic-workflow-layer.md`
   - Shared vocabulary: project instructions, skills, artifact contracts,
     validation checklists, and behavior evals.
4. `references/context-engineering-guide.md`
   - Where information belongs: project instructions, skills, artifacts, and
     evals.
5. `references/agent-safety-guide.md`
   - Tool access, untrusted inputs, secret handling, and reproducibility risks.
6. `skills/README.md`
   - Student progression for inspecting, using, adapting, and evaluating
     skills.
7. `references/how-to-write-a-skill.md`
   - Step-by-step guide for creating or improving a skill.
8. `skills/skill-readiness-checklist.md`
   - Rubric for deciding whether a skill is course-ready or capstone-ready.

Instructor note: teach `agentic-workflow-layer.md` as the conceptual spine.
Teach `references/how-to-write-a-skill.md` later, after students have seen
manual measurement workflows fail in realistic ways.

## Week 0: Setup And Agent Readiness

Goal: get the environment working and show that agent work needs compact
instructions, checkpoints, and validation.

Required sequence:

1. `README.md`
2. `AGENTS.md`
3. `pre-course/00_smoke_test.md`
4. `pre-course/notebook_00_explore_dataset.md`
5. `agentic-workflow-layer.md`
6. `references/context-engineering-guide.md`
7. `artifacts/checkpoint-template.md`

Light skill inspection:

1. `skills/README.md`
2. One small course skill, preferably
   `skills/course-examples/mmm-variable-mapping/SKILL.md`

Instructor note: do not ask students to write a skill yet. The goal is only to
notice what a skill is and what failure it prevents.

## Week 1: Measurement Foundations And EDA

Goal: ground students in measurement logic before agentic replay.

Required sequence:

1. `pre-course/reading_01_what_is_mmm.md`
2. `pre-course/reading_02_stats_refresher.md`
3. `pre-course/reading_03_causal_inference.md`
4. `week-1/session-1/session_01_measurement_landscape.md`
5. `week-1/session-2/session_02_data_prep_eda.md`
6. `week-1/offline/notebook_01_eda.md`
7. `references/context-engineering-guide.md`
8. `artifacts/agent-output-validation-checklist.md`
9. `labs/week-1/agentic-eda-lab.md`
10. `artifacts/eda-findings-template.md`

Optional reinforcement:

- `pre-course/reading_04_bayesian_thinking.md`

Instructor note: keep the agentic lesson narrow. Students should learn that
agent output is not accepted until the data grain, columns, missingness,
correlations, and visual checks are verified.

## Week 2: MMM From Scratch And Skill-Guided Iteration

Goal: teach MMM mechanics first, then use skills to preserve modeling judgment.

Required sequence:

1. `week-2/offline/notebook_02_adstock_saturation.md`
2. `week-2/offline/demo_spend_vs_exposure.ipynb`
3. `week-2/session-3/session_03_ols_mmm.md`
4. `references/agentic-mmm-demo-guide.md`
5. `skills/course-examples/mmm-variable-classification/SKILL.md`
6. `skills/course-examples/mmm-variable-mapping/SKILL.md`
7. `artifacts/variable-map-template.md`
8. `labs/week-2/skill-guided-mmm-lab.md`
9. `artifacts/model-iteration-log-template.md`
10. `week-2/offline/notebook_03_pymc_basics.md`
11. `week-2/session-4/session_04_bayesian_mmm.md`
12. `references/bayesian-vs-frequentist-mmm-guide.md`

Skill-building checkpoint:

1. `references/how-to-write-a-skill.md`
2. `skills/skill-readiness-checklist.md`

Optional reinforcement:

- `week-2/offline/reading_05_two_stage_funnel_mmm.md`
- `week-2/session-4/nuts-divergences-explainer.md`

Instructor note: introduce skill-writing only after students have seen why
naive agent workflows misclassify variables or skip diagnostics. The skill is
the response to a real failure, not an abstract prompt-engineering exercise.

## Week 3: Meridian, Attribution, And Framework-Specific Skills

Goal: make framework setup inspectable before expensive or confusing modeling
runs.

Required Meridian sequence:

1. `week-3/session-5/session_05_meridian.md`
2. `week-3/offline/notebook_04_meridian_setup.md`
3. `skills/course-examples/meridian-model/SKILL.md`
4. `artifacts/meridian-model-spec-template.md`
5. `labs/week-3/meridian-skill-lab.md`
6. `week-3/offline/notebook_05_meridian_post_modeling.ipynb`

Meridian deep dives:

1. `week-3/offline/reading_06_meridian_priors.md`
2. `week-3/offline/reading_07_meridian_knots.md`
3. `week-3/offline/reading_08_time_varying_intercepts.md`
4. `week-3/offline/reading_09_pymc_marketing_time_effects.md`

Attribution sequence:

1. `week-3/offline/reading_10_attribution_models.md`
2. `week-3/offline/notebook_06_attribution.md`
3. `week-3/offline/attribution-deep-dives/README.md`
4. `week-3/offline/attribution-deep-dives/markov_chain_attribution.md`
5. `week-3/offline/attribution-deep-dives/notebook_markov_chain_attribution.md`
6. `week-3/offline/attribution-deep-dives/probabilistic_attribution.md`
7. `week-3/offline/attribution-deep-dives/notebook_probabilistic_attribution.md`
8. `week-3/offline/attribution-deep-dives/regression_based_attribution.md`
9. `week-3/offline/attribution-deep-dives/notebook_regression_based_attribution.md`
10. `week-3/offline/attribution-deep-dives/sequence_models_attribution.md`
11. `week-3/offline/attribution-deep-dives/notebook_sequence_model_attribution.md`
12. `week-3/session-6/session_06_attribution_byod.md`

Instructor note: the Meridian skill is intentionally framework-specific. Use it
to show why skills are useful when a tool has strict array, dimension, and
input-contract requirements.

## Week 4: Experiments, Calibration, Evals, And Capstone

Goal: connect causal evidence, MMM calibration, and behavior evals.

Required sequence:

1. `week-4/session-7/session_07_experimentation_geolift.md`
2. `week-4/offline/notebook_07_power_analysis.md`
3. `week-4/offline/causalpy-marketing-examples/README.md`
4. `week-4/offline/causalpy-marketing-examples/causal_design_map.md`
5. `week-4/offline/causalpy-marketing-examples/notebook_panel_methods.md`
6. `week-4/offline/causalpy-marketing-examples/notebook_cross_sectional_methods.md`
7. `skills/course-examples/marketing-lift-design/SKILL.md`
8. `references/geo-experiment-agent-demo-guide.md`
9. `week-4/offline/reading_11_calibrating_mmm.md`
10. `week-4/session-8/session_08_prior_calibration.md`
11. `skills/course-examples/calibration-strategy/SKILL.md`
12. `artifacts/calibration-handoff-template.md`
13. `labs/week-4/evals-and-calibration-lab.md`
14. `evals/skills/mmm-variable-mapping/cases.yaml`

Capstone sequence:

1. `capstone/capstone-guide.md`
2. `capstone/capstone-template.md`
3. `references/how-to-write-a-skill.md`
4. `skills/skill-readiness-checklist.md`
5. `artifacts/model-iteration-log-template.md`

Optional March reference:

- `capstone/march-legacy/reading_07_capstone_guide.md`

Instructor note: evals should come after students have something worth testing.
Push them toward severe failures: dropped causal caveats, wrong variable roles,
over-applied calibration evidence, unsupported budget recommendations, or
Meridian setup proceeding despite missing data.

## Capstone Reading Path For Students

Use this condensed path when students are focused on their final project.

1. `capstone/capstone-guide.md`
2. `capstone/capstone-template.md`
3. The week materials that match their method:
   - MMM: Week 2 readings, notebooks, and variable-mapping skill
   - Meridian: Week 3 Meridian readings, notebooks, and model skill
   - Experiment or calibration: Week 4 experiment, CausalPy-inspired
     quasi-experiment, and calibration materials
   - Attribution: Week 3 attribution readings and notebooks
4. `artifacts/agent-output-validation-checklist.md`
5. `artifacts/model-iteration-log-template.md`
6. `references/how-to-write-a-skill.md`
7. `skills/skill-readiness-checklist.md`

Capstone rule: every team should be able to point to the artifact that preserves
each major decision. If the decision exists only in chat, it is not durable
enough for submission.

## When Students Are Behind

Minimum catch-up path:

1. `README.md`
2. `agentic-workflow-layer.md`
3. `course-map.md`
4. The current week's primary notebook or session file
5. The current week's lab
6. The artifact template for the current week
7. `artifacts/agent-output-validation-checklist.md`

Then backfill earlier technical readings as needed.

## Instructor Teaching Rhythm

For each week, keep the rhythm consistent:

1. Manual concept.
2. Manual notebook.
3. Agent replay.
4. Validation checklist.
5. Artifact handoff.
6. Skill or eval reflection.

That rhythm keeps the course from becoming tool marketing. The agentic layer is
there to make measurement work more inspectable, repeatable, and honest.
