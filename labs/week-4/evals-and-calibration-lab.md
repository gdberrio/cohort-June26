# Week 4 Lab: Calibration Handoff And Behavior Evals

## Goal

Turn experiment evidence into a calibration handoff, then write an eval case
that checks whether a skill preserves the causal caveats.

## Source Material

Use:

```text
week-4/session-7/session_07_experimentation_geolift.ipynb
week-4/session-8/session_08_prior_calibration.ipynb
references/geo-experiment-agent-demo-guide.md
skills/course-examples/calibration-strategy/SKILL.md
skills/course-examples/marketing-lift-design/SKILL.md
```

## Prompt

```text
Using skills/course-examples/calibration-strategy/SKILL.md as the workflow
guide, create a Calibration Handoff for an MMM from a geo-lift or lift-test
result.
Summarize the treatment, effect estimate, uncertainty, population, time period,
transferability limits, and whether the evidence should be used as a prior,
guardrail, sensitivity check, or context only.
```

## Student Verification

Complete `artifacts/calibration-handoff-template.md`.

Then create one behavior eval case under `evals/skills/<skill>/cases.yaml`.

The eval should include:

- `id`
- `skill`
- `prompt`
- `expected`
- `disallowed`

## Reflection

Answer:

1. What caveat would be most dangerous for an agent to drop?
2. What behavior should the eval explicitly forbid?
3. Is this eval testing routing, workflow behavior, or final interpretation?
