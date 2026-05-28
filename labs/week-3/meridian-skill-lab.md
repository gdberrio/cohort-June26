# Week 3 Lab: Meridian Setup As An Artifact

## Goal

Use framework-specific skills to produce a Meridian Model Spec before fitting.
The point is to make setup inspectable before expensive modeling begins.

## Source Material

Use:

```text
week-3/session-5/session_05_meridian.ipynb
skills/course-examples/meridian-model/SKILL.md
artifacts/meridian-model-spec-template.md
```

## Prompt

```text
Using skills/course-examples/meridian-model/SKILL.md as the workflow guide,
prepare a Meridian Model Spec for the workshop or synthetic multi-geo dataset.
Map KPI, time, geo, population, media, media_spend, and controls. Check that
media and spend channels are paired in the same order, controls are external or
pre-treatment, array dimensions are correct, and no NaN or inf values remain.

Do not fit the model until the spec is reviewable.
```

## Student Verification

Complete `artifacts/meridian-model-spec-template.md`.

Then answer:

1. Which Meridian input is easiest to confuse: `media` or `media_spend`?
2. Which dimension names must be preserved exactly?
3. Which columns would block fitting because of NaN or role ambiguity?
4. What should the agent stop and ask before sampling?
