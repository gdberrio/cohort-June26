# Skills Bridge

The June cohort uses skills as reusable workflow assets. This folder documents
how students should inspect, use, adapt, and evaluate skills during the course.

The June student repo is self-contained. Course examples live under
`skills/course-examples/` and are intentionally small enough to inspect in class.

## Skill Learning Progression

### 1. Inspect

Start with course-specific examples:

- `skills/course-examples/mmm-variable-classification/SKILL.md`
- `skills/course-examples/mmm-variable-mapping/SKILL.md`
- `skills/course-examples/meridian-model/SKILL.md`
- `skills/course-examples/marketing-lift-design/SKILL.md`
- `skills/course-examples/calibration-strategy/SKILL.md`

Questions:

- What task should trigger this skill?
- What mistake does it prevent?
- What evidence or artifact should it produce?
- What is too specific to the March workshop dataset?

### 2. Use

Run or adapt one of the course example skills:

- `mmm-variable-mapping`
- `meridian-model`
- `marketing-lift-design`
- `calibration-strategy`

Questions:

- Which artifact did it create or update?
- What did you have to verify manually?
- Did it stop when required context was missing?

### 3. Critique

Compare skill output against `../artifacts/agent-output-validation-checklist.md`.

Questions:

- Did it overclaim causality?
- Did it misclassify any variables?
- Did it preserve uncertainty and caveats?
- What instruction would have prevented a mistake?

### 4. Adapt Or Build

Create or modify a skill for a repeated workflow:

- client-specific data dictionary
- local variable mapping
- Meridian setup
- experiment QA
- post-model readout

Read `../references/how-to-write-a-skill.md` for the step-by-step build path.
Use `skill-readiness-checklist.md` before calling it done.

### 5. Evaluate

Write one behavior eval case for the skill:

- prompt
- expected behavior
- disallowed behavior
- optional artifact assertion

The eval should target a severe failure mode, not generic helpfulness.

## Recommended Skill Skeleton

```markdown
---
name: short-skill-name
description: >
  Clear trigger boundary. Use this when...
argument-hint: <optional arguments>
---

# Skill Title

## Goal

What this skill accomplishes.

## Use This When

- Specific trigger case.

## Skip This When

- Adjacent task that belongs to another skill.

## Inputs

- Required context, files, or artifacts.

## Workflow

1. Inspect evidence.
2. Make decisions.
3. Produce or update artifact.
4. Stop or hand off.

## Output

Name the artifact or final response shape.

## Anti-Patterns

- Common mistake to avoid.
```
