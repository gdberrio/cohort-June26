# How To Write A Skill, Step By Step

Skills are reusable workflow files for tasks that agents will perform more than
once. A good skill does not replace analyst judgment. It preserves the judgment
that should guide the agent each time the workflow repeats.

Use this reading when you need to create or improve a course skill for a lab or
capstone.

## What A Skill Is For

Write a skill when a task is:

- repeated across projects or teams
- fragile enough that a generic agent often skips important steps
- full of domain rules that are easy to forget
- expected to produce a durable artifact
- risky enough that the wrong answer deserves a behavior eval

Do not write a skill for every prompt. A checklist, artifact template, notebook,
or one-time instruction may be enough when the workflow is still changing or the
task will not repeat.

## Step 1: Name The Repeated Judgment

Start with the failure you want to prevent.

Weak starting point:

```text
Help me build an MMM.
```

Better starting point:

```text
Before fitting an MMM, classify variables into KPI, media spend, media volume,
controls, competitors, exclusions, and unresolved fields. Flag mediators,
outcome leakage, and aggregate/component duplication.
```

The second version is closer to a skill because it names a repeatable decision
process and the failure modes it should catch.

## Step 2: Define The Trigger Boundary

The skill description is the routing instruction. It should tell the agent when
to load the skill.

Use this shape:

```yaml
---
name: short-skill-name
description: >
  What this skill does. Use this when the user asks for specific trigger cases,
  file types, workflows, or domain tasks.
---
```

Good descriptions are specific:

```yaml
description: >
  Create a model-ready variable map for an MMM dataset. Use this when students
  need to classify columns before fitting OLS, Bayesian, or Meridian models and
  need explicit caveats about mediators, controls, leakage, and duplicate
  channel representations.
```

Weak descriptions are vague:

```yaml
description: >
  Helps with marketing analytics.
```

The agent cannot reliably choose a skill from a vague description.

## Step 3: Say When Not To Use It

A skill should have boundaries. Add a "Skip This When" section so the agent does
not apply the workflow to adjacent tasks.

Example:

```markdown
## Skip This When

- The task is exploratory profiling only; start with EDA first.
- The question is about Meridian array construction; use `meridian-model`.
- The user is asking for causal identification proof from observational data.
```

This keeps the skill from becoming a catch-all prompt dump.

## Step 4: List Required Inputs

Make missing context visible. Required inputs help the agent stop instead of
guessing.

For an MMM variable-mapping skill, inputs might include:

- dataset path and grain
- data dictionary or column descriptions
- config file naming paid media, KPI, channel groups, or controls
- business question and KPI definition

If an input is missing, the workflow should say whether to continue with caveats
or stop and ask.

## Step 5: Write Observable Workflow Steps

The workflow should describe actions a reviewer can see in the output, artifact,
notebook, or command log.

Prefer:

```markdown
1. Identify the observation grain.
2. Name the KPI and verify it is an outcome.
3. Classify columns into role groups.
4. Check for duplicate channel representations.
5. Flag mediators, leakage, post-treatment controls, and ambiguous timing.
6. Recommend a first-pass modeling set and list exclusions.
```

Avoid:

```markdown
Think carefully about the best model.
```

Thinking is not enough. The skill needs visible decisions, checks, and outputs.

## Step 6: Define The Artifact Contract

A course skill should usually create, consume, or update an artifact. Name it
explicitly.

Examples:

- variable map: `artifacts/variable-map-template.md`
- model iteration log: `artifacts/model-iteration-log-template.md`
- Meridian model spec: `artifacts/meridian-model-spec-template.md`
- calibration handoff: `artifacts/calibration-handoff-template.md`

The artifact contract should answer:

- What file or table should exist after the workflow?
- What decisions must it preserve?
- What caveats or open questions must remain visible?
- What should a human reviewer check before the next step?

## Step 7: Add Expert Judgment And Anti-Patterns

This is where the skill earns its keep. Include rules that a base model may miss.

For MMM, useful anti-patterns include:

- selecting variables only because they correlate with the KPI
- including both aggregate and component channel variables
- treating clicks, visits, or leads as independent drivers when they may be
  mediators
- using competitor spend as owned media
- reporting fit metrics without diagnostics and causal caveats

For experiments, useful anti-patterns include:

- treating a noisy lift estimate as a precise truth
- dropping transferability caveats when calibrating MMM
- recommending budget changes without checking uncertainty
- ignoring spillover, contamination, or implementation failures

## Step 8: Keep The Main File Concise

Put the essential workflow in `SKILL.md`. Move long details into a nearby
reference file only when the skill would become hard to inspect.

Useful structure:

```text
skill-name/
|-- SKILL.md
|-- references/
|   `-- detailed-domain-rules.md
`-- scripts/
    `-- validate_artifact.py
```

Add scripts only for mechanical checks that should be deterministic, such as
validating YAML, checking required artifact headings, or inspecting schema
matches. Do not add scripts just because code feels impressive.

## Step 9: Add A Compact Example

Examples make the intended behavior concrete.

Example prompt:

```text
Create a variable map for `data/MMM_Workshop_Data.csv` using
`data/data_dict.csv`. Separate KPI, owned media spend, media volume, controls,
competitor variables, exclusions, and unresolved variables. Flag duplicate
channel representations and possible mediators.
```

Expected behavior:

- read the data dictionary before assigning roles
- preserve unresolved variables instead of guessing
- complete or draft the variable-map artifact
- stop before fitting a model

Disallowed behavior:

- include competitor spend as owned media
- include aggregate and component channel variables together without warning
- fit a model before the variable map is reviewable

## Step 10: Write One Behavior Eval

Every course-ready skill should have at least one eval case for a severe failure
mode.

A minimal eval case looks like:

```yaml
- id: competitor-spend-not-owned-media
  skill: mmm-variable-mapping
  prompt: >
    Create an MMM variable map from a dataset containing TV_Spends,
    Paid_Search_Spends, Brand_B_ATL_Spends, Sales_Revenue_Total, and Date.
  expected: >
    Classifies Brand_B_ATL_Spends as competitor/context or unresolved, not as
    owned media. Flags that competitor spend should not be treated as a media
    treatment controlled by the brand.
  disallowed: >
    Treats Brand_B_ATL_Spends as owned media spend because the column name
    contains "Spends".
```

The eval should test a high-risk mistake, not whether the agent sounds helpful.

## Step 11: Review With The Readiness Checklist

Before calling a skill done, check it against
`skills/skill-readiness-checklist.md`.

The most important questions are:

- Is the trigger boundary clear?
- Are required inputs and stop conditions explicit?
- Does the workflow produce a reviewable artifact?
- Does it include domain judgment that a generic agent may miss?
- Does it avoid unsupported causal claims?
- Is there at least one behavior eval for a realistic severe failure?

## Worked Mini Example

Start with a vague instruction:

```text
Help students analyze experiments.
```

Turn it into a skill boundary:

```yaml
---
name: calibration-strategy
description: >
  Convert experiment evidence into an MMM calibration handoff. Use this when a
  lift test, geo experiment, holdout, or spend shock should constrain or inform
  model interpretation without overstating what the experiment proves.
---
```

Add the workflow:

```markdown
## Workflow

1. State the experiment estimand in plain language.
2. Record the effect estimate and uncertainty interval.
3. Compare the experiment population and timing with the MMM population and
   modeling window.
4. Decide whether the evidence should be used as a prior, guardrail,
   sensitivity check, diagnostic comparison, or context only.
5. Translate the estimate only as far as the evidence supports.
6. Preserve caveats that affect transferability.
7. Complete `artifacts/calibration-handoff-template.md`.
```

Add the key anti-pattern:

```markdown
## Anti-Patterns

- Do not treat a single geo-lift estimate as universal channel truth.
- Do not drop uncertainty when turning an experiment into an MMM prior.
- Do not recommend budget changes without checking model diagnostics.
```

Now the skill has a clear trigger, workflow, artifact, and failure boundary.

## Final Rule

A skill is not a longer prompt. It is a compact operating procedure for a task
where repeated judgment matters.
