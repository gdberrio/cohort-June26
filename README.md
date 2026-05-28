# Advanced Marketing Measurement with Coding Agents (Cohort June-26)

This cohort teaches marketing measurement and the agentic workflows that make
the work repeatable. Students build the core methods manually, then learn to
encode the judgment into reusable skills, artifacts, validation checklists, and
eval cases.

The course builds on the March 2026 bootcamp material, but the June version is
not "MMM plus an AI bonus." The agentic workflow layer is part of every week.

## Course Promise

By the end of the cohort, students can:

- build and critique MMM, attribution, and experiment workflows
- use coding agents to accelerate analysis without outsourcing judgment
- structure long agent sessions with compact instructions, checkpoints, and
  phase separation
- create reusable skill files for repeated marketing science workflows
- define artifact contracts that make agent output inspectable
- write lightweight behavior eval cases for high-risk skill failures

## Curriculum Shape

| Stage | Measurement Focus | Agentic Workflow Focus | Core Deliverable |
|---|---|---|---|
| Week 0 | Environment and repository setup | Agent readiness, project instructions, checkpoints | Smoke test plus first checkpoint |
| Week 1 | Measurement landscape, DAGs, EDA | Context engineering and agentic EDA verification | EDA findings artifact |
| Week 2 | Adstock, saturation, OLS, Bayesian MMM | Skill-guided variable mapping and diagnostics | Variable map plus model iteration log |
| Week 3 | Meridian and attribution | Framework-specific skills and artifact contracts | Meridian model spec or fit record |
| Week 4 | Experiments, calibration, capstone | Behavior evals and reusable workflow assets | Skill or skill patch plus eval case |

## Relationship To March Cohort

This repo now contains the March course content that June keeps:

- `pre-course/`
- `week-1/` through `week-4/`
- `utils/`
- `data/`
- `capstone/march-legacy/`

June uses that material as the measurement spine and adds:

- agentic replay labs
- artifact templates
- skill-building exercises
- validation checklists
- behavior eval examples
- capstone requirements for reusable workflows

For the recommended reading order across the cohort, see
`references/reading-roadmap.md`.

## Repository Layout

```text
cohort-June26/
├── README.md
├── AGENTS.md
├── course-map.md
├── agentic-workflow-layer.md
├── artifacts/
├── capstone/
├── data/
├── evals/
├── labs/
├── pre-course/
├── references/
├── utils/
└── week-1/ ... week-4/
```

## Environment Setup

Use the lightweight default install for setup, Week 1, EDA, OLS MMM,
attribution, and most agentic labs:

```bash
uv sync
```

Install heavier stacks only when the relevant week needs them:

```bash
uv sync --extra bayesian   # Week 2 Bayesian MMM
uv sync --extra meridian   # Week 3 Google Meridian
uv sync --extra geolift    # Week 4 GeoLift/R bridge
uv sync --extra apps       # Optional Shiny demo
uv sync --all-extras       # Instructor/full environment
```

Students using `pip` can start with `requirements.txt`, then add the matching
optional file such as `requirements-bayesian.txt` or `requirements-meridian.txt`.

## Working With Course Skills

The course is self-contained. Student-facing skills live in
`skills/course-examples/`:

- `mmm-variable-classification`
- `mmm-variable-mapping`
- `meridian-model`
- `marketing-lift-design`
- `calibration-strategy`

Use them as teaching artifacts and workflow dependencies. The core progression
is:

1. Inspect a course-specific skill.
2. Use a workflow skill on course data.
3. Critique the output against a validation checklist.
4. Adapt or create a skill.
5. Write one behavior eval case for a high-risk failure.

For a step-by-step build path, see
`references/how-to-write-a-skill.md`.

## Validation

There is no single build system yet. Validate changes by:

- reading changed Markdown end to end
- running modified notebooks from this repo when notebook content changes
- checking eval YAML structure before using it in a runner
- keeping generated outputs out of the course source unless they are deliberate
  examples

Useful commands:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute pre-course/00_smoke_test.ipynb
uv run --extra bayesian jupyter nbconvert --to notebook --execute week-2/offline/notebook_03_pymc_basics.ipynb
uv run jupyter nbconvert --to notebook --execute week-1/session-1/session_01_measurement_landscape.ipynb
```
