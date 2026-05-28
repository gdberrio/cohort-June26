# AGENTS.md

June-26 is a course repository for Advanced Marketing Measurement with Coding
Agents. Content is educational. Prioritize clarity, reproducibility, and
student learning over clever automation.

## Current Goal

Build the June cohort from the ported March 2026 bootcamp:

- keep March's measurement curriculum as the technical spine
- add agentic workflow labs and artifacts throughout the course
- teach skills, artifact contracts, validation checklists, safety, and evals
- avoid turning the course into tool marketing

## Key Locations

| Path | Purpose |
|---|---|
| `README.md` | June cohort framing and schedule |
| `course-map.md` | Week-by-week mapping from March material to June updates |
| `agentic-workflow-layer.md` | Shared concepts for agents, skills, artifacts, and evals |
| `artifacts/` | Student-facing templates and checklists |
| `labs/` | Agentic lab prompts and exercises by week |
| `evals/` | Lightweight behavior eval examples |
| `pre-course/` | Setup, smoke test, and introductory readings |
| `week-1/` through `week-4/` | Ported measurement notebooks and readings |
| `utils/` | Ported helper package for EDA, MMM, and geo experiments |
| `data/` | Ported course datasets and demo scripts |
| `capstone/` | June capstone plus archived March capstone templates |

## Editing Guidance

- Keep files concise and course-facing.
- Prefer reusable templates and labs over long abstract essays.
- When adapting March material, preserve the original measurement logic and add
  agentic replay/validation sections around it.
- Keep student-facing skill exercises self-contained under
  `skills/course-examples/`.
- Use ASCII unless a source file already uses non-ASCII.

## Agentic Curriculum Principles

- Fundamentals first: students must understand the method before delegating work.
- Skills encode judgment; they do not replace judgment.
- Every agent-generated analysis needs a validation surface.
- Long sessions need checkpoints, phase separation, and compact instructions.
- Artifacts beat chat memory for durable decisions.
- Evals test severe failure modes, not generic helpfulness.

## Validation Commands

There is no formal test suite for this folder yet.

Use these checks as relevant:

```bash
find . -maxdepth 3 -type f | sort
python - <<'PY'
import pathlib, yaml
for path in pathlib.Path("evals").glob("**/*.yaml"):
    yaml.safe_load(path.read_text())
    print(f"ok: {path}")
PY
```

For copied or modified notebooks, validate from this repo:

```bash
uv run jupyter nbconvert --to notebook --execute pre-course/00_smoke_test.ipynb
uv run --extra bayesian jupyter nbconvert --to notebook --execute week-2/offline/notebook_03_pymc_basics.ipynb
uv run --extra meridian jupyter nbconvert --to notebook --execute week-3/offline/notebook_04_meridian_setup.ipynb
```
