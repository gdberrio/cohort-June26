# Context Engineering For Analytics Agents

This guide adapts the context-engineering lightning session into required June
course practice.

## Why It Matters

Long analytical sessions degrade quietly. The model may still write plausible
code and confident summaries while losing track of earlier exclusions, model
decisions, or caveats.

The fix is not simply "use a bigger context window." The fix is structured
context:

- compact project instructions
- skill files for task-specific methodology
- checkpoint artifacts for current decisions
- phase separation
- explicit validation

## What Belongs Where

| Context Surface | Belongs Here | Does Not Belong Here |
|---|---|---|
| Project instructions | setup, repo layout, validation commands | every domain rule |
| Skill | task workflow, anti-patterns, artifact contract | stale project status |
| Artifact | decisions, evidence, caveats, handoff | vague chat summary |
| Eval | severe failure scenario | broad course notes |

## Session Pattern

1. Start with a compact goal.
2. Ask the agent to inspect only what is needed.
3. Write findings to an artifact.
4. Review and approve decisions.
5. Move to implementation.
6. Write a checkpoint before switching tasks or sessions.

## Checkpoint Timing

Write a checkpoint:

- after EDA
- after variable mapping
- before fitting a costly model
- after rejecting a model spec
- before creating stakeholder recommendations
- whenever the session has drifted across multiple topics

## Recovery Pattern

If an agent starts contradicting prior decisions:

1. Stop adding corrective prompts.
2. Open the latest checkpoint.
3. Update the checkpoint if needed.
4. Start a fresh phase from the checkpoint and relevant artifacts.

## Course Rule

No major analytical decision should live only in chat. It should be preserved in
an artifact, checkpoint, notebook, or eval case.

