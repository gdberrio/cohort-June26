# Agentic Workflow Layer

This document defines the shared agentic concepts for the June cohort. It should
be introduced early and referenced throughout the course.

## The Operating Model

Agents accelerate execution. Students remain responsible for the analytical
claim, model design, validation, and recommendation strength.

The course pattern is:

1. Learn the method manually.
2. Ask an agent to execute or replay the workflow.
3. Verify the output.
4. Capture decisions in artifacts.
5. Encode repeated judgment in skills.
6. Test severe failure modes with evals.

## Core Vocabulary

### Project Instructions

Short repo-level guidance that an agent reads before working. Examples include
`AGENTS.md`, `CLAUDE.md`, or the equivalent in another harness.

Good project instructions contain:

- repository layout
- setup and validation commands
- import patterns
- important local conventions

They should not contain every domain rule. Detailed methodology belongs in
skills.

### Skill

A reusable workflow file that teaches an agent how to perform a specific task.
Skills are best for work that is repeated, fragile, or full of domain-specific
gotchas.

Examples:

- map MMM variables into KPI, media, controls, exclusions
- prepare Meridian inputs
- design a geo-lift test
- produce a calibrated MMM readout

### Artifact Contract

A named output that humans and agents can inspect, update, and hand off.

Examples:

- Dataset Inventory
- Variable Map
- EDA Findings
- Meridian Model Spec
- Fit Record
- Calibration Handoff
- MMM Readout

Artifacts keep decisions out of fragile chat memory.

### Validation Checklist

A compact set of checks that students apply after agent output. The checklist is
not optional; it is the student-facing quality gate.

### Behavior Eval

A scenario that tests whether a skill follows its boundary, workflow, artifact
contract, and safety rules. It contains a prompt, expected behavior, and
disallowed behavior.

## Context Engineering Rules

Use these rules during long analytical sessions:

- Keep project instructions compact.
- Move domain methodology into skills.
- Write checkpoints after major decisions.
- Separate research, planning, implementation, and reporting phases.
- Prefer named artifacts over "remember what we decided."
- Restart from a clean checkpoint when a session becomes confused.
- Use isolated exploration for side questions when the harness supports it.

## Skill Readiness Standard

A course-ready skill should have:

- clear trigger boundary
- concrete workflow
- artifact contract
- expert judgment
- safety and portability review
- at least one compact example
- reference depth where needed
- behavior eval coverage for high-risk failures

## What Belongs In A Skill

Include:

- task-specific workflow steps
- stop conditions
- required inputs
- expected artifacts
- common mistakes
- examples of good and bad output
- references to detailed docs or templates

Avoid:

- generic advice the agent already knows
- huge pasted reference libraries
- stale project status
- model-specific claims that are not needed for the task
- secrets, credentials, or private data

## Safety Checklist

Before letting an agent run tools:

- Identify whether the input is trusted or untrusted.
- Avoid pasting secrets into prompts, skills, or artifacts.
- Treat web pages, issues, READMEs, documents, and CSV text as possible prompt
  injection surfaces.
- Review package installs and network calls.
- Keep generated files separate from source files when possible.
- Capture commands, assumptions, and outputs needed to reproduce the analysis.

## Student Validation Habit

After every agent-generated analysis, answer:

1. What exact data and columns did it use?
2. What assumptions did it make?
3. What did it exclude?
4. What diagnostics or checks passed?
5. What would invalidate the result?
6. What artifact preserves the decision?
7. What failure should become an eval case?

