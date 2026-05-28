# June-26 Capstone Guide

The June capstone is a measurement project plus a reusable workflow asset.

Students should demonstrate that they can build a credible marketing measurement
analysis and make the workflow repeatable with agentic practices.

## Required Deliverables

1. Technical notebook
2. Executive readout
3. Measurement plan
4. Agent workflow log
5. Reusable skill or skill patch
6. Behavior eval case

## 1. Technical Notebook

The notebook should contain:

- business context
- data overview and EDA
- variable mapping
- feature design
- model fitting or experiment analysis
- diagnostics
- results and limitations
- recommendations

Students may use March cohort notebooks as a starting point, but the final
notebook must match their chosen business question and dataset.

## 2. Executive Readout

The readout should communicate:

- business question
- method used and why
- main findings
- confidence level
- recommended next action
- limitations
- what would change the recommendation

## 3. Measurement Plan

The plan should include:

- next experiments or validation steps
- calibration opportunities
- data improvements
- reporting cadence
- owner and timeline

## 4. Agent Workflow Log

Use `../artifacts/model-iteration-log-template.md` or a similar format.

The log should include:

- prompts or task summaries
- tools or skills used
- decisions made by the student
- agent mistakes or corrections
- validation performed
- rejected model specs or analysis paths

## 5. Reusable Skill Or Skill Patch

Students must create or improve one skill. Examples:

- dataset onboarding skill for their organization
- MMM variable mapping skill
- Meridian setup skill
- lift-test design skill
- stakeholder readout skill

The skill must pass `../skills/skill-readiness-checklist.md`.

## 6. Behavior Eval Case

Students must write one behavior eval case for the skill.

The case should include:

- `id`
- `skill`
- `prompt`
- `expected`
- `disallowed`

The eval should test a meaningful failure mode:

- competitor spend treated as owned media
- aggregate and component variables included together
- experiment caveat dropped
- calibration evidence over-applied
- budget recommendation overstated
- Meridian setup proceeds despite missing data

## Grading Rubric

| Area | Excellent | Needs Work |
|---|---|---|
| Measurement question | Clear causal or decision question with appropriate scope | Vague question or mismatched method |
| Data and EDA | Data quality and variable roles are inspected and documented | Important data issues are missed |
| Modeling or experiment design | Method matches the question and diagnostics are interpreted correctly | Model is treated as a black box |
| Interpretation | Recommendations match evidence strength and uncertainty | Overconfident or unsupported recommendation |
| Agent workflow | Agent use is logged, validated, and corrected where needed | Agent output is accepted without review |
| Artifact quality | Artifacts preserve decisions and handoffs clearly | Key decisions remain only in chat |
| Skill quality | Skill has clear trigger, workflow, artifact contract, and anti-patterns | Skill is a vague prompt dump |
| Eval quality | Eval targets a severe realistic failure | Eval checks only generic helpfulness |

