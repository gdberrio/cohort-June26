# Week 1 Lab: Agentic EDA With Verification

## Goal

Replay the manual EDA workflow with a coding agent, then verify the output
instead of accepting it at face value.

## Source Notebook

Use:

```text
week-1/offline/notebook_01_eda.ipynb
```

## Prompt

```text
Using the March cohort workshop data and config, run an EDA pass for MMM.
Identify the KPI, time grain, paid media variables, media volume variables,
controls, competitor variables, missingness, zero inflation, lag relationships,
and correlation or structural redundancy risks.

Save the conclusions into an EDA Findings artifact. Do not make modeling
recommendations yet; focus on what the data can and cannot support.
```

## Student Verification

Complete `artifacts/eda-findings-template.md`.

Then answer:

1. Did the agent use the intended KPI?
2. Did it distinguish spend, exposure, competitor, and control variables?
3. Did it catch aggregate/component redundancy?
4. Did it identify missing columns that should be excluded?
5. Which chart or statistic would you inspect manually before trusting it?

## Expected Learning

The agent can produce the EDA faster than writing each plot from scratch, but
the student must still verify variable roles and data quality.

