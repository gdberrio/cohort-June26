# Agent Safety Guide

Coding agents can read files, edit files, run shell commands, call tools, and
sometimes access the internet. That power is useful for analytics work, but it
changes the risk profile.

## Core Risks

- Prompt injection in web pages, READMEs, issues, documents, and CSV fields.
- Secret leakage through prompts, logs, generated artifacts, or tool calls.
- Unreviewed package installs or scripts.
- Untrusted data causing misleading analysis.
- Irreproducible work hidden inside chat history.

## Before Using Tools

Ask:

1. Is the input trusted?
2. Could this file or webpage contain instructions aimed at the agent?
3. Does the agent need network access?
4. Could this command modify source data?
5. Are secrets or credentials present?
6. What artifact will preserve the result?

## Safe Defaults For Course Work

- Use local course data unless the exercise explicitly requires web data.
- Keep secrets out of prompts and Markdown files.
- Review shell commands before running them.
- Prefer dry-runs when available.
- Keep generated reports and intermediate outputs separate from source content.
- Record assumptions and commands needed to reproduce results.

## Student Submission Requirement

For any capstone that uses an agent, include a short safety note:

- tool access used
- external data used
- secrets avoided
- validation performed
- reproducibility artifacts included

