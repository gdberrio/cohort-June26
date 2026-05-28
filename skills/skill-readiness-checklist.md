# Skill Readiness Checklist

Use this checklist before treating a skill as course-ready or capstone-ready.

## Trigger Boundary

- [ ] The description says when to use the skill.
- [ ] The description includes realistic trigger phrases.
- [ ] The skill says when not to use it.
- [ ] Adjacent skills or workflows are named when relevant.

## Concrete Workflow

- [ ] The workflow has observable steps.
- [ ] Required inputs are explicit.
- [ ] Stop conditions are explicit.
- [ ] Handoffs are explicit.
- [ ] The skill does not over-prescribe private reasoning.

## Artifact Contract

- [ ] The skill names the artifact it creates, consumes, or updates.
- [ ] The artifact has a template or schema.
- [ ] The artifact preserves decisions, caveats, and open questions.

## Expert Judgment

- [ ] The skill includes domain rules a base model may miss.
- [ ] It includes common failure modes.
- [ ] It distinguishes evidence strength from recommendation strength.
- [ ] It avoids unsupported causal claims.

## Safety And Portability

- [ ] It avoids secrets and credentials.
- [ ] It works without hard-coded local-only paths unless intentionally course-specific.
- [ ] It warns about untrusted inputs when relevant.
- [ ] It does not require a specific agent harness unless the course exercise is harness-specific.

## Progressive Disclosure

- [ ] The main `SKILL.md` is concise.
- [ ] Detailed references live in `references/` when needed.
- [ ] Scripts are used only for mechanical, repeatable, costly failure modes.

## Example And Eval

- [ ] It has at least one compact example.
- [ ] It has at least one behavior eval case for a severe failure mode.
- [ ] The eval includes expected and disallowed behavior.

