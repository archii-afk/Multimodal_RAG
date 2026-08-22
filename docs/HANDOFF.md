# Current handoff

Keep this page current while work is in progress. Replace stale task details after a handoff; durable facts belong in `PROJECT_CONTEXT.md` or `DECISIONS.md`.

## Active task

- **Owner:** Claude Code
- **Branch:** `main`
- **Goal:** Finalize design with Codex review, then land the node/edge schema and storage layer so ingestion workers can be built against it.
- **Status:** Design written and accepted by user; awaiting Codex critique.

## Completed

- Problem statement reviewed with user; confirmed requirements in `docs/PROJECT_CONTEXT.md`.
- Architecture chosen (Option A) and recorded in `docs/DECISIONS.md`.
- Design spec: `docs/superpowers/specs/2026-08-22-multimodal-rag-design.md`.

## Checks run

- None yet (no code).

## Open questions

- Codex: any objections to the node/edge model, the frame-sampling cap, or the work split? Anything in ingestion you would do differently?
- User is downloading videos into `data/raw/`.

## Next action

1. Codex reviews the spec and replies here or in chat.
2. Claude scaffolds the project (`pyproject`, `src/mmrag/`, schema, storage) on `claude/schema`, merges to `main`.
3. Codex starts `codex/ingestion` worktree: video/audio/frame/pdf/entity workers with fixture tests.
