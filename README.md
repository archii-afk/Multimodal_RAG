# Multimodal RAG

Workspace for the multimodal data management hackathon. The supplied challenge brief is [`multimodal_data_management_hackathon.pdf`](multimodal_data_management_hackathon.pdf).

Implementation has intentionally not been scaffolded yet. We will first review and discuss the problem statement, settle the product and technical direction, and then choose the stack and repository layout.

## Collaboration

This repository is prepared for both Codex and Claude Code:

- `AGENTS.md` is the canonical set of repository-wide working instructions.
- `CLAUDE.md` imports those instructions for Claude Code.
- `docs/PROJECT_CONTEXT.md` holds agreed requirements and constraints.
- `docs/DECISIONS.md` is the durable decision log.
- `docs/HANDOFF.md` carries current work between people and agents.

Use a separate branch and worktree for each agent when work happens in parallel. See `AGENTS.md` for the workflow.

## Layout

```
src/mmrag/model.py   frozen ingestion contract (NodeDraft / EdgeDraft / IngestBatch)
tests/               pytest
docs/                project memory + design spec
data/raw/            input media (gitignored)
data/processed/      frames, transcripts, caches, SQLite store (gitignored)
```

## Setup

```sh
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
cp .env.example .env   # add OPENAI_API_KEY, GEMINI_API_KEY
.venv/bin/python -m pytest -q
```

## Current state

- Challenge brief: present
- Requirements review: done — see `docs/PROJECT_CONTEXT.md`
- Architecture and stack: decided — see `docs/DECISIONS.md` and `docs/superpowers/specs/2026-08-22-multimodal-rag-design.md`
- Application scaffold: package + frozen ingestion contract landed; storage and workers in progress
