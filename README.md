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
src/mmrag/model.py      frozen ingestion contract (NodeDraft / EdgeDraft / IngestBatch)
src/mmrag/ingest/       workers: audio (Whisper), video frames, vision (Gemini), pdf, claims
src/mmrag/store.py      SQLite evidence graph (nodes + edges, atomic batch insert, key merging)
src/mmrag/linker.py     co_occurs_at (time overlap), illustrates (claim∩frame entities), same_topic
src/mmrag/embeddings.py OpenAI embeddings + numpy vector index
src/mmrag/retrieval.py  text_only | flat_multimodal | graph modes, typed-path expansion
src/mmrag/pipeline.py   files → workers → store → linkers → embeddings
src/mmrag/answer.py     cited answer composition
src/mmrag/evaluation.py recall of labelled evidence per mode
src/mmrag/cli.py, web.py  CLI and one-page FastAPI demo
eval/                   labelled question sets
tests/                  pytest (offline; API clients injected)
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

## Usage

```sh
mmrag ingest data/raw/talk.mp4 data/raw/atlassian-doc.pdf --presenter "Name"
mmrag query "What architecture was discussed for reducing database load?" --mode graph
mmrag eval eval/questions.json        # recall for text_only / flat_multimodal / graph
mmrag serve                           # http://127.0.0.1:8000
```
`--offline` swaps in a hash embedder and no LLM (smoke tests only).

## Current state

- Challenge brief: present
- Requirements review: done — see `docs/PROJECT_CONTEXT.md`
- Architecture and stack: decided — see `docs/DECISIONS.md` and `docs/superpowers/specs/2026-08-22-multimodal-rag-design.md`
- Storage, linking, retrieval, eval, CLI and web page: landed (`claude/storage`)
- Ingestion workers: in progress (`codex/ingestion`)
