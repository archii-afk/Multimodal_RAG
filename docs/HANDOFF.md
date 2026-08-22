# Current handoff

Keep this page current while work is in progress. Replace stale task details after a handoff; durable facts belong in `PROJECT_CONTEXT.md` or `DECISIONS.md`.

## Active tasks

### Codex — ingestion workers — DONE, ready for Claude review
- **Branch/worktree:** `codex/ingestion` in `../Multimodal_RAG-codex`, rebased onto `main` at `e77ef15`.
- **Commits:** `26bec5f` audio, `468c188` video frames, `baee252` vision, `7d0a953` PDF, `314921e` claims.
- Implemented all five frozen-contract workers with no changes to `src/mmrag/model.py`.
- Whisper, Gemini, and claim-extraction calls are cached on disk and resumable; tests use generated media/PDF fixtures and mocked API boundaries, with no keys.
- **Integration note:** `extract_claims(batch)` returns the enriched full batch because its edges reference the input evidence refs (required by `IngestBatch` validation). In `pipeline.py`, replace `claims = ...; batch = batch.merge(claims)` with `batch = w.extract_claims(batch, model=self.llm_model)` to avoid duplicate refs.
- **Primary video:** not present in either checkout's `data/raw/` as of 2026-08-23, so the real Whisper pass and system/architecture name list remain pending.

### Claude — storage, linker, retrieval — DONE, merged to `main`
- `store.py`, `embeddings.py` (numpy index; OpenAI or hash embedder), `linker.py`, `retrieval.py` (3 modes), `pipeline.py`, `answer.py`, `evaluation.py`, `cli.py`, `web.py`. 38 tests.
- `pipeline.Workers.default()` imports `mmrag.ingest.{audio,video,vision,pdf,claims}` by the contract names — Codex: keep those module paths and function names.
- Next for Claude: review + integrate `codex/ingestion`, then run real ingestion on the primary video once it is in `data/raw/`, pick PDFs, label `eval/questions.json`.

## Completed

- Requirements reviewed; design spec accepted; Codex design review incorporated (`docs/DECISIONS.md` 2026-08-23).
- `src/mmrag/model.py` contract + 8 passing tests; `uv` venv; `.env.example`.
- Five ingestion workers: audio, hybrid video frames, Gemini vision/OCR, page-aware PDF, and claim/entity extraction.

## Checks run

- Before workers: `.venv/bin/python -m pytest -q` → 8 passed.
- After rebase and combined integration: `.venv/bin/python -m pytest -q` → 44 passed.

## Open questions

- Primary video is still pending in `data/raw/`; system/architecture names cannot be extracted yet.
- PDF sources: decide after the first real transcript pass.

## Next action

- Claude reviews and integrates `codex/ingestion`, applying the `extract_claims` pipeline assignment noted above.
- Once the primary video appears, run the real audio pass, record its system names here, and choose overlapping Atlassian PDFs.

## Integration rule

Claude merges. Rebase on `main` before opening for integration; never merge the other agent's unreviewed work.
