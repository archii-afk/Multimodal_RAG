# Current handoff

Keep this page current while work is in progress. Replace stale task details after a handoff; durable facts belong in `PROJECT_CONTEXT.md` or `DECISIONS.md`.

## Active tasks

### Codex — ingestion workers
- **Branch/worktree:** `codex/ingestion` → `git worktree add ../Multimodal_RAG-codex -b codex/ingestion main`
- **Goal:** Implement, against the frozen contract in `src/mmrag/model.py`, in this order:
  1. `src/mmrag/ingest/audio.py` — `ingest_audio(path, *, source_ref, presenter) -> IngestBatch`: ffmpeg → 16 kHz mono low-bitrate; Whisper API (OpenAI) segment timestamps; emits `SourceDraft` for the file, `transcript_segment` nodes, `next` edges, `spoken_by` edges to `key:person:<presenter>`. Silence-split fallback only if >25 MB.
  2. `src/mmrag/ingest/video.py` — `sample_video_frames(path, *, source_ref, interval_s=5.0, scene_threshold=0.20, min_gap_s=2.0, max_frames=80) -> IngestBatch`: hybrid sampling, perceptual-hash dedupe, ≥1 frame/30 s, writes JPEGs to `data/processed/frames/<sha>/`, emits `frame` nodes with validity windows (midpoints) and `provenance.sampled_at`.
  3. `src/mmrag/ingest/vision.py` — `ingest_frames(batch, *, model) -> IngestBatch`: Gemini structured JSON per frame; disk cache `data/processed/cache/vision/<sha256>-<model>-<prompt_version>.json`; concurrency 2–4 + backoff; fills frame `content`, emits `ocr_block` (part_of frame), `entity` nodes via `canonical_entity_key`, `depicts`/`mentions` edges, `model_confidence`.
  4. `src/mmrag/ingest/pdf.py` — `ingest_pdf(path, *, source_ref, chunk_tokens=650, overlap_tokens=80) -> IngestBatch`: PyMuPDF, chunks never cross pages, 1-based `page`, char offsets in provenance.
  5. `src/mmrag/ingest/claims.py` — `extract_claims(batch, *, model) -> IngestBatch`: batched LLM pass over transcript segments + pdf chunks → `claim` nodes (`canonical_key="claim:<normalized statement>"`), `expresses`/`supports`/`involves`/`mentions` edges.
- **Constraints:** pure functions, no DB access; every LLM/API call cached and resumable; tests on tiny fixtures (a 10 s synthetic clip generated with ffmpeg, a 2-page generated PDF) that run without API keys by mocking the client; `.venv` via `uv venv -p 3.12 && uv pip install -e ".[dev]"`; add deps to `pyproject.toml` with a one-line reason in the commit. Do not edit `model.py` — if the contract blocks you, write the question here and stop.
- **Also:** after the first Whisper pass on the primary video, list the system names the engineer uses in this file so the PDF sources can be chosen.

### Claude — storage, linker, retrieval
- **Branch/worktree:** `claude/storage`
- **Goal:** `src/mmrag/store.py` (SQLite schema, `insert_batch` with atomic ref resolution and `canonical_key` merging, derived `part_of`), embeddings + vector index, `linker.py` (`co_occurs_at`, `illustrates`, `same_topic`), `retrieval.py` (typed-path expansion, three run modes), CLI, minimal FastAPI page, eval harness.

## Completed

- Requirements reviewed; design spec accepted; Codex design review incorporated (`docs/DECISIONS.md` 2026-08-23).
- `src/mmrag/model.py` contract + 8 passing tests; `uv` venv; `.env.example`.

## Checks run

- `.venv/bin/python -m pytest -q` → 8 passed.

## Open questions

- User is placing the primary video in `data/raw/`.
- PDF sources: decide after first transcript pass (see Codex "Also").

## Integration rule

Claude merges. Rebase on `main` before opening for integration; never merge the other agent's unreviewed work.
