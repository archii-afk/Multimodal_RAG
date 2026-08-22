# Decision log

Record decisions that affect architecture, product scope, dependencies, data contracts, evaluation, deployment, or collaboration. Keep entries concise and append new decisions; if one is superseded, link the replacement instead of deleting history.

## Entry template

### YYYY-MM-DD — Decision title

- **Status:** Proposed | Accepted | Superseded
- **Context:** What prompted the decision.
- **Decision:** What was selected.
- **Consequences:** Important tradeoffs and follow-up work.

## Decisions

### 2026-08-22 — Defer implementation scaffolding

- **Status:** Accepted
- **Context:** The repository currently contains the challenge brief, and the user wants to discuss the problem before implementation starts.
- **Decision:** Establish collaboration and project-memory files now; select the stack and application structure after reviewing the brief.
- **Consequences:** No premature framework or dependency choices are encoded in the repository.

### 2026-08-22 — Evidence graph in SQLite + vector index (Option A)

- **Status:** Accepted
- **Context:** 13-hour live hackathon; judges penalize "transcribe everything and vector-search it"; 40% of score is representation + cross-modal/temporal linking. Alternatives: B) Neo4j + dedicated vector DB, C) flat vector RAG with metadata.
- **Decision:** Store all extracted items as typed evidence nodes with edges (`co_occurs_at`, `mentions`, `depicts`, `same_topic`, `part_of`, `next`) in SQLite; embeddings in `sqlite-vec` (numpy fallback). Retrieval = cross-modal vector search + 1–2 hop graph expansion. Text-only baseline reuses the same store with no expansion.
- **Consequences:** Zero infra setup; single-file demo. Graph queries are hand-written SQL rather than Cypher. Option B is the documented future expansion path.

### 2026-08-22 — Stack and work split

- **Status:** Accepted
- **Context:** Keys available for OpenAI and Gemini; two agents (Claude Code, Codex) working in parallel.
- **Decision:** Python 3.11+, ffmpeg, OpenAI Whisper API (ASR), Gemini vision (frame description + OCR + entities in one call), PyMuPDF, OpenAI embeddings, FastAPI + minimal HTML, pytest. Claude owns schema/storage/linker/retrieval/eval/integration; Codex owns ingestion workers and PDF sourcing, built against the schema Claude defines first.
- **Consequences:** Schema must land on `main` before Codex starts workers. Single speaker per video, so no diarization; speaker = presenter label.

### 2026-08-23 — Adopt Codex design-review refinements; freeze ingestion contract

- **Status:** Accepted
- **Context:** Codex reviewed the Option A spec and flagged: no node for propositions, presenter only as text, frames as instants, confounded baseline, scene detection unreliable on Excalidraw screens, uncached vision calls, and the schedule being serialized on storage.
- **Decision:** Add `claim` nodes with `expresses`/`illustrates`/`supports`/`involves` edges; persons as entity nodes with `spoken_by`; frame validity windows with overlap-weighted `co_occurs_at`; a flat-multimodal ablation alongside the text-only baseline; hybrid 5 s + scene sampling with perceptual-hash dedupe; on-disk vision cache keyed by sha256+model+prompt_version; typed-path-only graph expansion. The worker/store boundary is frozen in `src/mmrag/model.py` (`NodeDraft`, `EdgeDraft`, `IngestBatch`, batch-local refs, `key:<canonical_key>` references, `canonical_key` merging). Changing it requires a new DECISIONS entry.
- **Consequences:** Codex builds workers against the contract immediately while Claude builds storage. Deferred: second video, PDF embedded images/bboxes; eval set is 5 questions; PDFs chosen after the first transcript pass. Tooling: `uv` + Python 3.12 venv.
