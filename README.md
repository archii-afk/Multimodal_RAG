# Multimodal RAG — an evidence graph for RAG-ready data

Submission for the *Multimodal Data Management Pipeline for RAG-Ready Systems* hackathon
(brief: [`multimodal_data_management_hackathon.pdf`](multimodal_data_management_hackathon.pdf)).

It turns video, audio, screen frames, on-screen text and PDFs into a **graph of evidence** —
every item keeps its modality, source, timestamp/page, confidence and provenance, and is linked
to the other evidence that was present at the same moment or makes the same claim — so a query
can return *the sentence, the diagram that was on screen when it was said, the page that backs
it, and who said it*, each traceable to the original file.

```
"How does the async task flow work, who explained it, and where was the diagram shown?"

  text-only RAG   → the right sentences at 08:04–09:09. No speaker. No diagram.
  this system     → sentences + frames at 08:00–09:30 + presenter + claims → a cited answer
```

---

## 1. Problem and approach

Text RAG chunks and embeds. When knowledge is split across a speaker's words, a diagram, the
text inside the diagram and the moment they coincided, chunking destroys the relationships.
The brief's own example — *what architecture was discussed, who explained it, where was the
diagram shown* — cannot be answered from any single chunk.

Our approach: treat every extracted item as a **node** with full provenance, and make the
relationships **edges** — temporal (*this was on screen while this was said*), semantic (*this
frame illustrates this claim*), and cross-file (*this page supports this claim*). Retrieval
seeds per modality, then walks typed paths through those edges. A text-only baseline and a
flat-multimodal ablation run on the *same* store, so the comparison isolates the graph's
contribution.

## 2. Architecture

```
 data/raw/*.mp4 ─┬─ ffmpeg ─── audio.mp3 ── Whisper ─────────── transcript_segment (+ spoken_by, next)
                 └─ ffmpeg ─── frames/*.jpg ── Gemini 2.5 Flash ── frame, ocr_block (+ depicts, mentions)
 data/raw/*.pdf ──── PyMuPDF ─────────────────────────────────── pdf_chunk (page-bounded)
                                        │
                     GPT-4.1-mini claim pass ───────────────────── claim, entity (+ expresses, supports, involves)
                                        ▼
                     SQLite  nodes ── edges ── embeddings (numpy index)
                                        ▼
                     linkers: co_occurs_at (time overlap) · illustrates (entity ∩ + co-occurrence) · same_topic (cross-source cosine)
                                        ▼
                     retrieval: text_only | flat_multimodal | graph (per-modality seeds → typed paths) → cited answer
```

### 2.1 Ingestion (`src/mmrag/ingest/`)

Each worker is a pure function `(path) -> IngestBatch`; none touches the database. Every
external call is cached on disk under `sha256(content) + model + prompt_version`, so re-runs
after a code change are free and only a prompt change re-runs that worker.

| worker | what it does |
|---|---|
| `audio.py` | `ffmpeg` → 16 kHz mono 32 kbps MP3 (a 40-min talk is 9.6 MB, under Whisper's 25 MB; silence-split fallback exists). One Whisper call with segment timestamps → `transcript_segment` nodes, `next` chain, `spoken_by` → presenter `person`. |
| `video.py` | Hybrid sampling: a frame every 5 s **plus** scene-change candidates, de-duplicated by 64-bit difference hash, ≥1 frame per 30 s guaranteed. Each frame gets a **validity window** (midpoints to neighbours), not an instant. 40 min → 121 JPEGs → 93 kept. |
| `vision.py` | One Gemini call per frame (3 threads, JSON schema): screen description as a knowledge statement, OCR text, entities, `is_diagram`, model confidence. Emits `ocr_block` children and `depicts`/`mentions` edges. |
| `pdf.py` | PyMuPDF; ~650-token chunks with overlap that **never cross a page**; page + character offsets in provenance. |
| `claims.py` | Batched GPT pass over segments and chunks → `claim` nodes ("the worker writes the result to DynamoDB") with `expresses` / `supports` / `involves` / `mentions` edges. |

### 2.2 Structured representation (`src/mmrag/model.py`, `store.py`)

One `nodes` table, one `edges` table, one `embeddings` table in SQLite.

**Node** — `kind` (`source`, `transcript_segment`, `frame`, `ocr_block`, `image`, `pdf_chunk`,
`entity`, `claim`), `modality`, `content`, `source_id`, `t_start`/`t_end`, `page`, `bbox`,
`speaker`, `confidence` (method default used for ranking), `model_confidence` (LLM's own, stored,
not trusted), `canonical_key` (merge key), `provenance` (extractor, model, prompt version, frame
path, sha256, sample instant). Sources carry `path`, `mime_type`, `sha256`, `duration`,
`presenter`.

**Edges** — `part_of`, `next`, `spoken_by`, `mentions`, `depicts`, `expresses`, `supports`,
`involves`, `co_occurs_at` (weight = seconds of overlap), `illustrates` (weight = shared
entities), `same_topic` (weight = cosine ≥ 0.55, cross-source only, ≤3 per node).

**Contract** — workers build `NodeDraft`/`EdgeDraft` with batch-local refs; an edge may target
`key:<canonical_key>` for a node that lives in another batch. The store resolves refs
atomically, merges entities/claims by key and sources by sha256. This is how five independent
workers converge on one `DynamoDB` entity without coordinating.

### 2.3 Cross-modal and temporal linking (`src/mmrag/linker.py`)

- `co_occurs_at`: indexed interval query per frame → every segment spoken while the frame was on screen.
- `illustrates`: a frame illustrates a claim **only if** it depicts an entity the claim involves
  **and** it co-occurs with a segment expressing the claim. A DynamoDB diagram at 20:30 does not
  attach to a sentence at 09:04.
- `same_topic`: cosine ≥ 0.55 between nodes from different files — how a sentence finds its supporting page. (0.55 was chosen from the measured distribution: p90 of best cross-source match is 0.49, and every pair above 0.55 was on-topic.)

### 2.4 Retrieval (`src/mmrag/retrieval.py`)

| mode | search over | graph | purpose |
|---|---|---|---|
| `text_only` | transcript + pdf | none | the required text-centric baseline |
| `flat_multimodal` | all nodes | none | ablation: is it the frames or the graph? |
| `graph` | all nodes, top-k **per modality** | typed paths | the system |

Typed paths only (no unbounded expansion): `segment → claim ← frame` (illustrates),
`segment → claim ← pdf_chunk` (supports), `segment ↔ frame` (co_occurs_at, top-2 by overlap),
`segment → person` (spoken_by), `* ↔ same_topic` (1 hop).
Score = `cosine × 0.8^hops × confidence` (× an overlap factor for co-occurrence hops).
Every evidence item records the path that reached it; GPT composes the answer citing `[E#]` ids.

## 3. Key design decisions

| decision | why |
|---|---|
| Evidence graph in **SQLite + numpy**, not Neo4j + a vector DB | 13-hour build; ~1.2k nodes per 40-min video; zero infra, single-file demo. The schema ports to Neo4j unchanged (see §7). |
| **Claims** as first-class nodes | Entities alone can't answer "what architecture reduces X"; a claim is the proposition, and frames/pages attach to it. |
| Frames as **time windows** | An instant links to one random sentence; a window links to everything said while the diagram was visible, weighted by overlap. |
| **Fixed 5 s sampling + hash dedupe** over scene detection | Scene detection under-fires on incremental whiteboard strokes and over-fires on scrolling. |
| **Three modes on one store** | Makes the baseline comparison honest: one flag, same data. |
| **Frozen worker/store contract** | Let two agents (Claude Code, Codex) build ingestion and storage in parallel on separate branches. |
| LLM self-reported confidence **stored, not ranked on** | Uncalibrated; ranking uses per-method defaults (ASR 0.9, OCR 0.7, vision 0.75, PDF 1.0, claim 0.7). |
| Cache every API call by content hash | Ingestion was re-run ~6 times while fixing bugs at near-zero cost. |

Full log: [`docs/DECISIONS.md`](docs/DECISIONS.md). Design spec:
[`docs/superpowers/specs/2026-08-22-multimodal-rag-design.md`](docs/superpowers/specs/2026-08-22-multimodal-rag-design.md).

## 4. Dataset and demo

- **Primary:** a 40:06 screen recording of an ex-Atlassian senior engineer explaining their AWS
  provisioning architecture on an Excalidraw whiteboard (Envoy sidecars, Open Service Broker,
  Packer AMIs + CloudFormation, a FastAPI → SQS → worker → DynamoDB async-task pattern).
  Single speaker; not redistributed (place it in `data/raw/`).
- **Documents (5 PDFs, `data/raw/docs/`, printed from public pages):** Envoy architecture
  terminology and xDS dynamic configuration; the Open Service Broker API spec (67 pp); Atlassian
  "Cloud architecture and operational practices"; the Asynchronous Request-Reply pattern
  (queue → worker → status endpoint → client polling). Chosen after the first transcript pass for
  exact entity overlap with what the engineer draws.

Result of ingesting all six sources: **1,714 nodes** (616 segments, 93 frames of which 49 are
diagrams, 87 OCR blocks, 102 PDF chunks, 388 claims, 409 entities), **7,048 edges** including
707 `co_occurs_at`, 48 `illustrates`, 153 cross-source `same_topic`; 0 warnings.

Demo question and what each mode returns are in [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md);
the web page (`mmrag serve`) shows evidence cards with modality, location
(`Test_video.mp4 @ 08:30–09:00`, `doc.pdf p.3`), score, the path taken, and frame thumbnails.

## 5. Evaluation against text-centric RAG

`mmrag eval eval/questions.json` runs a hand-labelled question set through all three modes and
reports recall of the required evidence (matched by source, kind, and timestamp ± 5 s or page).

| mode | recall@8 | found / required |
|---|---|---|
| text_only (baseline) | **0.54** | 7 / 13 |
| flat_multimodal (ablation) | **0.62** | 8 / 13 |
| graph | **0.92** | 12 / 13 |

Five questions over the 40-min talk + five PDFs; 13 required evidence items (6 transcript
segments, 4 frames, 3 PDF pages). Text-only finds sentences but never a frame. Adding frames to
flat search gains one item. The graph's typed paths recover every frame and all but one PDF page
(it returns the right document on a neighbouring page). The 0.62 → 0.92 gap is attributable to
the links, not to the extra modalities. Reproduce: `mmrag eval eval/questions.json`.

Format of a question: `{"id", "text", "required": [{"source": "talk.mp4", "kind": "frame", "t": 512}, {"source": "doc.pdf", "kind": "pdf_chunk", "page": 3}]}` — see `eval/questions.example.json`.

## 6. Setup and usage

```sh
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
cp .env.example .env            # OPENAI_API_KEY, GOOGLE_API_KEY (Gemini); ffmpeg must be on PATH
.venv/bin/python -m pytest -q   # 46 tests, all offline (API clients are injected)

mmrag ingest data/raw/talk.mp4 data/raw/docs/*.pdf --presenter "Name"
mmrag query "How does the async task flow work, who explained it, and where was the diagram shown?" --mode graph
mmrag eval eval/questions.json
mmrag serve                     # http://127.0.0.1:8000
```

If `mmrag`/`import mmrag` fails on macOS after install, run with `PYTHONPATH=src python -m mmrag.cli …`
(some macOS setups flag the venv's `.pth` file hidden, which Python 3.12.13 then ignores).
`--offline` swaps in a hash embedder and no LLM for smoke tests.

Layout:

```
src/mmrag/model.py      frozen contract: NodeDraft / EdgeDraft / IngestBatch
src/mmrag/ingest/       audio, video, vision, pdf, claims workers
src/mmrag/store.py      SQLite nodes + edges, atomic batch insert, key merging
src/mmrag/linker.py     co_occurs_at · illustrates · same_topic
src/mmrag/embeddings.py OpenAI embeddings + numpy index
src/mmrag/retrieval.py  three modes, typed-path expansion
src/mmrag/pipeline.py   files → workers → store → linkers → embeddings
src/mmrag/answer.py     cited answer composition
src/mmrag/evaluation.py recall harness
src/mmrag/cli.py, web.py
eval/                   labelled question sets
docs/                   spec, decisions, handoff, demo script
data/raw, data/processed   media, frames, caches, evidence.db (gitignored)
```

Stack: Python 3.12 · ffmpeg · OpenAI Whisper / `text-embedding-3-small` / GPT-4.1-mini ·
Gemini 2.5 Flash · PyMuPDF · SQLite · numpy · FastAPI · pytest.

## 7. Future improvements

- **Option B:** nodes/edges to Neo4j, vectors to a dedicated store; typed paths become Cypher. The schema is unchanged.
- Speaker diarization for multi-speaker recordings (currently one presenter per file).
- Image-region provenance (bounding boxes on diagrams and PDF figures).
- Semantic event segmentation instead of 5 s frames and ASR sentence breaks.
- Temporal questions ("how did the design change over the talk") over the `next` chain and claim timestamps.
- Confidence-weighted retrieval tuned on the eval set; cross-file entity resolution with an alias map.

## 8. Status

Done: ingestion, representation, linking, retrieval, web/CLI, evaluation on the real corpus
(video + 5 PDFs). Remaining: record the demo (`docs/DEMO_SCRIPT.md`).

## Collaboration

Built by Claude Code and Codex in parallel worktrees against one frozen contract; `AGENTS.md`
is the shared rulebook, `docs/HANDOFF.md` the live handoff, `docs/DECISIONS.md` the decision log.
