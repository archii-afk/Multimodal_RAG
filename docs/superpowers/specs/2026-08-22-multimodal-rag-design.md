# Multimodal RAG — design spec (v1, hackathon)

Date: 2026-08-22 (rev. 2026-08-23 after Codex review) · Status: Accepted by user · Brief: `multimodal_data_management_hackathon.pdf`

## 1. Goal

Turn a realistic multimodal corpus (video, audio, images/frames, PDFs) into an
**evidence graph** that preserves cross-modal and temporal links, and expose a
retrieval interface that answers questions whose evidence spans modalities, e.g.
*"What architecture was discussed for reducing database load, who explained it,
and where was the corresponding diagram shown?"* — returning transcript text,
the presenter, the timestamp, the frame, and the related document page.

Priority: **working demo first**; baseline comparison second; stretch goals only
if time remains. Judging weights: ingestion 20 / representation 20 /
cross-modal+temporal 20 / retrieval 20 / innovation 15 / demo 5.

## 2. Dataset

- Primary: ~40-min talking video of an ex-Atlassian senior engineer explaining
  Atlassian's AWS architecture with Excalidraw-style diagrams on screen.
- Secondary (deferred until the primary path works end to end): one short
  (3–10 min) architecture video by Archisman, for cross-file linking.
- Documents: 2–3 public Atlassian engineering blog posts / architecture docs
  exported to PDF, chosen **after the first transcript pass** for exact entity
  overlap with what the engineer names on screen.
- Images: diagram frames extracted from video, plus any standalone diagram
  images pulled from the blog posts.
- All raw media lives in `data/raw/` (gitignored). Processed artifacts in
  `data/processed/` (gitignored).

## 3. Architecture: evidence graph in SQLite + vector index (Option A)

Chosen over Neo4j + external vector DB (Option B, future expansion) and pure
vector RAG with metadata (Option C, rejected: it is what judges penalize).

```
data/raw/*  ──► ingestion workers ──► evidence nodes + edges ──► SQLite (graph)
                                                          └──► vector index
query ──► embed ──► top-k across ALL modalities ──► expand 1–2 hops ──► bundle
```

### 3.1 Node model (one table `nodes`)

| field        | notes |
|--------------|-------|
| id           | uuid, assigned by the store |
| kind         | `source` · `transcript_segment` · `frame` · `ocr_block` · `image` · `pdf_chunk` · `entity` · `claim` |
| modality     | `audio` · `video` · `image` · `document` · `entity` |
| content      | text: transcript text, OCR text, vision description, PDF chunk, entity name, claim statement |
| source_id    | FK → source node (authoritative; `part_of` edges are derived from it) |
| t_start, t_end | seconds. Frames carry a **validity window** (midpoints to adjacent frames), not an instant |
| page         | 1-based PDF page; chunks never span pages |
| speaker      | presenter label (denormalized; the graph edge `spoken_by` is authoritative) |
| confidence   | 0–1 used for ranking; method-level default (ASR/OCR/PDF/LLM) |
| model_confidence | raw LLM self-reported confidence, stored but not trusted for ranking |
| canonical_key | entities/persons/claims: normalized key used to merge duplicates across sources |
| provenance   | JSON: extractor + model + prompt_version, raw path, frame path, sha256, similarity threshold |
| embedding_id | FK into vector index |

Source nodes carry structured fields: `title`, `path`, `mime_type`, `sha256`, `duration`, `presenter`.

**Claims.** A claim is a proposition extracted from speech or text, e.g.
*"Atlassian added read replicas to reduce load on the primary database."*
Entities alone ("read replica") cannot answer "what architecture reduces
database load"; claims can. Claims are extracted in one batched LLM pass over
transcript segments and PDF chunks.

**Persons.** The presenter is an `entity` node with `canonical_key=person:<name>`.

### 3.2 Edge model (`edges`: src, dst, kind, weight, provenance)

| kind           | src → dst | meaning / weight |
|----------------|-----------|------------------|
| `part_of`      | node → source; ocr_block → frame; image → pdf_chunk | derived structure |
| `next`         | segment → segment | temporal order within a source |
| `co_occurs_at` | transcript_segment ↔ frame | time overlap; weight = overlap seconds |
| `spoken_by`    | transcript_segment → person entity | who said it |
| `mentions`     | segment / pdf_chunk / ocr_block → entity | textual mention |
| `depicts`      | frame / image → entity | from vision description |
| `expresses`    | transcript_segment → claim | speech states the claim |
| `illustrates`  | frame / image → claim | visual shows the claim |
| `supports`     | pdf_chunk → claim | document backs the claim |
| `involves`     | claim → entity | claim is about these entities |
| `same_topic`   | node ↔ node, **cross-source only** | embedding similarity; capped at N per node; threshold in provenance |

Entities are normalized (lowercased canonical name + alias map) so the same
system ("Tenant Context Service", "TCS") links across video, OCR, and PDF.

### 3.3 Ingestion contract

Workers are pure functions returning an `IngestBatch` of `NodeDraft`/`EdgeDraft`
with **batch-local `ref`s**; the store resolves refs to ids atomically and merges
entities/claims by `canonical_key`. Workers never touch the database. The
contract lives in `src/mmrag/model.py` and is frozen; changes require a
DECISIONS entry.

```python
def ingest_audio(path, *, source_ref, presenter) -> IngestBatch
def sample_video_frames(path, *, source_ref, interval_s=5.0,
                        scene_threshold=0.20, min_gap_s=2.0, max_frames=80) -> IngestBatch
def ingest_frames(batch, *, model) -> IngestBatch      # adds descriptions/OCR/entities
def ingest_pdf(path, *, source_ref, chunk_tokens=650, overlap_tokens=80) -> IngestBatch
def extract_claims(batch, *, model) -> IngestBatch     # transcript + pdf → claims
```

### 3.4 Ingestion workers

- **audio**: `ffmpeg` → 16 kHz mono 32–48 kbps (40 min fits Whisper's 25 MB);
  if not, split at silences into ~10-min chunks with 1–2 s overlap, offset
  timestamps, dedupe. Whisper API with segment timestamps → transcript_segments
  + `spoken_by` presenter.
- **frames**: hybrid sampling — fixed every 5 s plus scene-change candidates at
  0.15–0.25, near-duplicate rejection by perceptual hash, guaranteed ≥1 frame
  per 30 s, target 40–80 frames. (Pure scene detection under-fires on incremental
  Excalidraw strokes and over-fires on scroll/cursor.) Each frame gets a validity
  window.
- **vision**: Gemini, one call per retained frame returning structured JSON
  `{description, ocr_text, entities, is_diagram, model_confidence}`; cached on
  disk by `sha256(frame)+model+prompt_version`; concurrency 2–4 with backoff;
  resumable. Entities from this call are normalized deterministically — no
  second entity pass on frames.
- **pdf**: PyMuPDF, chunk within page boundaries by blocks/headings,
  ~650 tokens with ~80 overlap, record page + char offsets. Embedded images and
  bboxes deferred.
- **claims + entities**: one batched LLM pass over transcript segments and PDF
  chunks → claim nodes, entity nodes, `expresses`/`supports`/`involves`/`mentions`.
- **linker**: time-overlap → `co_occurs_at`; frame entities ∩ claim entities →
  `illustrates`; cross-source embedding similarity → `same_topic`.

### 3.5 Retrieval

1. Embed query; vector search across all node kinds.
2. Entity/claim match from query → seed nodes.
3. Expand along **typed paths only** (no unrestricted recursion):
   `segment → claim → frame` (illustrates), `segment → claim → pdf_chunk`
   (supports), `segment ↔ frame` (co_occurs_at), `segment → person` (spoken_by),
   `* → same_topic` (1 hop, capped).
4. Score = similarity × confidence; bundle into **evidence items** each with
   modality, source, timestamp/page, frame thumbnail, and the full path taken.
5. LLM composes the answer citing evidence ids.

### 3.6 Evaluation (deliverable)

Three runs over the same store and a 5-question hand-labelled set (the demo
query + four single-link checks), measuring recall of required evidence:

| run | modalities | graph expansion |
|-----|-----------|-----------------|
| text-only baseline | transcript + pdf | no |
| flat multimodal ablation | all | no |
| full system | all | yes |

The ablation isolates the graph's contribution from merely having frames.

### 3.7 Interface

CLI (`ingest`, `query`) + one FastAPI page: query form and evidence cards with
thumbnails and timestamps. No auth, no polish.

## 4. Stack

Python 3.11+, `ffmpeg`, OpenAI (Whisper, embeddings, chat), Gemini (vision),
PyMuPDF, SQLite + `sqlite-vec` (switch to numpy brute-force if it fights back for >20 min — corpus is small),
FastAPI, pytest. Keys via `.env` (documented in `.env.example`).

## 5. Work split

- **Claude**: schema + storage layer, linker, retrieval, baseline eval, integration.
- **Codex**: ingestion workers (video/audio/frames/pdf/entities) against the
  schema, each with tests on small fixtures; PDF sourcing.
- Separate worktrees/branches per `AGENTS.md`; Claude integrates.

## 6. Out of scope / future

Neo4j + dedicated vector DB (Option B), diarization, image-region provenance,
temporal change queries, semantic event segmentation — list in README as
future improvements.

## 7. Risks

- Frame count/cost: hybrid sampling + dedupe targets 40–80 frames; vision cache
  makes reruns free.
- Whisper API 25 MB limit ⇒ low-bitrate mono first, silence-split fallback.
- Entity normalization quality drives cross-modal linking; keep an alias map.
- Claim extraction quality drives the demo query; review claims for the
  primary video by hand before the demo.

## 8. Review history

- 2026-08-22 Codex design review: added claims, `spoken_by`, frame validity
  windows, structured sources, typed-path expansion, flat-multimodal ablation,
  hybrid frame sampling, vision cache, frozen ingestion contract, and the cuts
  above. No objections to Option A or the work split.
