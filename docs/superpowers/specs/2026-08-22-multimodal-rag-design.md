# Multimodal RAG — design spec (v1, hackathon)

Date: 2026-08-22 · Status: Accepted by user · Brief: `multimodal_data_management_hackathon.pdf`

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
- Secondary: one short (3–10 min) architecture video by Archisman, for
  cross-file linking.
- Documents: public Atlassian engineering blog posts / architecture docs
  exported to PDF, chosen to overlap the systems drawn in the video.
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
| id           | uuid |
| kind         | `transcript_segment` · `frame` · `ocr_block` · `image` · `pdf_chunk` · `entity` · `source` |
| modality     | `audio` · `video` · `image` · `document` · `entity` |
| content      | text: transcript text, OCR text, vision description, PDF chunk, entity name |
| source_id    | FK → source node (file) |
| t_start, t_end | seconds (media) or null |
| page / bbox  | PDF page, image region (provenance to region, stretch) |
| speaker      | presenter label (single speaker per video; no diarization) |
| confidence   | 0–1: ASR avg logprob, OCR confidence, LLM self-reported, or 1.0 |
| provenance   | JSON: extractor name + version/model, raw file path, frame path |
| embedding_id | FK into vector index |

### 3.2 Edge model (`edges`: src, dst, kind, weight, provenance)

| kind           | meaning |
|----------------|---------|
| `co_occurs_at` | transcript_segment ↔ frame/ocr_block overlapping in time |
| `mentions`     | transcript_segment / pdf_chunk / ocr_block → entity |
| `depicts`      | frame / image → entity (from vision description) |
| `same_topic`   | cross-source link via embedding similarity above threshold |
| `part_of`      | node → source; ocr_block → frame |
| `next`         | temporal ordering of segments within a source |

Entities are normalized (lowercased canonical name + aliases) so the same
system ("Tenant Context Service", "TCS") links across video, OCR, and PDF.

### 3.3 Ingestion workers (each a module with a pure function `ingest(path) -> nodes, edges`)

- **video**: `ffmpeg` → audio track; scene-change frame sampling (ffmpeg
  `select='gt(scene,0.3)'`, capped + min-gap) → frames with timestamps.
- **audio**: OpenAI Whisper API with segment timestamps → transcript_segments.
- **frames/images**: Gemini vision, one call per frame returning
  `{description, ocr_text, entities, is_diagram, confidence}`.
- **pdf**: PyMuPDF → page text chunks (+ embedded images as image nodes).
- **entities**: LLM pass over segments/OCR/chunks → entity nodes + `mentions`.
- **linker**: time-overlap → `co_occurs_at`; embedding similarity → `same_topic`.

### 3.4 Retrieval

1. Embed query; vector search across all node kinds (not only text).
2. Optional entity match from query → seed entity nodes.
3. Expand 1–2 hops along `co_occurs_at`, `mentions`, `depicts`, `same_topic`.
4. Score = similarity × confidence (stretch: confidence weighting), group into
   an **evidence bundle**: answer text + list of evidence items each with
   modality, source, timestamp/page, frame thumbnail path.
5. LLM composes final answer citing evidence ids.

### 3.5 Baseline (deliverable)

Same store, restricted to transcript + pdf nodes, no graph expansion. Same
query set; compare whether required multimodal evidence is retrieved
(recall@k over a hand-labelled ~10-question set).

### 3.6 Interface

CLI (`ingest`, `query`) + minimal FastAPI page showing the evidence bundle
with frame thumbnails and timestamps. No auth, no polish.

## 4. Stack

Python 3.11+, `ffmpeg`, OpenAI (Whisper, embeddings, chat), Gemini (vision),
PyMuPDF, SQLite + `sqlite-vec` (fallback: numpy brute-force — corpus is small),
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

- 40-min video ⇒ ~hundreds of frames; cap vision calls (~150) via scene
  threshold + min gap.
- Whisper API 25 MB limit ⇒ split audio into chunks, offset timestamps.
- Entity normalization quality drives cross-modal linking; keep an alias map.
