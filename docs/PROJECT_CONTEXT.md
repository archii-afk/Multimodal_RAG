# Project context

## Objective

Build a solution for the multimodal data management hackathon described in `multimodal_data_management_hackathon.pdf`. Full design: `docs/superpowers/specs/2026-08-22-multimodal-rag-design.md`.

## Source of truth

The PDF is the authoritative brief. This document holds only what has been confirmed with the user.

## Confirmed requirements (from the brief, reviewed 2026-08-22)

- Ingest four modalities: video, audio, images, PDFs/documents; realistic data, not toys.
- Unified representation per item: content, modality, timestamp, source, entities, relationships, confidence, provenance.
- Preserve cross-modal + temporal links (speech ↔ visual evidence at the same moment).
- Retrieval across modalities with traceability to original sources.
- Deliverables: README, ingestion pipeline, structured representation, query interface, demo with ≥3 modalities where the answer needs >1 modality, evaluation vs text-only RAG baseline, architecture explanation, future improvements.
- Out of scope: training models, building ASR/OCR/vision/vector DB from scratch, auth, polished UI, every media format.

## Constraints

- Live 13-hour hackathon (started 2026-08-22). No prep time.
- Priority order: working demo → text-only baseline comparison → stretch goals.
- API keys available: OpenAI, Gemini (others on request). Keys live in `.env`, never in Git.
- Demo query to target: "What architecture was discussed for reducing database load, who explained it, and where was the corresponding diagram shown?"

## Dataset (confirmed)

- ~40-min talking video: ex-Atlassian senior engineer explaining Atlassian's AWS architecture with Excalidraw-style diagrams on screen (single speaker). Path: `data/raw/` (gitignored).
- One short (3–10 min) architecture video by Archisman for cross-file linking.
- Public Atlassian engineering blog posts / architecture docs, exported to PDF, chosen to overlap systems shown in the video. No private company files exist.

## Success criteria

- A query whose answer requires transcript + frame/OCR + PDF evidence returns all three, each with timestamp/page and source file.
- Baseline (transcript + PDF only, no graph) visibly misses evidence the full system retrieves.

## Open questions

- Which Atlassian public docs best overlap the video (decide after first transcript pass).
- Whether `sqlite-vec` installs cleanly; fallback is numpy brute-force search.
