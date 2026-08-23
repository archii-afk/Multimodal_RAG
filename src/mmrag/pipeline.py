"""End-to-end ingestion: files → workers → store → linkers → embeddings.

Workers are the frozen functions from the contract (spec §3.3), implemented in
``mmrag.ingest.*`` by the ingestion branch. They are injected so the pipeline
is testable without media tools or API keys.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .embeddings import Embedder, VectorIndex
from .linker import link_claims_to_frames, link_same_topic, link_time_overlap
from .model import IngestBatch
from .store import SQLiteStore

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
PDF_EXT = {".pdf"}


@dataclass
class Workers:
    ingest_audio: Callable[..., IngestBatch]
    sample_video_frames: Callable[..., IngestBatch]
    ingest_frames: Callable[..., IngestBatch]
    ingest_pdf: Callable[..., IngestBatch]
    extract_claims: Callable[..., IngestBatch]

    @classmethod
    def default(cls) -> "Workers":
        from .ingest.audio import ingest_audio
        from .ingest.claims import extract_claims
        from .ingest.pdf import ingest_pdf
        from .ingest.video import sample_video_frames
        from .ingest.vision import ingest_frames

        return cls(ingest_audio, sample_video_frames, ingest_frames, ingest_pdf, extract_claims)


def source_ref_for(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return f"src:{slug}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]}"


class Pipeline:
    def __init__(self, store: SQLiteStore, embedder: Embedder, workers: Workers | None = None, *,
                 vision_model: str = "gemini-2.5-flash", llm_model: str = "gpt-4.1-mini",
                 same_topic_threshold: float = 0.55, frames_per_minute: float = 3.0,
                 log: Callable[[str], None] = print):
        self.store, self.embedder = store, embedder
        self.workers = workers or Workers.default()
        self.vision_model, self.llm_model = vision_model, llm_model
        self.same_topic_threshold, self.log = same_topic_threshold, log
        self.frames_per_minute = frames_per_minute

    def ingest(self, paths: list[Path], *, presenter: str) -> dict:
        w = self.workers
        report = {"sources": 0, "nodes": 0, "edges": 0, "co_occurs_at": 0, "illustrates": 0, "same_topic": 0, "embedded": 0}
        source_ids: list[str] = []
        for path in paths:
            ref = source_ref_for(path)
            ext = path.suffix.lower()
            self.log(f"[ingest] {path.name} as {ref}")
            if ext in VIDEO_EXT or ext in AUDIO_EXT:
                batch = w.ingest_audio(path, source_ref=ref, presenter=presenter)
                if ext in VIDEO_EXT:
                    duration = next((n.duration for n in batch.nodes if getattr(n, "duration", None)), None) or 0.0
                    # Cap scales with length; the sampler guarantees >=1 frame / 30 s, so never go below that.
                    max_frames = max(80, int(duration / 60 * self.frames_per_minute) + 1, int(duration / 30) + 2)
                    frames = w.sample_video_frames(path, source_ref=ref, max_frames=max_frames)
                    self.log(f"[frames] {len(frames.nodes)} sampled")
                    frames = w.ingest_frames(frames, model=self.vision_model)
                    batch = batch.merge(frames)
            elif ext in PDF_EXT:
                batch = w.ingest_pdf(path, source_ref=ref)
            else:
                self.log(f"[skip] unsupported {path.name}")
                continue
            # extract_claims returns the enriched full batch (its edges reference input refs).
            batch = w.extract_claims(batch, model=self.llm_model)
            res = self.store.insert_batch(batch)
            for wmsg in batch.warnings:
                self.log(f"[warn] {wmsg}")
            report["sources"] += 1
            report["nodes"] += len(batch.nodes)
            report["edges"] += len(batch.edges)
            source_ids.append(res.ref_to_node_id[ref])
        for sid in source_ids:
            report["co_occurs_at"] += link_time_overlap(self.store, sid)
        report["illustrates"] += link_claims_to_frames(self.store)
        index = VectorIndex(self.store, self.embedder)
        report["embedded"] = index.embed_missing()
        report["same_topic"] = link_same_topic(self.store, index, threshold=self.same_topic_threshold)
        self.log(f"[done] {report}")
        return report
