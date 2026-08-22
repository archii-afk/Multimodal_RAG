"""Ingestion contract shared by all workers and the evidence store.

This module is FROZEN (see docs/DECISIONS.md). Workers return an ``IngestBatch``
built from ``NodeDraft``/``EdgeDraft`` using batch-local ``ref`` strings; the
store resolves refs to database ids atomically and merges nodes that share a
``canonical_key``. Workers never touch the database.

Design: docs/superpowers/specs/2026-08-22-multimodal-rag-design.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class NodeKind(StrEnum):
    SOURCE = "source"
    TRANSCRIPT_SEGMENT = "transcript_segment"
    FRAME = "frame"
    OCR_BLOCK = "ocr_block"
    IMAGE = "image"
    PDF_CHUNK = "pdf_chunk"
    ENTITY = "entity"  # includes persons: canonical_key = "person:<name>"
    CLAIM = "claim"


class Modality(StrEnum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    DOCUMENT = "document"
    ENTITY = "entity"  # entity and claim nodes


class EdgeKind(StrEnum):
    PART_OF = "part_of"  # node -> source; ocr_block -> frame; image -> pdf_chunk
    NEXT = "next"  # segment -> following segment within a source
    CO_OCCURS_AT = "co_occurs_at"  # transcript_segment <-> frame; weight = overlap seconds
    SPOKEN_BY = "spoken_by"  # transcript_segment -> person entity
    MENTIONS = "mentions"  # segment / pdf_chunk / ocr_block -> entity
    DEPICTS = "depicts"  # frame / image -> entity
    EXPRESSES = "expresses"  # transcript_segment -> claim
    ILLUSTRATES = "illustrates"  # frame / image -> claim
    SUPPORTS = "supports"  # pdf_chunk -> claim
    INVOLVES = "involves"  # claim -> entity
    SAME_TOPIC = "same_topic"  # cross-source only; weight = similarity


# Method-level confidence defaults used for ranking. LLM self-reported
# confidence is stored separately as ``model_confidence`` and is not trusted.
DEFAULT_CONFIDENCE: dict[str, float] = {
    "asr": 0.9,
    "ocr": 0.7,
    "vision": 0.75,
    "pdf": 1.0,
    "llm_claim": 0.7,
    "llm_entity": 0.8,
    "linker_time": 1.0,
    "linker_similarity": 0.6,
}


@dataclass(frozen=True)
class NodeDraft:
    """A node as produced by a worker, before it has a database id.

    ``ref`` must be unique within the batch (e.g. ``"seg:0042"``). Source nodes
    are referenced by ``source_ref``; the store derives ``part_of`` edges from it.
    For frames, ``t_start``/``t_end`` is the validity window, not the sample
    instant; put the sample instant in ``provenance["sampled_at"]``.
    """

    ref: str
    kind: NodeKind
    modality: Modality
    content: str
    source_ref: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    page: int | None = None  # 1-based
    bbox: tuple[float, float, float, float] | None = None
    speaker: str | None = None
    confidence: float = 1.0
    model_confidence: float | None = None
    canonical_key: str | None = None  # required for entity/claim merging
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("ref must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.t_start is not None and self.t_end is not None and self.t_end < self.t_start:
            raise ValueError(f"t_end < t_start for {self.ref}")
        if self.kind in (NodeKind.ENTITY, NodeKind.CLAIM) and not self.canonical_key:
            raise ValueError(f"{self.kind} node {self.ref} requires canonical_key")
        if self.kind is not NodeKind.SOURCE and self.source_ref is None:
            raise ValueError(f"{self.kind} node {self.ref} requires source_ref")


@dataclass(frozen=True)
class SourceDraft(NodeDraft):
    """Convenience for source nodes with structured file metadata.

    ``content`` holds the human title. Use ``kind=NodeKind.SOURCE``.
    """

    path: str = ""
    mime_type: str = ""
    sha256: str = ""
    duration: float | None = None
    presenter: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not NodeKind.SOURCE:
            raise ValueError("SourceDraft must have kind=SOURCE")


@dataclass(frozen=True)
class EdgeDraft:
    src_ref: str
    dst_ref: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestBatch:
    """Output of every worker. Refs in edges must resolve within this batch or
    to a ``canonical_key`` already known to the store (e.g. an entity)."""

    nodes: tuple[NodeDraft, ...]
    edges: tuple[EdgeDraft, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        refs = [n.ref for n in self.nodes]
        dupes = {r for r in refs if refs.count(r) > 1}
        if dupes:
            raise ValueError(f"duplicate refs in batch: {sorted(dupes)}")
        known = set(refs)
        for e in self.edges:
            for r in (e.src_ref, e.dst_ref):
                if r not in known and not r.startswith("key:"):
                    raise ValueError(
                        f"edge {e.kind} references unknown ref {r!r}; "
                        "use a batch ref or 'key:<canonical_key>'"
                    )

    def merge(self, other: IngestBatch) -> IngestBatch:
        return IngestBatch(self.nodes + other.nodes, self.edges + other.edges, self.warnings + other.warnings)


@dataclass(frozen=True)
class InsertResult:
    ref_to_node_id: dict[str, str]


class EvidenceStore(Protocol):
    def insert_batch(self, batch: IngestBatch) -> InsertResult: ...


def canonical_entity_key(name: str) -> str:
    """Deterministic key for entity merging: lowercase, collapse whitespace,
    strip punctuation except hyphens. Aliases are resolved by the linker."""
    import re

    s = re.sub(r"[^\w\s-]", "", name.lower())
    return "entity:" + re.sub(r"\s+", " ", s).strip()


def person_key(name: str) -> str:
    return "person:" + " ".join(name.lower().split())
