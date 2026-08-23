"""Post-ingestion linking: relationships that need the whole store.

- ``link_time_overlap``: transcript_segment <-> frame overlap within a source → co_occurs_at (weight = seconds).
- ``link_claims_to_frames``: a frame illustrates a claim when the frame depicts an entity the claim
  involves AND the frame co-occurs with a segment that expresses the claim.
- ``link_same_topic``: cross-source embedding similarity (see embeddings.py); capped per node.
"""

from __future__ import annotations

from .model import EdgeKind, NodeKind
from .store import SQLiteStore


def link_time_overlap(store: SQLiteStore, source_id: str) -> int:
    frames = [n for n in store.nodes_by_kind(NodeKind.FRAME) if n.source_id == source_id]
    count = 0
    with store.conn:
        for f in frames:
            if f.t_start is None or f.t_end is None:
                continue
            for seg in store.nodes_overlapping(source_id, f.t_start, f.t_end, kind=NodeKind.TRANSCRIPT_SEGMENT):
                overlap = min(f.t_end, seg.t_end) - max(f.t_start, seg.t_start)
                if overlap <= 0:
                    continue
                store._insert_edge(seg.id, f.id, EdgeKind.CO_OCCURS_AT, overlap, {"linker": "time_overlap"})
                count += 1
    return count


def link_claims_to_frames(store: SQLiteStore) -> int:
    count = 0
    with store.conn:
        for claim in store.nodes_by_kind(NodeKind.CLAIM):
            claim_entities = {n.id for n, _ in store.neighbors(claim.id, [EdgeKind.INVOLVES])}
            if not claim_entities:
                continue
            segments = [n for n, _ in store.neighbors(claim.id, [EdgeKind.EXPRESSES], direction="in")]
            candidate_frames: dict[str, float] = {}
            for seg in segments:
                for frame, e in store.neighbors(seg.id, [EdgeKind.CO_OCCURS_AT]):
                    candidate_frames[frame.id] = max(candidate_frames.get(frame.id, 0.0), e.weight)
            for fid, overlap in candidate_frames.items():
                depicted = {n.id for n, _ in store.neighbors(fid, [EdgeKind.DEPICTS])}
                shared = claim_entities & depicted
                if shared:
                    store._insert_edge(fid, claim.id, EdgeKind.ILLUSTRATES, float(len(shared)),
                                       {"linker": "claim_frame", "overlap_s": overlap, "shared_entities": len(shared)})
                    count += 1
    return count


def link_same_topic(store: SQLiteStore, index, threshold: float = 0.55, cap: int = 3) -> int:
    """Cross-source similarity edges between content nodes, top-``cap`` per node above ``threshold``."""
    import numpy as np

    if index._mat is None:
        index._load()
    ids, mat = index._ids, index._mat
    if not ids:
        return 0
    meta = {r["id"]: (r["source_id"], r["kind"]) for r in store.conn.execute(
        "SELECT id, source_id, kind FROM nodes WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)}
    linkable = {NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK, NodeKind.FRAME, NodeKind.OCR_BLOCK, NodeKind.IMAGE}
    sims = mat @ mat.T
    count = 0
    with store.conn:
        for i, nid in enumerate(ids):
            src_i, kind_i = meta[nid]
            if NodeKind(kind_i) not in linkable:
                continue
            order = np.argsort(-sims[i])
            added = 0
            for j in order:
                if added >= cap or sims[i, j] < threshold:
                    break
                oid = ids[j]
                if oid == nid:
                    continue
                src_j, kind_j = meta[oid]
                if src_j == src_i or NodeKind(kind_j) not in linkable:
                    continue
                store._insert_edge(nid, oid, EdgeKind.SAME_TOPIC, float(sims[i, j]),
                                   {"linker": "same_topic", "model": index.embedder.name, "threshold": threshold})
                added += 1
                count += 1
    return count
