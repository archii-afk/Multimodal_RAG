"""Retrieval over the evidence graph with three run modes (spec §3.5–3.6).

- TEXT_ONLY: baseline — vector search over transcript + pdf nodes, no graph.
- FLAT_MULTIMODAL: ablation — vector search over all node kinds, no graph.
- GRAPH: vector seeds over all kinds, then expansion along typed paths only.

Typed paths (each is a list of (edge kind, direction) hops from a seed):
  segment -expresses-> claim <-illustrates- frame
  segment -expresses-> claim <-supports- pdf_chunk
  segment -co_occurs_at-> frame            (and reverse)
  segment -spoken_by-> person
  frame  -illustrates-> claim <-expresses- segment
  pdf    -supports->   claim <-expresses- segment
  *      -same_topic-> *                    (1 hop, capped)
Every evidence item records the path that reached it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .embeddings import VectorIndex
from .model import EdgeKind, Modality, NodeKind
from .store import Edge, Node, SQLiteStore

TEXT_KINDS = [NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK]
SEED_KINDS = [NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK, NodeKind.FRAME, NodeKind.OCR_BLOCK,
              NodeKind.IMAGE, NodeKind.CLAIM]
EVIDENCE_KINDS = {NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK, NodeKind.FRAME, NodeKind.OCR_BLOCK,
                  NodeKind.IMAGE}

# (edge kind, direction) hop sequences, keyed by the seed's node kind.
PATHS: dict[NodeKind, list[list[tuple[EdgeKind, str]]]] = {
    NodeKind.TRANSCRIPT_SEGMENT: [
        [(EdgeKind.EXPRESSES, "out"), (EdgeKind.ILLUSTRATES, "in")],
        [(EdgeKind.EXPRESSES, "out"), (EdgeKind.SUPPORTS, "in")],
        [(EdgeKind.CO_OCCURS_AT, "out")],
        [(EdgeKind.SAME_TOPIC, "both")],
    ],
    NodeKind.FRAME: [
        [(EdgeKind.ILLUSTRATES, "out"), (EdgeKind.EXPRESSES, "in")],
        [(EdgeKind.ILLUSTRATES, "out"), (EdgeKind.SUPPORTS, "in")],
        [(EdgeKind.CO_OCCURS_AT, "in")],
        [(EdgeKind.SAME_TOPIC, "both")],
    ],
    NodeKind.PDF_CHUNK: [
        [(EdgeKind.SUPPORTS, "out"), (EdgeKind.EXPRESSES, "in")],
        [(EdgeKind.SUPPORTS, "out"), (EdgeKind.ILLUSTRATES, "in")],
        [(EdgeKind.SAME_TOPIC, "both")],
    ],
    NodeKind.CLAIM: [
        [(EdgeKind.EXPRESSES, "in")], [(EdgeKind.ILLUSTRATES, "in")], [(EdgeKind.SUPPORTS, "in")],
    ],
    NodeKind.OCR_BLOCK: [[(EdgeKind.PART_OF, "out")]],
    NodeKind.IMAGE: [[(EdgeKind.PART_OF, "out")], [(EdgeKind.SAME_TOPIC, "both")]],
}
HOP_DECAY = 0.8
SAME_TOPIC_CAP = 3
CO_OCCURS_CAP = 2  # keep only the longest-overlapping neighbours per hop
# Seed groups: top-k is taken per group so every modality gets a chance to seed expansion.
SEED_GROUPS = [[NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK], [NodeKind.FRAME, NodeKind.OCR_BLOCK, NodeKind.IMAGE],
               [NodeKind.CLAIM]]


class Mode(StrEnum):
    TEXT_ONLY = "text_only"
    FLAT_MULTIMODAL = "flat_multimodal"
    GRAPH = "graph"


@dataclass
class Evidence:
    node: Node
    source: Node | None
    score: float
    similarity: float
    path: tuple[Edge, ...] = ()  # edges walked from the seed; () for direct hits
    seed_id: str | None = None


@dataclass
class Bundle:
    query: str
    mode: Mode
    evidence: list[Evidence]
    claims: list[Node] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self, store: SQLiteStore, index: VectorIndex):
        self.store, self.index = store, index

    def retrieve(self, query: str, mode: Mode = Mode.GRAPH, k: int = 8) -> Bundle:
        if mode is Mode.TEXT_ONLY:
            hits = self.index.search(query, k=k, kinds=TEXT_KINDS)
            return self._bundle(query, mode, {h.node_id: self._direct(h.node_id, h.score) for h in hits})
        if mode is Mode.FLAT_MULTIMODAL:
            hits = self.index.search(query, k=k, kinds=SEED_KINDS)
        else:
            hits = sorted({h.node_id: h for g in SEED_GROUPS for h in self.index.search(query, k=k, kinds=g)}.values(),
                          key=lambda h: -h.score)
        hits = [h for h in hits if h.score > 0]
        found: dict[str, Evidence] = {}
        for h in hits:
            node = self.store.get_node(h.node_id)
            if node.kind in EVIDENCE_KINDS:
                found[h.node_id] = self._direct(h.node_id, h.score)
            if mode is Mode.GRAPH:
                self._expand(node, h.score, found)
        return self._bundle(query, mode, found)

    # -- helpers ---------------------------------------------------------

    def _direct(self, node_id: str, sim: float) -> Evidence:
        node = self.store.get_node(node_id)
        return Evidence(node, self.store.source_of(node), sim * node.confidence, sim)

    def _expand(self, seed: Node, seed_sim: float, found: dict[str, Evidence]) -> None:
        for hops in PATHS.get(seed.kind, []):
            frontier: list[tuple[Node, tuple[Edge, ...]]] = [(seed, ())]
            for depth, (ekind, direction) in enumerate(hops, start=1):
                nxt: list[tuple[Node, tuple[Edge, ...]]] = []
                for node, path in frontier:
                    nbrs = self.store.neighbors(node.id, [ekind], direction=direction)
                    if ekind is EdgeKind.SAME_TOPIC:
                        nbrs = sorted(nbrs, key=lambda ne: -ne[1].weight)[:SAME_TOPIC_CAP]
                    elif ekind is EdgeKind.CO_OCCURS_AT:
                        nbrs = sorted(nbrs, key=lambda ne: -ne[1].weight)[:CO_OCCURS_CAP]
                    for n, e in nbrs:
                        if n.id == seed.id:
                            continue
                        nxt.append((n, path + (e,)))
                frontier = nxt
            for node, path in frontier:
                if node.kind not in EVIDENCE_KINDS:
                    continue
                score = seed_sim * (HOP_DECAY ** len(path)) * node.confidence
                if path and path[-1].kind is EdgeKind.CO_OCCURS_AT:
                    score *= min(1.0, 0.5 + path[-1].weight / 20.0)  # longer overlap ranks higher
                prev = found.get(node.id)
                if prev is None or score > prev.score:
                    found[node.id] = Evidence(node, self.store.source_of(node), score, seed_sim, path, seed.id)

    def _bundle(self, query: str, mode: Mode, found: dict[str, Evidence]) -> Bundle:
        evidence = sorted(found.values(), key=lambda e: -e.score)
        claims: dict[str, Node] = {}
        speakers: list[str] = []
        if mode is Mode.GRAPH:
            for ev in evidence:
                for c, _ in self.store.neighbors(ev.node.id, [EdgeKind.EXPRESSES, EdgeKind.ILLUSTRATES, EdgeKind.SUPPORTS]):
                    claims[c.id] = c
                if ev.node.kind is NodeKind.TRANSCRIPT_SEGMENT:
                    for p, _ in self.store.neighbors(ev.node.id, [EdgeKind.SPOKEN_BY]):
                        name = ev.node.speaker or p.content  # display name lives on the segment
                        if name not in speakers:
                            speakers.append(name)
        return Bundle(query, mode, evidence, list(claims.values()), speakers)
