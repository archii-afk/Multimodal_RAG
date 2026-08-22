"""SQLite evidence store: nodes + edges + atomic batch insert.

Ref resolution rules (see model.py):
- batch-local refs map to new or merged node ids;
- ``key:<canonical_key>`` resolves to an existing node with that key, or creates
  a bare entity/claim node for it;
- nodes with a ``canonical_key`` merge into an existing node with the same key;
- source nodes merge by ``sha256``.
``part_of`` edges to the source are derived from ``source_ref``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .model import EdgeDraft, EdgeKind, IngestBatch, InsertResult, Modality, NodeDraft, NodeKind, SourceDraft

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  modality TEXT NOT NULL,
  content TEXT NOT NULL,
  source_id TEXT REFERENCES nodes(id),
  t_start REAL, t_end REAL,
  page INTEGER,
  bbox TEXT,
  speaker TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  model_confidence REAL,
  canonical_key TEXT UNIQUE,
  provenance TEXT NOT NULL DEFAULT '{}',
  -- source-only fields
  path TEXT, mime_type TEXT, sha256 TEXT UNIQUE, duration REAL, presenter TEXT
);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS nodes_source_time ON nodes(source_id, t_start, t_end);
CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL REFERENCES nodes(id),
  dst TEXT NOT NULL REFERENCES nodes(id),
  kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  provenance TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst, kind);
"""


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    modality: Modality
    content: str
    source_id: str | None
    t_start: float | None
    t_end: float | None
    page: int | None
    bbox: tuple[float, float, float, float] | None
    speaker: str | None
    confidence: float
    model_confidence: float | None
    canonical_key: str | None
    provenance: dict
    path: str | None = None
    mime_type: str | None = None
    sha256: str | None = None
    duration: float | None = None
    presenter: str | None = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    weight: float
    provenance: dict


def _row_to_node(r: sqlite3.Row) -> Node:
    return Node(
        id=r["id"], kind=NodeKind(r["kind"]), modality=Modality(r["modality"]), content=r["content"],
        source_id=r["source_id"], t_start=r["t_start"], t_end=r["t_end"], page=r["page"],
        bbox=tuple(json.loads(r["bbox"])) if r["bbox"] else None, speaker=r["speaker"],
        confidence=r["confidence"], model_confidence=r["model_confidence"], canonical_key=r["canonical_key"],
        provenance=json.loads(r["provenance"]), path=r["path"], mime_type=r["mime_type"], sha256=r["sha256"],
        duration=r["duration"], presenter=r["presenter"],
    )


def _row_to_edge(r: sqlite3.Row) -> Edge:
    return Edge(r["src"], r["dst"], EdgeKind(r["kind"]), r["weight"], json.loads(r["provenance"]))


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI runs sync handlers in a threadpool; access is serialized per request.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # ---- writes -----------------------------------------------------------

    def insert_batch(self, batch: IngestBatch) -> InsertResult:
        ref_to_id: dict[str, str] = {}
        with self.conn:  # one transaction
            # sources first so source_ref can resolve
            ordered = sorted(batch.nodes, key=lambda n: n.kind is not NodeKind.SOURCE)
            for n in ordered:
                ref_to_id[n.ref] = self._upsert_node(n, ref_to_id)
            for e in batch.edges:
                self._insert_edge(self._resolve(e.src_ref, ref_to_id), self._resolve(e.dst_ref, ref_to_id),
                                  e.kind, e.weight, e.provenance)
        return InsertResult(ref_to_id)

    def _upsert_node(self, n: NodeDraft, ref_to_id: dict[str, str]) -> str:
        existing = None
        if isinstance(n, SourceDraft) and n.sha256:
            existing = self.conn.execute("SELECT id FROM nodes WHERE sha256=?", (n.sha256,)).fetchone()
        elif n.canonical_key:
            existing = self.conn.execute("SELECT id FROM nodes WHERE canonical_key=?", (n.canonical_key,)).fetchone()
        if existing:
            return existing["id"]

        source_id = None
        if n.source_ref is not None:
            if n.source_ref not in ref_to_id:
                raise ValueError(f"node {n.ref}: source_ref {n.source_ref!r} not in batch")
            source_id = ref_to_id[n.source_ref]

        nid = str(uuid.uuid4())
        src_fields = (n.path, n.mime_type, n.sha256, n.duration, n.presenter) if isinstance(n, SourceDraft) else (None,) * 5
        self.conn.execute(
            "INSERT INTO nodes (id, kind, modality, content, source_id, t_start, t_end, page, bbox, speaker, "
            "confidence, model_confidence, canonical_key, provenance, path, mime_type, sha256, duration, presenter) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, n.kind, n.modality, n.content, source_id, n.t_start, n.t_end, n.page,
             json.dumps(n.bbox) if n.bbox else None, n.speaker, n.confidence, n.model_confidence,
             n.canonical_key, json.dumps(n.provenance), *src_fields),
        )
        if source_id is not None:
            self._insert_edge(nid, source_id, EdgeKind.PART_OF, 1.0, {"derived": "source_ref"})
        return nid

    def _resolve(self, ref: str, ref_to_id: dict[str, str]) -> str:
        if ref in ref_to_id:
            return ref_to_id[ref]
        if not ref.startswith("key:"):
            raise ValueError(f"unresolvable ref {ref!r}")
        key = ref[4:]
        row = self.conn.execute("SELECT id FROM nodes WHERE canonical_key=?", (key,)).fetchone()
        if row:
            return row["id"]
        prefix, _, name = key.partition(":")
        if prefix in ("entity", "person"):
            kind = NodeKind.ENTITY
        elif prefix == "claim":
            kind = NodeKind.CLAIM
        else:
            raise ValueError(f"cannot auto-create node for {ref!r}: unknown key prefix {prefix!r}")
        nid = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO nodes (id, kind, modality, content, canonical_key, provenance) VALUES (?,?,?,?,?,?)",
            (nid, kind, Modality.ENTITY, name, key, json.dumps({"auto_created_from": ref})),
        )
        ref_to_id[ref] = nid
        return nid

    def _insert_edge(self, src: str, dst: str, kind: EdgeKind, weight: float, prov: dict) -> None:
        self.conn.execute(
            "INSERT INTO edges (src, dst, kind, weight, provenance) VALUES (?,?,?,?,?) "
            "ON CONFLICT(src, dst, kind) DO UPDATE SET weight=excluded.weight, provenance=excluded.provenance",
            (src, dst, kind, weight, json.dumps(prov)),
        )

    # ---- reads ------------------------------------------------------------

    def get_node(self, node_id: str) -> Node:
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        return _row_to_node(row)

    def find_by_canonical_key(self, key: str) -> list[Node]:
        return [_row_to_node(r) for r in self.conn.execute("SELECT * FROM nodes WHERE canonical_key=?", (key,))]

    def nodes_by_kind(self, *kinds: NodeKind) -> list[Node]:
        q = ",".join("?" * len(kinds))
        return [_row_to_node(r) for r in self.conn.execute(f"SELECT * FROM nodes WHERE kind IN ({q})", kinds)]

    def edges_from(self, node_id: str, kind: EdgeKind | None = None) -> list[Edge]:
        sql, args = "SELECT * FROM edges WHERE src=?", [node_id]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [_row_to_edge(r) for r in self.conn.execute(sql, args)]

    def edges_to(self, node_id: str, kind: EdgeKind | None = None) -> list[Edge]:
        sql, args = "SELECT * FROM edges WHERE dst=?", [node_id]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [_row_to_edge(r) for r in self.conn.execute(sql, args)]

    def neighbors(self, node_id: str, kinds: list[EdgeKind], direction: str = "out") -> list[tuple[Node, Edge]]:
        """Nodes reachable by one edge of the given kinds. direction: out | in | both."""
        q = ",".join("?" * len(kinds))
        out: list[tuple[Node, Edge]] = []
        if direction in ("out", "both"):
            rows = self.conn.execute(
                f"SELECT n.*, e.src AS e_src, e.dst AS e_dst, e.kind AS e_kind, e.weight AS e_weight, "
                f"e.provenance AS e_prov FROM edges e JOIN nodes n ON n.id=e.dst WHERE e.src=? AND e.kind IN ({q})",
                [node_id, *kinds]).fetchall()
            out += [(_row_to_node(r), Edge(r["e_src"], r["e_dst"], EdgeKind(r["e_kind"]), r["e_weight"], json.loads(r["e_prov"]))) for r in rows]
        if direction in ("in", "both"):
            rows = self.conn.execute(
                f"SELECT n.*, e.src AS e_src, e.dst AS e_dst, e.kind AS e_kind, e.weight AS e_weight, "
                f"e.provenance AS e_prov FROM edges e JOIN nodes n ON n.id=e.src WHERE e.dst=? AND e.kind IN ({q})",
                [node_id, *kinds]).fetchall()
            out += [(_row_to_node(r), Edge(r["e_src"], r["e_dst"], EdgeKind(r["e_kind"]), r["e_weight"], json.loads(r["e_prov"]))) for r in rows]
        return out

    def nodes_overlapping(self, source_id: str, t_start: float, t_end: float, kind: NodeKind | None = None) -> list[Node]:
        sql = "SELECT * FROM nodes WHERE source_id=? AND t_start IS NOT NULL AND t_start < ? AND t_end > ?"
        args: list = [source_id, t_end, t_start]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [_row_to_node(r) for r in self.conn.execute(sql + " ORDER BY t_start", args)]

    def source_of(self, node: Node) -> Node | None:
        return self.get_node(node.source_id) if node.source_id else None

    def close(self) -> None:
        self.conn.close()
