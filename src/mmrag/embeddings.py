"""Embeddings and a brute-force vector index over store nodes.

The corpus is small (thousands of nodes), so vectors live in an ``embeddings``
table and search is a numpy dot product. ``Embedder`` is injectable:
``OpenAIEmbedder`` for real runs, ``HashEmbedder`` for offline tests.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .model import NodeKind
from .store import SQLiteStore

EMBED_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
  node_id TEXT PRIMARY KEY REFERENCES nodes(id),
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);
"""

NON_EMBEDDED_KINDS = (NodeKind.SOURCE,)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, dim), unit-norm rows


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedding for tests. Not semantic."""

    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client=None):
        self.name = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            chunk = [t if t.strip() else " " for t in texts[i : i + 256]]
            resp = self.client.embeddings.create(model=self.name, input=chunk)
            vecs += [d.embedding for d in resp.data]
        return _normalize(np.asarray(vecs, dtype=np.float32))


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@dataclass(frozen=True)
class Hit:
    node_id: str
    score: float


class VectorIndex:
    def __init__(self, store: SQLiteStore, embedder: Embedder):
        self.store, self.embedder = store, embedder
        store.conn.executescript(EMBED_SCHEMA)
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def embed_missing(self) -> int:
        kinds_q = ",".join("?" * len(NON_EMBEDDED_KINDS))
        rows = self.store.conn.execute(
            f"SELECT id, content FROM nodes WHERE kind NOT IN ({kinds_q}) "
            "AND id NOT IN (SELECT node_id FROM embeddings)", NON_EMBEDDED_KINDS).fetchall()
        rows = [r for r in rows if r["content"].strip()]
        if not rows:
            return 0
        vecs = self.embedder.embed([r["content"] for r in rows])
        with self.store.conn:
            self.store.conn.executemany(
                "INSERT INTO embeddings (node_id, model, dim, vec) VALUES (?,?,?,?)",
                [(r["id"], self.embedder.name, vecs.shape[1], vecs[i].tobytes()) for i, r in enumerate(rows)])
        self._mat = None
        return len(rows)

    def _load(self) -> None:
        rows = self.store.conn.execute("SELECT node_id, dim, vec FROM embeddings").fetchall()
        self._ids = [r["node_id"] for r in rows]
        self._mat = (np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
                     if rows else np.zeros((0, 1), dtype=np.float32))

    def search(self, query: str, k: int = 10, kinds: list[NodeKind] | None = None,
               modalities: list[str] | None = None) -> list[Hit]:
        if self._mat is None:
            self._load()
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        scores = self._mat @ q
        allowed: set[str] | None = None
        if kinds or modalities:
            sql, args = "SELECT id FROM nodes WHERE 1=1", []
            if kinds:
                sql += f" AND kind IN ({','.join('?' * len(kinds))})"; args += list(kinds)
            if modalities:
                sql += f" AND modality IN ({','.join('?' * len(modalities))})"; args += list(modalities)
            allowed = {r["id"] for r in self.store.conn.execute(sql, args)}
        order = np.argsort(-scores)
        hits: list[Hit] = []
        for i in order:
            nid = self._ids[i]
            if allowed is not None and nid not in allowed:
                continue
            hits.append(Hit(nid, float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]
