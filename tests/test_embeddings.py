import hashlib

import numpy as np
import pytest

from mmrag.embeddings import HashEmbedder, VectorIndex
from mmrag.model import IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft
from mmrag.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "g.db")


def _src():
    return SourceDraft(ref="src:v", kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="talk",
                       path="x.mp4", mime_type="video/mp4", sha256="s1")


def _seg(ref, text):
    return NodeDraft(ref=ref, kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, content=text,
                     source_ref="src:v", t_start=0, t_end=1)


def test_hash_embedder_is_deterministic_and_unit_norm():
    e = HashEmbedder(dim=32)
    a, b = e.embed(["read replica"])[0], e.embed(["read replica"])[0]
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-6


def test_index_embeds_only_unembedded_nodes_and_searches(store):
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("s1", "read replica reduces database load"),
                                                _seg("s2", "lunch menu"))))
    emb = HashEmbedder(dim=64)
    idx = VectorIndex(store, emb)
    assert idx.embed_missing() == 2
    assert idx.embed_missing() == 0
    hits = idx.search("read replica database load", k=2)
    assert hits[0].node_id == res.ref_to_node_id["s1"] and hits[0].score > hits[1].score


def test_search_can_filter_by_kind(store):
    store.insert_batch(IngestBatch(nodes=(_src(), _seg("s1", "a"))))
    idx = VectorIndex(store, HashEmbedder(dim=16))
    idx.embed_missing()
    assert idx.search("a", k=5, kinds=[NodeKind.FRAME]) == []
    assert len(idx.search("a", k=5, kinds=[NodeKind.TRANSCRIPT_SEGMENT])) == 1


def test_source_nodes_are_not_embedded(store):
    store.insert_batch(IngestBatch(nodes=(_src(),)))
    assert VectorIndex(store, HashEmbedder(dim=16)).embed_missing() == 0
