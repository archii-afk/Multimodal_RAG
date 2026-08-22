import pytest

from mmrag.linker import link_claims_to_frames, link_time_overlap
from mmrag.model import EdgeDraft, EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft, canonical_entity_key
from mmrag.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "g.db")


def _src():
    return SourceDraft(ref="src:v", kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="talk",
                       path="x.mp4", mime_type="video/mp4", sha256="s1", duration=100.0, presenter="Jane")


def _seg(ref, t0, t1):
    return NodeDraft(ref=ref, kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, content=ref,
                     source_ref="src:v", t_start=t0, t_end=t1)


def _frame(ref, t0, t1):
    return NodeDraft(ref=ref, kind=NodeKind.FRAME, modality=Modality.VIDEO, content=ref,
                     source_ref="src:v", t_start=t0, t_end=t1)


def test_time_overlap_creates_weighted_cooccurs_edges(store):
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("s1", 0, 10), _seg("s2", 10, 20), _frame("f1", 5, 15))))
    n = link_time_overlap(store, res.ref_to_node_id["src:v"])
    assert n == 2
    edges = store.edges_from(res.ref_to_node_id["s1"], EdgeKind.CO_OCCURS_AT)
    assert len(edges) == 1 and edges[0].weight == 5.0 and edges[0].dst == res.ref_to_node_id["f1"]


def test_time_overlap_skips_touching_intervals(store):
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("s1", 0, 10), _frame("f1", 10, 15))))
    assert link_time_overlap(store, res.ref_to_node_id["src:v"]) == 0


def test_frame_illustrates_claim_when_entities_intersect_and_cooccurs(store):
    k = canonical_entity_key("read replica")
    claim = NodeDraft(ref="c1", kind=NodeKind.CLAIM, modality=Modality.ENTITY, content="replicas cut load",
                      source_ref="src:v", canonical_key="claim:replicas cut load")
    res = store.insert_batch(IngestBatch(
        nodes=(_src(), _seg("s1", 0, 10), _frame("f1", 5, 15), _frame("f2", 50, 60), claim),
        edges=(EdgeDraft("s1", "c1", EdgeKind.EXPRESSES), EdgeDraft("c1", "key:" + k, EdgeKind.INVOLVES),
               EdgeDraft("f1", "key:" + k, EdgeKind.DEPICTS), EdgeDraft("f2", "key:" + k, EdgeKind.DEPICTS))))
    link_time_overlap(store, res.ref_to_node_id["src:v"])
    n = link_claims_to_frames(store)
    assert n == 1  # f2 depicts the entity but is not on screen while the claim is spoken
    ill = store.edges_to(res.ref_to_node_id["c1"], EdgeKind.ILLUSTRATES)
    assert [e.src for e in ill] == [res.ref_to_node_id["f1"]]
