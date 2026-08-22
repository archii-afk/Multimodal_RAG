import pytest

from mmrag.model import (
    EdgeDraft, EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft,
    canonical_entity_key, person_key,
)
from mmrag.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "g.db")


def _src(ref="src:v"):
    return SourceDraft(ref=ref, kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="talk",
                       path="data/raw/talk.mp4", mime_type="video/mp4", sha256="abc", duration=10.0,
                       presenter="Jane")


def _seg(ref, t0, t1, src="src:v", content="text"):
    return NodeDraft(ref=ref, kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO,
                     content=content, source_ref=src, t_start=t0, t_end=t1, speaker="Jane", confidence=0.9)


def test_insert_returns_ids_and_nodes_are_readable(store):
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("seg:1", 0, 5))))
    assert set(res.ref_to_node_id) == {"src:v", "seg:1"}
    node = store.get_node(res.ref_to_node_id["seg:1"])
    assert node.kind == NodeKind.TRANSCRIPT_SEGMENT and node.t_end == 5 and node.speaker == "Jane"


def test_part_of_edges_derived_from_source_ref(store):
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("seg:1", 0, 5))))
    edges = store.edges_from(res.ref_to_node_id["seg:1"], EdgeKind.PART_OF)
    assert [e.dst for e in edges] == [res.ref_to_node_id["src:v"]]


def test_key_reference_creates_entity_once_and_merges(store):
    jane = "key:" + person_key("Jane")
    b1 = IngestBatch(nodes=(_src(), _seg("seg:1", 0, 5)),
                     edges=(EdgeDraft("seg:1", jane, EdgeKind.SPOKEN_BY),))
    b2 = IngestBatch(nodes=(_src("src:w"), _seg("seg:2", 0, 5, src="src:w")),
                     edges=(EdgeDraft("seg:2", jane, EdgeKind.SPOKEN_BY),))
    r1 = store.insert_batch(b1)
    r2 = store.insert_batch(b2)
    persons = store.find_by_canonical_key(person_key("Jane"))
    assert len(persons) == 1
    spoken = store.edges_to(persons[0].id, EdgeKind.SPOKEN_BY)
    assert {e.src for e in spoken} == {r1.ref_to_node_id["seg:1"], r2.ref_to_node_id["seg:2"]}


def test_entity_node_with_same_canonical_key_merges_into_existing(store):
    k = canonical_entity_key("Tenant Context Service")
    e1 = NodeDraft(ref="e:1", kind=NodeKind.ENTITY, modality=Modality.ENTITY, content="Tenant Context Service",
                   source_ref="src:v", canonical_key=k)
    e2 = NodeDraft(ref="e:2", kind=NodeKind.ENTITY, modality=Modality.ENTITY, content="TCS",
                   source_ref="src:v", canonical_key=k)
    r1 = store.insert_batch(IngestBatch(nodes=(_src(), e1)))
    r2 = store.insert_batch(IngestBatch(nodes=(_src(), e2)))
    assert r1.ref_to_node_id["e:1"] == r2.ref_to_node_id["e:2"]
    assert len(store.find_by_canonical_key(k)) == 1


def test_source_reinsert_is_idempotent_by_sha256(store):
    r1 = store.insert_batch(IngestBatch(nodes=(_src(),)))
    r2 = store.insert_batch(IngestBatch(nodes=(_src(),)))
    assert r1.ref_to_node_id["src:v"] == r2.ref_to_node_id["src:v"]


def test_unresolvable_key_for_non_entity_kinds_is_rejected(store):
    bad = IngestBatch(nodes=(_src(), _seg("seg:1", 0, 5)),
                      edges=(EdgeDraft("seg:1", "key:nope", EdgeKind.MENTIONS),))
    with pytest.raises(ValueError, match="key:nope"):
        store.insert_batch(bad)


def test_nodes_in_time_range(store):
    frame = NodeDraft(ref="f:1", kind=NodeKind.FRAME, modality=Modality.VIDEO, content="diagram",
                      source_ref="src:v", t_start=3.0, t_end=8.0)
    res = store.insert_batch(IngestBatch(nodes=(_src(), _seg("seg:1", 0, 5), _seg("seg:2", 6, 9), frame)))
    hits = store.nodes_overlapping(res.ref_to_node_id["src:v"], 4.0, 7.0, kind=NodeKind.TRANSCRIPT_SEGMENT)
    assert sorted(n.t_start for n in hits) == [0.0, 6.0]


def test_neighbors_follow_typed_edges(store):
    k = canonical_entity_key("read replica")
    claim = NodeDraft(ref="c:1", kind=NodeKind.CLAIM, modality=Modality.ENTITY, source_ref="src:v",
                      content="read replicas reduce primary db load", canonical_key="claim:read replicas reduce load")
    res = store.insert_batch(IngestBatch(
        nodes=(_src(), _seg("seg:1", 0, 5), claim),
        edges=(EdgeDraft("seg:1", "c:1", EdgeKind.EXPRESSES), EdgeDraft("c:1", "key:" + k, EdgeKind.INVOLVES))))
    nbrs = store.neighbors(res.ref_to_node_id["seg:1"], [EdgeKind.EXPRESSES])
    assert [n.id for n, e in nbrs] == [res.ref_to_node_id["c:1"]]
    nbrs2 = store.neighbors(res.ref_to_node_id["c:1"], [EdgeKind.INVOLVES])
    assert nbrs2[0][0].canonical_key == k
