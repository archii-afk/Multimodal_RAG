import pytest

from mmrag.embeddings import HashEmbedder, VectorIndex
from mmrag.linker import link_claims_to_frames, link_time_overlap
from mmrag.model import EdgeDraft, EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft, canonical_entity_key, person_key
from mmrag.retrieval import Mode, Retriever
from mmrag.store import SQLiteStore


@pytest.fixture
def world(tmp_path):
    """Talk video + PDF. Segment s1 (0-10s) states the replica claim while frame f1 (5-15s) shows it;
    pdf chunk p1 supports it. Distractor frame f2 and segment s2."""
    store = SQLiteStore(tmp_path / "g.db")
    k = canonical_entity_key("read replica")
    video = SourceDraft(ref="src:v", kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="talk",
                        path="v.mp4", mime_type="video/mp4", sha256="v", presenter="Jane")
    pdf = SourceDraft(ref="src:p", kind=NodeKind.SOURCE, modality=Modality.DOCUMENT, content="atlassian blog",
                      path="p.pdf", mime_type="application/pdf", sha256="p")
    s1 = NodeDraft(ref="s1", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, source_ref="src:v",
                   content="we added read replicas to reduce database load", t_start=0, t_end=10, speaker="Jane")
    s2 = NodeDraft(ref="s2", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, source_ref="src:v",
                   content="the office coffee machine", t_start=20, t_end=30, speaker="Jane")
    f1 = NodeDraft(ref="f1", kind=NodeKind.FRAME, modality=Modality.VIDEO, source_ref="src:v",
                   content="diagram: primary postgres with two read replica boxes", t_start=5, t_end=15,
                   provenance={"frame_path": "f1.jpg"})
    f2 = NodeDraft(ref="f2", kind=NodeKind.FRAME, modality=Modality.VIDEO, source_ref="src:v",
                   content="slide: agenda", t_start=40, t_end=50)
    p1 = NodeDraft(ref="p1", kind=NodeKind.PDF_CHUNK, modality=Modality.DOCUMENT, source_ref="src:p",
                   content="Atlassian scaled Jira by routing read queries to replicas", page=3)
    claim = NodeDraft(ref="c1", kind=NodeKind.CLAIM, modality=Modality.ENTITY, source_ref="src:v",
                      content="read replicas reduce primary database load", canonical_key="claim:read replicas reduce load")
    batch = IngestBatch(
        nodes=(video, pdf, s1, s2, f1, f2, p1, claim),
        edges=(EdgeDraft("s1", "key:" + person_key("Jane"), EdgeKind.SPOKEN_BY),
               EdgeDraft("s2", "key:" + person_key("Jane"), EdgeKind.SPOKEN_BY),
               EdgeDraft("s1", "c1", EdgeKind.EXPRESSES), EdgeDraft("p1", "c1", EdgeKind.SUPPORTS),
               EdgeDraft("c1", "key:" + k, EdgeKind.INVOLVES), EdgeDraft("f1", "key:" + k, EdgeKind.DEPICTS)))
    res = store.insert_batch(batch)
    link_time_overlap(store, res.ref_to_node_id["src:v"])
    link_claims_to_frames(store)
    idx = VectorIndex(store, HashEmbedder(dim=128))
    idx.embed_missing()
    return store, idx, res.ref_to_node_id


QUERY = "what reduces database load and where was the diagram shown"


def _kinds(bundle):
    return {e.node.kind for e in bundle.evidence}


def test_text_only_baseline_never_returns_frames(world):
    store, idx, ids = world
    b = Retriever(store, idx).retrieve(QUERY, mode=Mode.TEXT_ONLY, k=5)
    assert NodeKind.FRAME not in _kinds(b)
    assert ids["s1"] in {e.node.id for e in b.evidence}


def test_flat_multimodal_returns_frames_by_similarity_without_paths(world):
    store, idx, ids = world
    b = Retriever(store, idx).retrieve(QUERY, mode=Mode.FLAT_MULTIMODAL, k=5)
    assert all(e.path == () for e in b.evidence)


def test_full_graph_connects_segment_frame_pdf_and_speaker(world):
    store, idx, ids = world
    b = Retriever(store, idx).retrieve(QUERY, mode=Mode.GRAPH, k=3)
    by_id = {e.node.id: e for e in b.evidence}
    assert ids["f1"] in by_id and ids["p1"] in by_id, "graph must pull the diagram and the document"
    assert ids["f2"] not in by_id, "distractor frame must not be pulled in"
    assert by_id[ids["f1"]].path and by_id[ids["f1"]].path[-1].kind == EdgeKind.ILLUSTRATES
    assert b.speakers == ["Jane"]


def test_evidence_items_carry_provenance(world):
    store, idx, ids = world
    b = Retriever(store, idx).retrieve(QUERY, mode=Mode.GRAPH, k=3)
    f1 = next(e for e in b.evidence if e.node.id == ids["f1"])
    assert f1.source.path == "v.mp4" and f1.node.t_start == 5 and f1.node.provenance["path"] == "f1.jpg"
