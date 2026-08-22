import pytest

from mmrag.model import (
    EdgeDraft, EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind,
    SourceDraft, canonical_entity_key, person_key,
)


def _src():
    return SourceDraft(ref="src:video", kind=NodeKind.SOURCE, modality=Modality.VIDEO,
                       content="Atlassian architecture talk", path="data/raw/talk.mp4",
                       mime_type="video/mp4", sha256="abc", duration=2400.0, presenter="Ex-Atlassian engineer")


def test_batch_with_segment_frame_and_cooccurrence_edge():
    seg = NodeDraft(ref="seg:1", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO,
                    content="we added read replicas", source_ref="src:video", t_start=720.0, t_end=760.0,
                    speaker="Ex-Atlassian engineer", confidence=0.9)
    frame = NodeDraft(ref="frame:3", kind=NodeKind.FRAME, modality=Modality.VIDEO, content="",
                      source_ref="src:video", t_start=730.0, t_end=745.0,
                      provenance={"sampled_at": 737.0, "path": "data/processed/frames/3.jpg"})
    batch = IngestBatch(nodes=(_src(), seg, frame),
                        edges=(EdgeDraft("seg:1", "frame:3", EdgeKind.CO_OCCURS_AT, weight=15.0),))
    assert len(batch.nodes) == 3


def test_edge_may_reference_known_canonical_key():
    seg = NodeDraft(ref="seg:1", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO,
                    content="x", source_ref="src:video")
    IngestBatch(nodes=(_src(), seg),
                edges=(EdgeDraft("seg:1", "key:" + person_key("Jane Doe"), EdgeKind.SPOKEN_BY),))


def test_unknown_ref_rejected():
    with pytest.raises(ValueError, match="unknown ref"):
        IngestBatch(nodes=(_src(),), edges=(EdgeDraft("src:video", "ghost", EdgeKind.PART_OF),))


def test_duplicate_refs_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        IngestBatch(nodes=(_src(), _src()))


def test_entity_requires_canonical_key():
    with pytest.raises(ValueError, match="canonical_key"):
        NodeDraft(ref="e:1", kind=NodeKind.ENTITY, modality=Modality.ENTITY, content="TCS", source_ref="src:video")


def test_non_source_requires_source_ref():
    with pytest.raises(ValueError, match="source_ref"):
        NodeDraft(ref="seg:1", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, content="x")


def test_canonical_keys_normalize():
    assert canonical_entity_key("Tenant  Context Service") == canonical_entity_key("tenant context service")
    assert canonical_entity_key("Read-Replica!") == "entity:read-replica"
    assert person_key("Jane  Doe") == "person:jane doe"


def test_merge_concatenates():
    a = IngestBatch(nodes=(_src(),))
    b = IngestBatch(nodes=(NodeDraft(ref="seg:1", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO,
                                     content="x", source_ref="src:video"),), warnings=("w",))
    m = a.merge(b)
    assert [n.ref for n in m.nodes] == ["src:video", "seg:1"] and m.warnings == ("w",)
