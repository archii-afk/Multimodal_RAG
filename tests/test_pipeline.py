from pathlib import Path

from mmrag.embeddings import HashEmbedder
from mmrag.model import EdgeDraft, EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft
from mmrag.pipeline import Pipeline, Workers
from mmrag.store import SQLiteStore


def fake_audio(path, *, source_ref, presenter):
    src = SourceDraft(ref=source_ref, kind=NodeKind.SOURCE, modality=Modality.VIDEO, content=path.name,
                      path=str(path), mime_type="video/mp4", sha256="sha-" + path.name, presenter=presenter)
    seg = NodeDraft(ref=f"{source_ref}/seg:0", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO,
                    content="read replicas reduce load", source_ref=source_ref, t_start=0, t_end=10, speaker=presenter)
    return IngestBatch(nodes=(src, seg))


def fake_frames(path, *, source_ref, **kw):
    f = NodeDraft(ref=f"{source_ref}/frame:0", kind=NodeKind.FRAME, modality=Modality.VIDEO, content="",
                  source_ref=source_ref, t_start=2, t_end=8, provenance={"sampled_at": 5})
    return IngestBatch(nodes=(f,))


def fake_vision(batch, *, model):
    nodes = tuple(NodeDraft(**{**n.__dict__, "content": "diagram of replicas"}) if n.kind is NodeKind.FRAME else n
                  for n in batch.nodes)
    return IngestBatch(nodes=nodes, edges=batch.edges)


def fake_pdf(path, *, source_ref, **kw):
    src = SourceDraft(ref=source_ref, kind=NodeKind.SOURCE, modality=Modality.DOCUMENT, content=path.name,
                      path=str(path), mime_type="application/pdf", sha256="sha-" + path.name)
    c = NodeDraft(ref=f"{source_ref}/chunk:0", kind=NodeKind.PDF_CHUNK, modality=Modality.DOCUMENT,
                  content="replicas", source_ref=source_ref, page=1)
    return IngestBatch(nodes=(src, c))


def fake_claims(batch, *, model):
    # Contract: returns the enriched FULL batch (claim edges reference input refs).
    claim = NodeDraft(ref="claim:0", kind=NodeKind.CLAIM, modality=Modality.ENTITY, content="replicas reduce load",
                      source_ref=batch.nodes[0].ref, canonical_key="claim:replicas reduce load")
    seg = next((n for n in batch.nodes if n.kind is NodeKind.TRANSCRIPT_SEGMENT), None)
    edges = batch.edges + ((EdgeDraft(seg.ref, "claim:0", EdgeKind.EXPRESSES),) if seg else ())
    return IngestBatch(batch.nodes + (claim,), edges)


def test_pipeline_ingests_video_and_pdf_then_links_and_embeds(tmp_path):
    (tmp_path / "talk.mp4").write_bytes(b"x"); (tmp_path / "doc.pdf").write_bytes(b"x")
    store = SQLiteStore(tmp_path / "g.db")
    p = Pipeline(store, HashEmbedder(32), Workers(fake_audio, fake_frames, fake_vision, fake_pdf, fake_claims),
                 vision_model="m", llm_model="m")
    report = p.ingest([tmp_path / "talk.mp4", tmp_path / "doc.pdf"], presenter="Jane")
    assert report["sources"] == 2 and report["co_occurs_at"] == 1 and report["embedded"] == 4  # seg, frame, chunk, claim
    frames = store.nodes_by_kind(NodeKind.FRAME)
    assert frames[0].content == "diagram of replicas"  # vision ran on sampled frames before insert
    assert len(store.nodes_by_kind(NodeKind.CLAIM)) == 1  # claims merged once, no duplicate refs
