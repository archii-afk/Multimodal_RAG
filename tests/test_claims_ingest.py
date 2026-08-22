from __future__ import annotations

from mmrag.ingest import claims
from mmrag.model import EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind


def test_extract_claims_deduplicates_claims_entities_and_uses_disk_cache(tmp_path, monkeypatch):
    batch = IngestBatch(
        (
            NodeDraft(
                ref="segment:0000",
                kind=NodeKind.TRANSCRIPT_SEGMENT,
                modality=Modality.AUDIO,
                content="Read replicas reduce load on Jira's primary database.",
                source_ref="source:video",
            ),
            NodeDraft(
                ref="pdf:0000",
                kind=NodeKind.PDF_CHUNK,
                modality=Modality.DOCUMENT,
                content="Jira uses read replicas to reduce load on its primary database.",
                source_ref="source:pdf",
                page=1,
            ),
        )
    )
    monkeypatch.setattr(claims, "CLAIMS_CACHE_ROOT", tmp_path / "cache")
    calls = []

    def fake_request(items, *, model):
        calls.append(items)
        return [
            {
                "evidence_ref": node["ref"],
                "statement": "Read replicas reduce load on Jira's primary database.",
                "entities": ["Jira", "Read Replicas"],
                "model_confidence": 0.88,
            }
            for node in items
        ]

    monkeypatch.setattr(claims, "_claim_request", fake_request)
    first = claims.extract_claims(batch, model="gpt-test")
    second = claims.extract_claims(batch, model="gpt-test")

    assert first == second
    assert len(calls) == 1
    assert first.nodes[: len(batch.nodes)] == batch.nodes
    assert len([node for node in first.nodes if node.kind is NodeKind.CLAIM]) == 1
    assert len([node for node in first.nodes if node.kind is NodeKind.ENTITY]) == 2
    edge_kinds = [edge.kind for edge in first.edges]
    assert edge_kinds.count(EdgeKind.EXPRESSES) == 1
    assert edge_kinds.count(EdgeKind.SUPPORTS) == 1
    assert edge_kinds.count(EdgeKind.INVOLVES) == 2
    assert edge_kinds.count(EdgeKind.MENTIONS) == 4
    assert all(
        edge.dst_ref.startswith("key:")
        for edge in first.edges
        if edge.kind in (EdgeKind.EXPRESSES, EdgeKind.SUPPORTS, EdgeKind.MENTIONS)
    )
