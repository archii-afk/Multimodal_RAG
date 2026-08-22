from __future__ import annotations

from PIL import Image

from mmrag.ingest import vision
from mmrag.model import EdgeKind, IngestBatch, Modality, NodeDraft, NodeKind


def _frame(ref, path, start):
    return NodeDraft(
        ref=ref,
        kind=NodeKind.FRAME,
        modality=Modality.VIDEO,
        content="",
        source_ref="source:video",
        t_start=start,
        t_end=start + 5,
        provenance={"frame_path": str(path), "sampled_at": start + 2.5},
    )


def test_ingest_frames_enriches_frames_and_caches_each_api_call(tmp_path, monkeypatch):
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    Image.new("RGB", (16, 16), "red").save(first_path)
    Image.new("RGB", (16, 16), "blue").save(second_path)
    batch = IngestBatch((_frame("frame:0000", first_path, 0), _frame("frame:0001", second_path, 5)))
    monkeypatch.setattr(vision, "VISION_CACHE_ROOT", tmp_path / "cache")
    calls = []

    def fake_request(path, *, model):
        calls.append(path)
        return {
            "description": "A Jira database read-replica diagram.",
            "ocr_text": "Jira Read Replica",
            "entities": ["Jira", "Read Replica", "Jira"],
            "is_diagram": True,
            "model_confidence": 0.93,
        }

    monkeypatch.setattr(vision, "_gemini_request", fake_request)
    first = vision.ingest_frames(batch, model="gemini-test")
    second = vision.ingest_frames(batch, model="gemini-test")

    assert first == second
    assert len(calls) == 2
    frames = [node for node in first.nodes if node.kind is NodeKind.FRAME]
    assert all(node.content.startswith("A Jira") for node in frames)
    assert all(node.model_confidence == 0.93 for node in frames)
    assert len([node for node in first.nodes if node.kind is NodeKind.OCR_BLOCK]) == 2
    assert len([node for node in first.nodes if node.kind is NodeKind.ENTITY]) == 2
    assert len([edge for edge in first.edges if edge.kind is EdgeKind.PART_OF]) == 2
    assert len([edge for edge in first.edges if edge.kind is EdgeKind.DEPICTS]) == 4
    assert len([edge for edge in first.edges if edge.kind is EdgeKind.MENTIONS]) == 4
