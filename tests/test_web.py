from fastapi.testclient import TestClient

from mmrag.embeddings import HashEmbedder, VectorIndex
from mmrag.model import IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft
from mmrag.store import SQLiteStore
from mmrag.web import create_app


def _client(tmp_path):
    store = SQLiteStore(tmp_path / "g.db")
    store.insert_batch(IngestBatch(nodes=(
        SourceDraft(ref="s", kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="t", path="t.mp4", mime_type="v", sha256="1"),
        NodeDraft(ref="x", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, content="replica", source_ref="s", t_start=1, t_end=2))))
    idx = VectorIndex(store, HashEmbedder(16)); idx.embed_missing()
    return TestClient(create_app(store, idx, offline=True))


def test_index_page_renders_form(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200 and "<form" in r.text


def test_api_query_returns_evidence(tmp_path):
    r = _client(tmp_path).get("/api/query", params={"q": "replica", "mode": "graph"})
    assert r.status_code == 200 and r.json()["evidence"][0]["kind"] == "transcript_segment"


def test_html_query_shows_evidence_cards(tmp_path):
    r = _client(tmp_path).get("/", params={"q": "replica", "mode": "text_only"})
    assert "transcript_segment" in r.text and "00:01-00:02" in r.text
