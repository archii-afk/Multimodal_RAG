import json

from mmrag.cli import build_parser, run_query
from mmrag.embeddings import HashEmbedder, VectorIndex
from mmrag.model import IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft
from mmrag.store import SQLiteStore


def test_parser_has_subcommands():
    p = build_parser()
    assert p.parse_args(["ingest", "a.mp4", "--presenter", "Jane"]).cmd == "ingest"
    assert p.parse_args(["query", "what?", "--mode", "text_only"]).mode == "text_only"
    assert p.parse_args(["eval", "q.json"]).cmd == "eval"
    assert p.parse_args(["serve"]).cmd == "serve"


def test_run_query_returns_json_bundle(tmp_path):
    store = SQLiteStore(tmp_path / "g.db")
    store.insert_batch(IngestBatch(nodes=(
        SourceDraft(ref="s", kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="t", path="t.mp4", mime_type="v", sha256="1"),
        NodeDraft(ref="x", kind=NodeKind.TRANSCRIPT_SEGMENT, modality=Modality.AUDIO, content="replica", source_ref="s", t_start=1, t_end=2))))
    idx = VectorIndex(store, HashEmbedder(16)); idx.embed_missing()
    out = run_query(store, idx, "replica", mode="graph", k=3, llm=None)
    assert out["evidence"][0]["kind"] == "transcript_segment" and out["evidence"][0]["location"] == "t.mp4 @ 00:01-00:02"
    json.dumps(out)
