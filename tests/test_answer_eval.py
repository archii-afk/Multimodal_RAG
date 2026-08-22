import json

from mmrag.answer import compose_answer, evidence_as_context
from mmrag.evaluation import Question, evaluate, matches
from mmrag.model import Modality, NodeKind
from mmrag.retrieval import Bundle, Evidence, Mode
from mmrag.store import Node


def _node(id, kind, content, t0=None, t1=None, page=None, src="srcv"):
    return Node(id=id, kind=kind, modality=Modality.AUDIO, content=content, source_id=src, t_start=t0, t_end=t1,
                page=page, bbox=None, speaker=None, confidence=1.0, model_confidence=None, canonical_key=None,
                provenance={})


def _src(id="srcv", path="v.mp4"):
    return Node(id=id, kind=NodeKind.SOURCE, modality=Modality.VIDEO, content="talk", source_id=None, t_start=None,
                t_end=None, page=None, bbox=None, speaker=None, confidence=1.0, model_confidence=None,
                canonical_key=None, provenance={}, path=path)


def test_context_lists_evidence_with_ids_and_locations():
    ev = [Evidence(_node("n1", NodeKind.TRANSCRIPT_SEGMENT, "hello", 12, 15), _src(), 0.9, 0.9),
          Evidence(_node("n2", NodeKind.PDF_CHUNK, "doc text", page=3, src="srcp"), _src("srcp", "p.pdf"), 0.5, 0.5)]
    ctx = evidence_as_context(ev)
    assert "[E1] transcript_segment v.mp4 @ 00:12-00:15" in ctx and "[E2] pdf_chunk p.pdf p.3" in ctx


def test_compose_answer_uses_injected_llm():
    calls = []
    def llm(system, user):
        calls.append(user); return "Read replicas [E1]."
    b = Bundle("q", Mode.GRAPH, [Evidence(_node("n1", NodeKind.TRANSCRIPT_SEGMENT, "replicas", 0, 1), _src(), 1, 1)],
               speakers=["Jane"])
    out = compose_answer(b, llm)
    assert out == "Read replicas [E1]." and "Jane" in calls[0]


def test_matches_by_source_kind_and_time_window():
    ev = Evidence(_node("n1", NodeKind.FRAME, "", 100, 110), _src(path="data/raw/talk.mp4"), 1, 1)
    assert matches(ev, {"source": "talk.mp4", "kind": "frame", "t": 105})
    assert not matches(ev, {"source": "talk.mp4", "kind": "frame", "t": 200})
    assert not matches(ev, {"source": "other.mp4", "kind": "frame", "t": 105})


def test_evaluate_reports_recall_per_mode():
    q = Question(id="q1", text="x", required=[{"source": "v.mp4", "kind": "transcript_segment", "t": 1},
                                               {"source": "v.mp4", "kind": "frame", "t": 1}])
    seg = Evidence(_node("n1", NodeKind.TRANSCRIPT_SEGMENT, "a", 0, 2), _src(), 1, 1)
    frame = Evidence(_node("n2", NodeKind.FRAME, "", 0, 2), _src(), 1, 1)
    def retrieve(text, mode, k):
        return Bundle(text, mode, [seg] if mode is Mode.TEXT_ONLY else [seg, frame])
    rep = evaluate([q], retrieve, modes=[Mode.TEXT_ONLY, Mode.GRAPH], k=5)
    assert rep["text_only"]["recall"] == 0.5 and rep["graph"]["recall"] == 1.0
    assert rep["graph"]["per_question"]["q1"]["found"] == 2
