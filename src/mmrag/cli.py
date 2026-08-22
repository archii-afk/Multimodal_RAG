"""mmrag CLI: ingest | query | eval | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from .answer import compose_answer, location
from .embeddings import HashEmbedder, OpenAIEmbedder, VectorIndex
from .evaluation import evaluate, format_report, load_questions
from .retrieval import Bundle, Mode, Retriever
from .store import SQLiteStore

DEFAULT_DB = Path("data/processed/evidence.db")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmrag")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--offline", action="store_true", help="hash embedder, no LLM (tests/smoke)")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest"); i.add_argument("paths", nargs="+", type=Path); i.add_argument("--presenter", required=True)
    q = sub.add_parser("query"); q.add_argument("text"); q.add_argument("--mode", choices=[m.value for m in Mode], default="graph")
    q.add_argument("-k", type=int, default=8); q.add_argument("--no-answer", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("questions", type=Path); e.add_argument("-k", type=int, default=8)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    return p


def _embedder(offline: bool):
    return HashEmbedder() if offline else OpenAIEmbedder()


def bundle_to_dict(b: Bundle) -> dict:
    return {
        "query": b.query, "mode": str(b.mode), "speakers": b.speakers,
        "claims": [c.content for c in b.claims],
        "evidence": [{
            "id": f"E{i}", "node_id": e.node.id, "kind": str(e.node.kind), "modality": str(e.node.modality),
            "location": location(e), "source": os.path.basename(e.source.path or "") if e.source else None,
            "t_start": e.node.t_start, "t_end": e.node.t_end, "page": e.node.page,
            "score": round(e.score, 4), "similarity": round(e.similarity, 4), "confidence": e.node.confidence,
            "content": e.node.content, "thumbnail": e.node.provenance.get("frame_path") or e.node.provenance.get("path"),
            "path": [f"{'->' if p.src != e.node.id else '<-'}{p.kind}" for p in e.path],
        } for i, e in enumerate(b.evidence, start=1)],
    }


def run_query(store: SQLiteStore, index: VectorIndex, text: str, *, mode: str, k: int,
              llm: Callable[[str, str], str] | None) -> dict:
    bundle = Retriever(store, index).retrieve(text, mode=Mode(mode), k=k)
    out = bundle_to_dict(bundle)
    out["answer"] = compose_answer(bundle, llm) if llm else None
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        from dotenv import load_dotenv
        load_dotenv()
    store = SQLiteStore(args.db)
    if args.cmd == "ingest":
        from .pipeline import Pipeline
        Pipeline(store, _embedder(args.offline)).ingest(args.paths, presenter=args.presenter)
        return 0
    index = VectorIndex(store, _embedder(args.offline))
    if args.cmd == "query":
        llm = None if (args.offline or args.no_answer) else __import__("mmrag.answer", fromlist=["openai_llm"]).openai_llm()
        out = run_query(store, index, args.text, mode=args.mode, k=args.k, llm=llm)
        print(json.dumps(out, indent=2))
        return 0
    if args.cmd == "eval":
        r = Retriever(store, index)
        report = evaluate(load_questions(args.questions), lambda t, m, k: r.retrieve(t, mode=m, k=k), k=args.k)
        print(format_report(report)); print(json.dumps(report, indent=2), file=sys.stderr)
        return 0
    if args.cmd == "serve":
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(store, index, offline=args.offline), host="127.0.0.1", port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
