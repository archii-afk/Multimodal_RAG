"""Minimal FastAPI demo: one query form, evidence cards with thumbnails. No auth, no polish."""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .cli import run_query
from .embeddings import VectorIndex
from .retrieval import Mode
from .store import SQLiteStore

PROCESSED = Path("data/processed")

PAGE = """<!doctype html><title>mmrag</title>
<style>
body{font:15px/1.4 system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#222}
form{display:flex;gap:.5rem;margin-bottom:1rem}input[type=text]{flex:1;padding:.5rem;font-size:1rem}
.answer{background:#f3f6ff;border:1px solid #cbd5ff;padding:1rem;border-radius:6px;white-space:pre-wrap;margin-bottom:1rem}
.meta{color:#555;font-size:.9rem;margin-bottom:1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.8rem}
.card{border:1px solid #ddd;border-radius:6px;padding:.7rem;background:#fff}
.card img{max-width:100%;border-radius:4px;margin:.4rem 0}
.kind{font-weight:600;text-transform:uppercase;font-size:.75rem;letter-spacing:.04em}
.kind.frame,.kind.image{color:#b35c00}.kind.transcript_segment{color:#0b6}.kind.pdf_chunk{color:#36c}.kind.ocr_block{color:#a0a}
.loc{color:#555;font-size:.85rem}.path{color:#888;font-size:.8rem}.content{margin-top:.4rem;max-height:9rem;overflow:auto}
</style>
<h2>Multimodal evidence search</h2>
<form method=get>
<input type=text name=q value="{{q}}" placeholder="What architecture was discussed for reducing database load, who explained it, and where was the diagram shown?">
<select name=mode>{{modes}}</select><button>Search</button></form>
{{body}}"""


def _render(out: dict | None, q: str, mode: str) -> str:
    modes = "".join(f'<option value="{m.value}"{" selected" if m.value == mode else ""}>{m.value}</option>' for m in Mode)
    if not out:
        return _fill(q, modes, "")
    parts = []
    if out.get("answer"):
        parts.append(f'<div class=answer>{html.escape(out["answer"])}</div>')
    meta = f"mode: {out['mode']} · speakers: {', '.join(out['speakers']) or '—'}"
    if out["claims"]:
        meta += " · claims: " + " | ".join(html.escape(c) for c in out["claims"][:4])
    parts.append(f"<div class=meta>{meta}</div><div class=cards>")
    for e in out["evidence"]:
        thumb = ""
        if e["thumbnail"] and e["kind"] in ("frame", "image"):
            rel = Path(e["thumbnail"])
            try:
                rel = rel.resolve().relative_to(PROCESSED.resolve())
                thumb = f'<img src="/processed/{rel.as_posix()}" alt="frame">'
            except ValueError:
                pass
        path = " ".join(html.escape(p) for p in e["path"]) or "direct hit"
        parts.append(
            f'<div class=card><span class="kind {e["kind"]}">{e["kind"]}</span> <b>{e["id"]}</b> '
            f'<span class=loc>{html.escape(e["location"])}</span> <span class=path>· score {e["score"]} · {path}</span>'
            f'{thumb}<div class=content>{html.escape(e["content"][:600])}</div></div>')
    parts.append("</div>")
    return _fill(q, modes, "".join(parts))


def _fill(q: str, modes: str, body: str) -> str:
    return PAGE.replace("{{q}}", html.escape(q)).replace("{{modes}}", modes).replace("{{body}}", body)


def create_app(store: SQLiteStore, index: VectorIndex, *, offline: bool = False) -> FastAPI:
    app = FastAPI(title="mmrag")
    llm = None
    if not offline:
        from .answer import openai_llm
        llm = openai_llm()
    if PROCESSED.exists():
        app.mount("/processed", StaticFiles(directory=str(PROCESSED)), name="processed")

    @app.get("/api/query")
    def api_query(q: str, mode: str = Query("graph"), k: int = 8, answer: bool = True):
        return run_query(store, index, q, mode=mode, k=k, llm=llm if answer else None)

    @app.get("/", response_class=HTMLResponse)
    def page(q: str = "", mode: str = "graph", k: int = 8):
        out = run_query(store, index, q, mode=mode, k=k, llm=llm) if q.strip() else None
        return _render(out, q, mode)

    return app
