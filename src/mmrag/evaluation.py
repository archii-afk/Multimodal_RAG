"""Retrieval evaluation: recall of hand-labelled required evidence per run mode.

Question file (JSON list): {"id", "text", "required": [{"source": "<basename>", "kind": "<node kind>",
"t": <seconds> | "page": <n>}]}. An evidence item matches when source basename and kind agree and the
time (within the node's window ±tolerance) or page matches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .retrieval import Bundle, Evidence, Mode

TOLERANCE_S = 5.0


@dataclass
class Question:
    id: str
    text: str
    required: list[dict]


def load_questions(path: Path) -> list[Question]:
    return [Question(**q) for q in json.loads(Path(path).read_text())]


def matches(e: Evidence, req: dict) -> bool:
    src = os.path.basename(e.source.path or "") if e.source else ""
    if src != req["source"] or str(e.node.kind) != req["kind"]:
        return False
    if "page" in req:
        return e.node.page == req["page"]
    if "t" in req:
        if e.node.t_start is None or e.node.t_end is None:
            return False
        return e.node.t_start - TOLERANCE_S <= req["t"] <= e.node.t_end + TOLERANCE_S
    return True


def evaluate(questions: list[Question], retrieve: Callable[[str, Mode, int], Bundle],
             modes: list[Mode] | None = None, k: int = 8) -> dict:
    modes = modes or list(Mode)
    report: dict = {}
    for mode in modes:
        per_q, found_total, req_total = {}, 0, 0
        for q in questions:
            bundle = retrieve(q.text, mode, k)
            found = sum(1 for req in q.required if any(matches(e, req) for e in bundle.evidence))
            per_q[q.id] = {"found": found, "required": len(q.required),
                           "kinds_returned": sorted({str(e.node.kind) for e in bundle.evidence})}
            found_total += found
            req_total += len(q.required)
        report[str(mode)] = {"recall": found_total / req_total if req_total else 0.0, "per_question": per_q}
    return report


def format_report(report: dict) -> str:
    lines = [f"{'mode':<18}{'recall':>8}"]
    for mode, r in report.items():
        lines.append(f"{mode:<18}{r['recall']:>8.2f}")
    return "\n".join(lines)
