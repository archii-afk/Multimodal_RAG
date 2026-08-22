"""Compose a cited answer from an evidence bundle. The LLM is injected as
``llm(system, user) -> str`` so tests run offline."""

from __future__ import annotations

import os
from typing import Callable

from .retrieval import Bundle, Evidence

SYSTEM = (
    "You answer questions using ONLY the provided evidence items. Cite evidence ids like [E2] after each "
    "claim. If the evidence includes a frame or image, say where it was shown (source file and timestamp). "
    "If a speaker is known, name them. If the evidence is insufficient, say so briefly."
)


def fmt_time(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def location(e: Evidence) -> str:
    name = os.path.basename(e.source.path) if e.source and e.source.path else (e.source.content if e.source else "?")
    if e.node.page is not None:
        return f"{name} p.{e.node.page}"
    if e.node.t_start is not None:
        return f"{name} @ {fmt_time(e.node.t_start)}-{fmt_time(e.node.t_end)}"
    return name


def evidence_as_context(evidence: list[Evidence]) -> str:
    lines = []
    for i, e in enumerate(evidence, start=1):
        text = e.node.content.strip().replace("\n", " ")
        lines.append(f"[E{i}] {e.node.kind} {location(e)}: {text[:700]}")
    return "\n".join(lines)


def compose_answer(bundle: Bundle, llm: Callable[[str, str], str]) -> str:
    parts = [f"Question: {bundle.query}"]
    if bundle.speakers:
        parts.append("Known speakers: " + ", ".join(bundle.speakers))
    if bundle.claims:
        parts.append("Extracted claims: " + " | ".join(c.content for c in bundle.claims[:5]))
    parts.append("Evidence:\n" + evidence_as_context(bundle.evidence))
    return llm(SYSTEM, "\n\n".join(parts))


def openai_llm(model: str = "gpt-4.1-mini") -> Callable[[str, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(system: str, user: str) -> str:
        r = client.chat.completions.create(model=model, temperature=0,
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""

    return call
