"""Cached batched claim and entity extraction from textual evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from mmrag.model import (
    DEFAULT_CONFIDENCE,
    EdgeDraft,
    EdgeKind,
    IngestBatch,
    Modality,
    NodeDraft,
    NodeKind,
    canonical_entity_key,
)

PROMPT_VERSION = "v1"
CLAIMS_CACHE_ROOT = Path("data/processed/cache/claims")
MAX_ITEMS_PER_CALL = 40
MAX_ATTEMPTS = 4

SYSTEM_PROMPT = """Extract explicit, retrieval-useful factual claims from the supplied evidence.
Do not add outside knowledge. Return JSON with a `claims` array. Each claim must contain:
- evidence_ref: exactly one supplied ref
- statement: a standalone proposition
- entities: named people, products, systems, services, or organizations involved
- model_confidence: 0 to 1
Return no claim for evidence that contains no factual proposition.
"""


def extract_claims(batch: IngestBatch, *, model: str) -> IngestBatch:
    """Append normalized claims/entities and their evidence edges to ``batch``."""
    if not model.strip():
        raise ValueError("model must be non-empty")
    evidence = [
        node
        for node in batch.nodes
        if node.kind in (NodeKind.TRANSCRIPT_SEGMENT, NodeKind.PDF_CHUNK) and node.content.strip()
    ]
    raw_claims: list[dict[str, Any]] = []
    for start in range(0, len(evidence), MAX_ITEMS_PER_CALL):
        raw_claims.extend(_extract_cached(evidence[start : start + MAX_ITEMS_PER_CALL], model))

    evidence_by_ref = {node.ref: node for node in evidence}
    existing_keys = {node.canonical_key for node in batch.nodes if node.canonical_key}
    claim_nodes: dict[str, NodeDraft] = {}
    entity_nodes: dict[str, NodeDraft] = {}
    edges = list(batch.edges)
    warnings = list(batch.warnings)
    seen_edges: set[tuple[str, str, EdgeKind]] = {
        (edge.src_ref, edge.dst_ref, edge.kind) for edge in edges
    }

    for raw_claim in raw_claims:
        evidence_ref = str(raw_claim.get("evidence_ref", ""))
        evidence_node = evidence_by_ref.get(evidence_ref)
        if evidence_node is None:
            warnings.append(f"claim extraction returned unknown evidence ref: {evidence_ref!r}")
            continue
        statement = str(raw_claim.get("statement", "")).strip()
        canonical_key = _canonical_claim_key(statement)
        if canonical_key == "claim:":
            continue
        confidence = _bounded_confidence(raw_claim.get("model_confidence", 0.0))
        claim_ref = f"claim:{hashlib.sha256(canonical_key.encode()).hexdigest()[:16]}"
        if canonical_key not in existing_keys and canonical_key not in claim_nodes:
            claim_nodes[canonical_key] = NodeDraft(
                ref=claim_ref,
                kind=NodeKind.CLAIM,
                modality=Modality.ENTITY,
                content=statement,
                source_ref=evidence_node.source_ref,
                confidence=DEFAULT_CONFIDENCE["llm_claim"],
                model_confidence=confidence,
                canonical_key=canonical_key,
                provenance={
                    "extractor": "openai_claim_extraction",
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                },
            )
        evidence_edge_kind = (
            EdgeKind.EXPRESSES
            if evidence_node.kind is NodeKind.TRANSCRIPT_SEGMENT
            else EdgeKind.SUPPORTS
        )
        _append_edge(
            edges,
            seen_edges,
            evidence_ref,
            f"key:{canonical_key}",
            evidence_edge_kind,
        )

        for raw_entity in raw_claim.get("entities", []):
            entity_name = str(raw_entity).strip()
            entity_key = canonical_entity_key(entity_name)
            if entity_key == "entity:":
                continue
            entity_ref = f"entity:claim:{hashlib.sha256(entity_key.encode()).hexdigest()[:16]}"
            if entity_key not in existing_keys and entity_key not in entity_nodes:
                entity_nodes[entity_key] = NodeDraft(
                    ref=entity_ref,
                    kind=NodeKind.ENTITY,
                    modality=Modality.ENTITY,
                    content=entity_name,
                    source_ref=evidence_node.source_ref,
                    confidence=DEFAULT_CONFIDENCE["llm_entity"],
                    model_confidence=confidence,
                    canonical_key=entity_key,
                    provenance={
                        "extractor": "openai_claim_extraction",
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                    },
                )
            _append_edge(edges, seen_edges, f"key:{canonical_key}", f"key:{entity_key}", EdgeKind.INVOLVES)
            _append_edge(edges, seen_edges, evidence_ref, f"key:{entity_key}", EdgeKind.MENTIONS)

    nodes = batch.nodes + tuple(claim_nodes.values()) + tuple(entity_nodes.values())
    return IngestBatch(nodes, tuple(edges), tuple(warnings))


def _extract_cached(evidence: list[NodeDraft], model: str) -> list[dict[str, Any]]:
    request_items = [
        {"ref": node.ref, "kind": node.kind.value, "content": node.content}
        for node in evidence
    ]
    request_bytes = json.dumps(request_items, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(request_bytes).hexdigest()
    model_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    cache_path = CLAIMS_CACHE_ROOT / f"{digest}-{model_key}-{PROMPT_VERSION}.json"
    if cache_path.exists():
        return list(json.loads(cache_path.read_text())["claims"])

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            claims = _claim_request(request_items, model=model)
            payload = {"claims": claims}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(cache_path)
            return claims
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _claim_request(items: list[dict[str, str]], *, model: str) -> list[dict[str, Any]]:
    """Perform one OpenAI API call; isolated for keyless tests."""
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "claim_extraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "evidence_ref": {"type": "string"},
                                    "statement": {"type": "string"},
                                    "entities": {"type": "array", "items": {"type": "string"}},
                                    "model_confidence": {"type": "number"},
                                },
                                "required": ["evidence_ref", "statement", "entities", "model_confidence"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["claims"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("claim model returned an empty response")
    return list(json.loads(content)["claims"])


def _append_edge(
    edges: list[EdgeDraft],
    seen: set[tuple[str, str, EdgeKind]],
    src_ref: str,
    dst_ref: str,
    kind: EdgeKind,
) -> None:
    key = (src_ref, dst_ref, kind)
    if key not in seen:
        seen.add(key)
        edges.append(
            EdgeDraft(
                src_ref,
                dst_ref,
                kind,
                provenance={"extractor": "claim_extraction", "prompt_version": PROMPT_VERSION},
            )
        )


def _canonical_claim_key(statement: str) -> str:
    normalized = re.sub(r"[^\w\s-]", "", statement.casefold())
    return "claim:" + re.sub(r"\s+", " ", normalized).strip()


def _bounded_confidence(value: object) -> float:
    return max(0.0, min(1.0, float(value)))
