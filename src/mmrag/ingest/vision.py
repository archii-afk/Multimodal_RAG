"""Cached Gemini enrichment for sampled video frames."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
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
VISION_CACHE_ROOT = Path("data/processed/cache/vision")
MAX_CONCURRENCY = 3
MAX_ATTEMPTS = 4

VISION_PROMPT = """Analyze this frame as evidence from a technical presentation.
Return JSON with exactly these fields:
- description: concise factual visual description
- ocr_text: all legible text, preserving reading order
- entities: distinct named systems, services, products, organizations, and people
- is_diagram: boolean
- model_confidence: number from 0 to 1
Do not infer details that are not visible.
"""


def ingest_frames(batch: IngestBatch, *, model: str) -> IngestBatch:
    """Return ``batch`` with frame descriptions plus OCR/entity evidence."""
    if not model.strip():
        raise ValueError("model must be non-empty")
    frames = [node for node in batch.nodes if node.kind is NodeKind.FRAME]
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, max(1, len(frames)))) as executor:
        futures = {executor.submit(_analyze_frame_cached, frame, model): frame for frame in frames}
        for future in as_completed(futures):
            frame = futures[future]
            try:
                results[frame.ref] = future.result()
            except Exception as exc:  # Preserve successful cached work for a resumable rerun.
                failures[frame.ref] = str(exc)

    nodes: list[NodeDraft] = []
    edges = list(batch.edges)
    entity_nodes: dict[str, NodeDraft] = {}
    for node in batch.nodes:
        result = results.get(node.ref)
        if node.kind is not NodeKind.FRAME or result is None:
            nodes.append(node)
            continue
        frame_provenance = dict(node.provenance)
        frame_provenance.update(
            {
                "vision_extractor": "gemini",
                "vision_model": model,
                "vision_prompt_version": PROMPT_VERSION,
                "is_diagram": bool(result["is_diagram"]),
            }
        )
        nodes.append(
            replace(
                node,
                content=result["description"],
                model_confidence=result["model_confidence"],
                provenance=frame_provenance,
            )
        )

        ocr_text = result["ocr_text"].strip()
        ocr_ref: str | None = None
        if ocr_text:
            ocr_ref = f"{node.ref}:ocr"
            nodes.append(
                NodeDraft(
                    ref=ocr_ref,
                    kind=NodeKind.OCR_BLOCK,
                    modality=Modality.IMAGE,
                    content=ocr_text,
                    source_ref=node.source_ref,
                    t_start=node.t_start,
                    t_end=node.t_end,
                    confidence=DEFAULT_CONFIDENCE["ocr"],
                    model_confidence=result["model_confidence"],
                    provenance={
                        "extractor": "gemini_vision_ocr",
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "frame_ref": node.ref,
                        "frame_path": node.provenance["frame_path"],
                    },
                )
            )
            edges.append(EdgeDraft(ocr_ref, node.ref, EdgeKind.PART_OF))

        for entity_name in result["entities"]:
            canonical_key = canonical_entity_key(entity_name)
            if canonical_key == "entity:":
                continue
            if canonical_key not in entity_nodes:
                short_hash = hashlib.sha256(canonical_key.encode()).hexdigest()[:16]
                entity_nodes[canonical_key] = NodeDraft(
                    ref=f"entity:vision:{short_hash}",
                    kind=NodeKind.ENTITY,
                    modality=Modality.ENTITY,
                    content=entity_name.strip(),
                    source_ref=node.source_ref,
                    confidence=DEFAULT_CONFIDENCE["llm_entity"],
                    model_confidence=result["model_confidence"],
                    canonical_key=canonical_key,
                    provenance={
                        "extractor": "gemini_vision",
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                    },
                )
            edges.append(EdgeDraft(node.ref, f"key:{canonical_key}", EdgeKind.DEPICTS))
            if ocr_ref and entity_name.casefold() in ocr_text.casefold():
                edges.append(EdgeDraft(ocr_ref, f"key:{canonical_key}", EdgeKind.MENTIONS))

    nodes.extend(entity_nodes.values())
    warnings = batch.warnings + tuple(
        f"vision analysis failed for {ref}: {message}" for ref, message in sorted(failures.items())
    )
    return IngestBatch(tuple(nodes), tuple(edges), warnings)


def _analyze_frame_cached(frame: NodeDraft, model: str) -> dict[str, Any]:
    frame_path = Path(str(frame.provenance.get("frame_path", "")))
    if not frame_path.is_file():
        raise FileNotFoundError(f"frame path for {frame.ref}: {frame_path}")
    digest = _sha256(frame_path)
    model_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    cache_path = VISION_CACHE_ROOT / f"{digest}-{model_key}-{PROMPT_VERSION}.json"
    if cache_path.exists():
        return _normalize_result(json.loads(cache_path.read_text()))
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = _normalize_result(_gemini_request(frame_path, model=model))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            temporary.replace(cache_path)
            return result
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _gemini_request(frame_path: Path, *, model: str) -> dict[str, Any]:
    """Perform one Gemini API call; isolated for keyless tests."""
    from google import genai
    from google.genai import types

    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
    response = genai.Client().models.generate_content(
        model=model,
        contents=[
            VISION_PROMPT,
            types.Part.from_bytes(data=frame_path.read_bytes(), mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "ocr_text": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "is_diagram": {"type": "boolean"},
                    "model_confidence": {"type": "number"},
                },
                "required": ["description", "ocr_text", "entities", "is_diagram", "model_confidence"],
            },
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    return json.loads(response.text)


def _normalize_result(value: dict[str, Any]) -> dict[str, Any]:
    confidence = max(0.0, min(1.0, float(value.get("model_confidence", 0.0))))
    seen: set[str] = set()
    entities: list[str] = []
    for raw_name in value.get("entities", []):
        name = str(raw_name).strip()
        key = canonical_entity_key(name)
        if name and key not in seen:
            seen.add(key)
            entities.append(name)
    return {
        "description": str(value.get("description", "")).strip(),
        "ocr_text": str(value.get("ocr_text", "")).strip(),
        "entities": entities,
        "is_diagram": bool(value.get("is_diagram", False)),
        "model_confidence": confidence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
