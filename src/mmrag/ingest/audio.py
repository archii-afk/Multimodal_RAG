"""Audio extraction and cached Whisper transcription."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import subprocess
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
    SourceDraft,
    person_key,
)

WHISPER_MODEL = "whisper-1"
PROMPT_VERSION = "v1"
MAX_WHISPER_BYTES = 25 * 1024 * 1024
AUDIO_ROOT = Path("data/processed/audio")
AUDIO_CACHE_ROOT = Path("data/processed/cache/audio")


def ingest_audio(path: str | Path, *, source_ref: str, presenter: str) -> IngestBatch:
    """Extract mono audio, transcribe it, and return timestamped evidence."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not source_ref:
        raise ValueError("source_ref must be non-empty")
    if not presenter.strip():
        raise ValueError("presenter must be non-empty")

    source_sha = _sha256(source_path)
    output_path = AUDIO_ROOT / source_sha / "audio.mp3"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "32k",
                str(output_path),
            ]
        )

    duration = _probe_duration(source_path)
    chunks = [(output_path, 0.0)]
    if output_path.stat().st_size > MAX_WHISPER_BYTES:
        chunks = _split_at_silences(output_path, duration)

    segments: list[dict[str, Any]] = []
    for chunk_path, offset in chunks:
        for segment in _transcribe_cached(chunk_path):
            segments.append(
                {
                    "start": float(segment["start"]) + offset,
                    "end": float(segment["end"]) + offset,
                    "text": str(segment["text"]).strip(),
                }
            )
    segments = _dedupe_segments(segments)

    mime_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    source_modality = Modality.VIDEO if mime_type.startswith("video/") else Modality.AUDIO
    presenter_canonical_key = person_key(presenter)
    person_ref = "person:presenter"
    nodes: list[NodeDraft] = [
        SourceDraft(
            ref=source_ref,
            kind=NodeKind.SOURCE,
            modality=source_modality,
            content=source_path.stem,
            path=str(source_path),
            mime_type=mime_type,
            sha256=source_sha,
            duration=duration,
            presenter=presenter,
            provenance={"extractor": "ffmpeg", "audio_path": str(output_path)},
        ),
        NodeDraft(
            ref=person_ref,
            kind=NodeKind.ENTITY,
            modality=Modality.ENTITY,
            content=presenter.strip(),
            source_ref=source_ref,
            canonical_key=presenter_canonical_key,
            confidence=1.0,
            provenance={"extractor": "presenter_label"},
        ),
    ]
    edges: list[EdgeDraft] = []
    prior_ref: str | None = None
    for index, segment in enumerate(segments):
        segment_ref = f"segment:{index:04d}"
        nodes.append(
            NodeDraft(
                ref=segment_ref,
                kind=NodeKind.TRANSCRIPT_SEGMENT,
                modality=Modality.AUDIO,
                content=segment["text"],
                source_ref=source_ref,
                t_start=segment["start"],
                t_end=segment["end"],
                speaker=presenter.strip(),
                confidence=DEFAULT_CONFIDENCE["asr"],
                provenance={
                    "extractor": "openai_whisper",
                    "model": WHISPER_MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "audio_path": str(output_path),
                },
            )
        )
        edges.append(EdgeDraft(segment_ref, f"key:{presenter_canonical_key}", EdgeKind.SPOKEN_BY))
        if prior_ref is not None:
            edges.append(EdgeDraft(prior_ref, segment_ref, EdgeKind.NEXT))
        prior_ref = segment_ref
    return IngestBatch(tuple(nodes), tuple(edges))


def _transcribe_cached(audio_path: Path) -> list[dict[str, Any]]:
    digest = _sha256(audio_path)
    model_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", WHISPER_MODEL)
    cache_path = AUDIO_CACHE_ROOT / f"{digest}-{model_key}-{PROMPT_VERSION}.json"
    if cache_path.exists():
        return list(json.loads(cache_path.read_text())["segments"])
    payload = {"segments": _whisper_request(audio_path, model=WHISPER_MODEL)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return list(payload["segments"])


def _whisper_request(audio_path: Path, *, model: str) -> list[dict[str, Any]]:
    """Perform one API call; isolated for keyless tests."""
    from openai import OpenAI

    with audio_path.open("rb") as audio:
        response = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).audio.transcriptions.create(
            model=model,
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    raw_segments = response.segments or []
    return [
        {
            "start": float(_value(item, "start")),
            "end": float(_value(item, "end")),
            "text": str(_value(item, "text")),
        }
        for item in raw_segments
    ]


def _value(value: object, name: str) -> object:
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _split_at_silences(audio_path: Path, duration: float) -> list[tuple[Path, float]]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=n=-35dB:d=0.5",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    silence_starts = [float(v) for v in re.findall(r"silence_start: ([0-9.]+)", result.stderr)]
    silence_ends = [float(v) for v in re.findall(r"silence_end: ([0-9.]+)", result.stderr)]
    silences = [(a + b) / 2 for a, b in zip(silence_starts, silence_ends, strict=False)]
    boundaries = [0.0]
    target = 600.0
    while target < duration:
        nearby = [point for point in silences if target - 90 <= point <= target + 90]
        boundary = min(nearby, key=lambda point: abs(point - target)) if nearby else target
        if boundary > boundaries[-1] + 60:
            boundaries.append(boundary)
        target = boundary + 600.0
    boundaries.append(duration)

    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=True)):
        overlap_start = max(0.0, start - (1.0 if index else 0.0))
        chunk_path = chunk_dir / f"chunk-{index:03d}.mp3"
        if not chunk_path.exists():
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(overlap_start),
                    "-i",
                    str(audio_path),
                    "-t",
                    str(end - overlap_start),
                    "-c",
                    "copy",
                    str(chunk_path),
                ]
            )
        chunks.append((chunk_path, overlap_start))
    return chunks


def _dedupe_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        normalized = " ".join(segment["text"].lower().split())
        if not normalized:
            continue
        if any(
            " ".join(previous["text"].lower().split()) == normalized
            and abs(previous["start"] - segment["start"]) <= 3.0
            for previous in result[-3:]
        ):
            continue
        result.append(segment)
    return result


def _probe_duration(path: Path) -> float:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(output)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
