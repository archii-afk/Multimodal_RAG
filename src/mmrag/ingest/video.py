"""Hybrid fixed-interval and scene-change video frame sampling."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

from PIL import Image

from mmrag.model import DEFAULT_CONFIDENCE, IngestBatch, Modality, NodeDraft, NodeKind

FRAME_ROOT = Path("data/processed/frames")
COVERAGE_WINDOW_S = 30.0
PERCEPTUAL_HASH_DISTANCE = 6


def sample_video_frames(
    path: str | Path,
    *,
    source_ref: str,
    interval_s: float = 5.0,
    scene_threshold: float = 0.20,
    min_gap_s: float = 2.0,
    max_frames: int = 80,
) -> IngestBatch:
    """Sample frames while preserving coverage of every 30-second bucket."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if interval_s <= 0 or min_gap_s < 0 or max_frames <= 0:
        raise ValueError("interval_s and max_frames must be positive; min_gap_s cannot be negative")
    if not 0 <= scene_threshold <= 1:
        raise ValueError("scene_threshold must be between 0 and 1")

    duration = _probe_duration(video_path)
    required_count = max(1, math.ceil(duration / COVERAGE_WINDOW_S))
    if required_count > max_frames:
        raise ValueError(
            f"max_frames={max_frames} cannot cover {duration:.1f}s at one frame per 30s "
            f"(requires {required_count})"
        )

    source_sha = _sha256(video_path)
    output_dir = FRAME_ROOT / source_sha
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors = [
        (bucket * COVERAGE_WINDOW_S + min((bucket + 1) * COVERAGE_WINDOW_S, duration)) / 2
        for bucket in range(required_count)
    ]
    fixed = _frange(0.0, duration, interval_s)
    scenes = _scene_times(video_path, scene_threshold)
    extras = _evenly_spaced(
        [
            candidate
            for candidate in sorted(set(fixed + scenes))
            if all(abs(candidate - anchor) >= min_gap_s for anchor in anchors)
        ],
        max_frames - len(anchors),
    )

    retained: list[tuple[float, Path, int]] = []
    # Coverage anchors are mandatory; extras are rejected when visually redundant.
    for sampled_at in anchors + extras:
        frame_path = output_dir / f"frame-{round(sampled_at * 1000):010d}.jpg"
        if not frame_path.exists():
            _extract_frame(video_path, sampled_at, frame_path)
        frame_hash = _difference_hash(frame_path)
        mandatory = sampled_at in anchors
        if not mandatory and any(
            abs(sampled_at - prior_time) < min_gap_s
            or _hamming_distance(frame_hash, prior_hash) <= PERCEPTUAL_HASH_DISTANCE
            for prior_time, _, prior_hash in retained
        ):
            continue
        retained.append((sampled_at, frame_path, frame_hash))

    retained.sort(key=lambda item: item[0])
    times = [item[0] for item in retained]
    nodes: list[NodeDraft] = []
    for index, (sampled_at, frame_path, frame_hash) in enumerate(retained):
        t_start = 0.0 if index == 0 else (times[index - 1] + sampled_at) / 2
        t_end = duration if index == len(times) - 1 else (sampled_at + times[index + 1]) / 2
        nodes.append(
            NodeDraft(
                ref=f"frame:{index:04d}",
                kind=NodeKind.FRAME,
                modality=Modality.VIDEO,
                content="",
                source_ref=source_ref,
                t_start=t_start,
                t_end=t_end,
                confidence=DEFAULT_CONFIDENCE["vision"],
                provenance={
                    "extractor": "ffmpeg_hybrid_sampler",
                    "frame_path": str(frame_path),
                    "sampled_at": sampled_at,
                    "sha256": _sha256(frame_path),
                    "perceptual_hash": f"{frame_hash:016x}",
                    "interval_s": interval_s,
                    "scene_threshold": scene_threshold,
                    "min_gap_s": min_gap_s,
                    "source_sha256": source_sha,
                },
            )
        )
    return IngestBatch(tuple(nodes))


def _scene_times(video_path: Path, threshold: float) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video_path),
            "-vf",
            f"select=gt(scene\\,{threshold}),showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]


def _extract_frame(video_path: Path, sampled_at: float, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{sampled_at:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "format=yuvj420p",
            "-q:v",
            "2",
            str(destination),
        ],
        check=True,
    )


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _evenly_spaced(values: list[float], limit: int) -> list[float]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    return [values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)] if limit > 1 else [values[len(values) // 2]]


def _frange(start: float, stop: float, step: float) -> list[float]:
    return [start + index * step for index in range(math.ceil(max(stop - start, 0.0) / step))]


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
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
    )
    return float(result.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
