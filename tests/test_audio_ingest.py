from __future__ import annotations

import subprocess

from mmrag.ingest import audio
from mmrag.model import EdgeKind, NodeKind


def _make_clip(path):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=1:d=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=10",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def test_ingest_audio_emits_source_segments_and_cached_api_result(tmp_path, monkeypatch):
    clip = tmp_path / "meeting.mp4"
    _make_clip(clip)
    monkeypatch.setattr(audio, "AUDIO_ROOT", tmp_path / "processed/audio")
    monkeypatch.setattr(audio, "AUDIO_CACHE_ROOT", tmp_path / "processed/cache/audio")
    calls = []

    def fake_request(path, *, model):
        calls.append((path, model))
        return [
            {"start": 0.2, "end": 2.0, "text": "We use read replicas."},
            {"start": 2.1, "end": 4.0, "text": "They reduce database load."},
        ]

    monkeypatch.setattr(audio, "_whisper_request", fake_request)
    first = audio.ingest_audio(clip, source_ref="source:video", presenter="Alex Smith")
    second = audio.ingest_audio(clip, source_ref="source:video", presenter="Alex Smith")

    assert first == second
    assert len(calls) == 1
    assert [node.kind for node in first.nodes].count(NodeKind.TRANSCRIPT_SEGMENT) == 2
    assert first.nodes[0].kind is NodeKind.SOURCE
    assert first.nodes[0].duration == 10.0
    assert any(node.canonical_key == "person:alex smith" for node in first.nodes)
    assert [edge.kind for edge in first.edges].count(EdgeKind.SPOKEN_BY) == 2
    assert [edge.kind for edge in first.edges].count(EdgeKind.NEXT) == 1
    assert all(
        edge.dst_ref == "key:person:alex smith"
        for edge in first.edges
        if edge.kind is EdgeKind.SPOKEN_BY
    )
