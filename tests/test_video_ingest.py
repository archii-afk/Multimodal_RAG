from __future__ import annotations

import subprocess

import pytest

from mmrag.ingest import video
from mmrag.model import NodeKind


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
            "color=c=black:s=160x90:r=2:d=5",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=160x90:r=2:d=5",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
    )


def test_sample_video_frames_emits_windows_and_frame_provenance(tmp_path, monkeypatch):
    clip = tmp_path / "diagram.mp4"
    _make_clip(clip)
    monkeypatch.setattr(video, "FRAME_ROOT", tmp_path / "processed/frames")

    batch = video.sample_video_frames(
        clip,
        source_ref="source:video",
        interval_s=2,
        min_gap_s=1,
        max_frames=8,
    )

    assert batch.nodes
    assert all(node.kind is NodeKind.FRAME for node in batch.nodes)
    assert batch.nodes[0].t_start == 0.0
    assert batch.nodes[-1].t_end == pytest.approx(10.0)
    for left, right in zip(batch.nodes, batch.nodes[1:], strict=False):
        assert left.t_end == pytest.approx(right.t_start)
    for node in batch.nodes:
        assert node.provenance["sampled_at"] >= 0
        assert node.provenance["sha256"]
        assert video.Path(node.provenance["frame_path"]).is_file()


def test_sample_video_frames_rejects_impossible_coverage(tmp_path, monkeypatch):
    clip = tmp_path / "diagram.mp4"
    _make_clip(clip)
    monkeypatch.setattr(video, "FRAME_ROOT", tmp_path / "frames")
    monkeypatch.setattr(video, "_probe_duration", lambda _: 301.0)

    with pytest.raises(ValueError, match="cannot cover"):
        video.sample_video_frames(clip, source_ref="source:video", max_frames=10)
