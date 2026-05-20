"""Disk-based stage cache for debugging.

Usage::

    cache = StageCache(hash_key="<sha256-of-input>", cache_dir=".violin_cache")

    segments = cache.get("transcription")
    if segments is None:
        segments = transcribe(...)
        cache.put("transcription", segments)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .transcriber import Segment


def _segments_to_dicts(segs: list[Segment]) -> list[dict[str, Any]]:
    return [
        {"id": s.id, "start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
        for s in segs
    ]


def _dicts_to_segments(data: list[dict[str, Any]]) -> list[Segment]:
    return [
        Segment(id=d["id"], start=d["start"], end=d["end"], text=d["text"], speaker=d.get("speaker", "SPEAKER_00"))
        for d in data
    ]


class StageCache:
    """Cache for pipeline intermediate results.

    Each stage is stored as a JSON file in ``cache_dir / key / stage.json``.
    """

    def __init__(self, key: str, cache_dir: str = ".violin_cache") -> None:
        self._root = Path(cache_dir) / key
        self._root.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        """Delete all cached data for this key."""
        shutil.rmtree(self._root, ignore_errors=True)

    # ── Transcription ────────────────────────────────────────

    def get_transcription(self) -> list[Segment] | None:
        return self._load_segments("transcription")

    def put_transcription(self, segments: list[Segment]) -> None:
        self._dump_segments("transcription", segments)

    # ── Translation ──────────────────────────────────────────

    def get_translation(self) -> list[Segment] | None:
        return self._load_segments("translation")

    def put_translation(self, segments: list[Segment]) -> None:
        self._dump_segments("translation", segments)

    # ── TTS paths ────────────────────────────────────────────

    def get_tts_paths(self) -> list[str] | None:
        path = self._root / "tts_paths.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def put_tts_paths(self, paths: list[str]) -> None:
        path = self._root / "tts_paths.json"
        with open(path, "w") as f:
            json.dump(paths, f, indent=2)

    # ── Internal ─────────────────────────────────────────────

    def _load_segments(self, name: str) -> list[Segment] | None:
        path = self._root / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return _dicts_to_segments(data)
        except Exception:
            return None

    def _dump_segments(self, name: str, segments: list[Segment]) -> None:
        path = self._root / f"{name}.json"
        data = _segments_to_dicts(segments)
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
