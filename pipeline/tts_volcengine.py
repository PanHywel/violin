"""Volcengine TTS backend — V3 HTTP Chunked API."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment

_DEFAULT_BASE_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

_NATIVE_VOICES: dict[str, dict[str, dict[str, str]]] = {
    "zh": {
        "zh_male": {"voice_type": "zh_male_m191_uranus_bigtts", "gender": "male", "description": "云舟 2.0 — 清爽沉稳男声"},
        "zh_female": {"voice_type": "zh_female_vv_uranus_bigtts", "gender": "female", "description": "Vivi 2.0 — 活泼灵动女声"},
    },
    "en": {
        "en_male": {"voice_type": "en_male_tim_uranus_bigtts", "gender": "male", "description": "Tim — 美式英语男声"},
        "en_female": {"voice_type": "en_female_dacey_uranus_bigtts", "gender": "female", "description": "Dacey — 美式英语女声"},
    },
}


def native_voices_for(language_code: str) -> list[str]:
    voices = _NATIVE_VOICES.get(language_code) or _NATIVE_VOICES["en"]
    names = list(voices.keys())
    male = next((n for n in names if voices[n]["gender"] == "male"), names[0])
    female = next((n for n in names if voices[n]["gender"] == "female"), names[-1])
    return [male, female]


def all_voices() -> dict[str, list[str]]:
    return {lang: list(voices.keys()) for lang, voices in _NATIVE_VOICES.items()}


def voice_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for voices in _NATIVE_VOICES.values():
        for name, meta in voices.items():
            out[name] = f"{meta['gender']} — {meta['description']} ({meta['voice_type']})"
    return out


def _required(config: dict[str, Any], key: str, env: str) -> str:
    value = config.get(key) or os.environ.get(env)
    if not value:
        raise RuntimeError(f"{env} environment variable is not set.")
    return value


def _resolve_voice_type(voice: str) -> str:
    for voices in _NATIVE_VOICES.values():
        if voice in voices:
            return voices[voice]["voice_type"]
    return voice or os.environ.get("VOLCENGINE_TTS_VOICE_TYPE", "")


def _to_wav(src_path: str, wav_path: str, tail_ms: int) -> None:
    af = []
    if tail_ms > 0:
        af = ["-af", f"apad=pad_dur={tail_ms / 1000:.3f}"]
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", src_path, *af, "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: dict[str, Any],
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    api_key = _required(client, "api_key", "VOLCENGINE_TTS_API_KEY")
    resource_id = _required(client, "resource_id", "VOLCENGINE_TTS_RESOURCE_ID")
    base_url = client.get("base_url") or os.environ.get("VOLCENGINE_TTS_BASE_URL", _DEFAULT_BASE_URL)
    cfg = _conf.get()
    voice_type = _resolve_voice_type(voice)
    if not voice_type:
        raise RuntimeError("VOLCENGINE_TTS_VOICE_TYPE is not set and no voice was selected.")

    req_headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
    }

    # ── V3 HTTP Chunked request ──────────────────────────────
    payload: dict[str, Any] = {
        "user": {"uid": os.environ.get("VOLCENGINE_TTS_UID", "violin")},
        "req_params": {
            "text": text,
            "speaker": voice_type,
            "audio_params": {
                "format": "mp3",
                "sample_rate": 24000,
            },
        },
    }
    # Optional model override (e.g. seed-tts-2.0-expressive)
    model_id = cfg["models"]["tts"].get("model", "volcano_tts")
    if model_id and model_id not in ("volcano_tts",):
        payload["req_params"]["model"] = model_id
    # Map V1 speed_ratio (0.1–2.0) to V3 speech_rate (-50–+100).
    # TTS 2.0 voices tend to speak slower than 1.0 voices, offset calculated:
    #   2.0x (rate=100) → 350s,  need 630s → ~1.1x (rate=10).
    base_rate = 10
    if speed is not None:
        base_rate += int((speed - 1.0) * 100)
    payload["req_params"]["audio_params"]["speech_rate"] = max(-50, min(100, base_rate))

    import json as _json
    print(f"      [TTS DEBUG] POST {base_url}")
    print(f"      [TTS DEBUG] Headers: {_json.dumps(req_headers, indent=2)}")
    print(f"      [TTS DEBUG] Body: {_json.dumps(payload, ensure_ascii=False, indent=2)}")

    response = httpx.post(
        base_url,
        headers=req_headers,
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        print(f"      [TTS DEBUG] Response status: {response.status_code}")
        print(f"      [TTS DEBUG] Response headers: {dict(response.headers)}")
        print(f"      [TTS DEBUG] Response body: {response.text}")
    response.raise_for_status()

    # ── Read chunked response ──────────────────────────────────
    audio_chunks: list[bytes] = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = chunk.get("code", 0)
        if code == 20000000:
            break  # normal end-of-stream
        if code > 0:
            raise RuntimeError(
                f"Volcengine TTS V3 error: {chunk.get('message', '')} (code {code})"
            )
        # code == 0 — audio data or sentence metadata
        data = chunk.get("data")
        if data:
            audio_chunks.append(base64.b64decode(data))

    if not audio_chunks:
        raise RuntimeError("Volcengine TTS V3 returned no audio data")
    audio_bytes = b"".join(audio_chunks)

    # ── Save & convert (same as V1) ──────────────────────────
    mp3_path = output_path + ".tmp.mp3"
    Path(mp3_path).write_bytes(audio_bytes)
    tcfg = cfg.get("tts", {})
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _to_wav(mp3_path, output_path, tail_ms)
    Path(mp3_path).unlink(missing_ok=True)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: dict[str, Any],
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    total = len(segments)
    paths = [""] * total
    vm = voice_map or {}

    def _do(idx: int, seg: Segment) -> tuple[int, str]:
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        seg_voice = vm.get(seg.speaker, voice)
        synthesize_segment(seg.text, seg_voice, path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
        return idx, path

    done_count = 0
    with ThreadPoolExecutor(max_workers=_conf.get()["tts"]["workers"]) as pool:
        futures = {pool.submit(_do, i, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx, path = future.result()
            paths[idx] = path
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"      TTS progress: {done_count}/{total} segments done")

    # Print total audio duration for debugging
    import wave as _wave
    total_secs = 0.0
    for p in paths:
        try:
            with _wave.open(p, "r") as wf:
                total_secs += wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass
    print(f"      [TTS] Total audio duration: {total_secs:.1f}s ({total_secs/60:.1f} min) "
          f"across {len(paths)} segments")

    return paths
