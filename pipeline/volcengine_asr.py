"""Volcengine ASR V3 adapter — URL-based big-model transcription API.

Follows the official demo pattern:

  - Auth: ``X-Api-App-Key`` + ``X-Api-Access-Key`` + ``X-Api-Resource-Id``
  - Task identity: client-generated UUID in ``X-Api-Request-Id`` header
  - Status: read from response header ``X-Api-Status-Code``
  - Query: POST with empty body, task ID in header
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import httpx

_DEFAULT_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
_DEFAULT_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

# API status codes returned in the ``X-Api-Status-Code`` response header
_STATUS_DONE = "20000000"
_STATUS_PROCESSING = ("20000001", "20000002")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set.")
    return value


def _seconds(value: Any) -> float:
    """Convert V3 API milliseconds to seconds."""
    if value is None:
        return 0.0
    return float(value) / 1000.0


def _as_obj(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _guess_format(url: str) -> str:
    ext = Path(url.rsplit("?", 1)[0]).suffix.lower().lstrip(".")
    return ext if ext in ("mp3", "wav", "ogg", "mp4", "m4a", "aac", "flac") else "mp3"


class _Transcriptions:
    def __init__(self, client: "VolcengineTranscriptionClient") -> None:
        self._client = client

    def create(
        self,
        *,
        file: tuple[str, BinaryIO],
        model: str,
        response_format: str = "verbose_json",
        timestamp_granularities: list[str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        return self._client.transcribe(file, model, timeout)


class _Audio:
    def __init__(self, client: "VolcengineTranscriptionClient") -> None:
        self.transcriptions = _Transcriptions(client)


class VolcengineTranscriptionClient:
    """V3 URL-based ASR client.

    The ``file`` parameter in ``transcribe()`` is ignored — the client
    always submits ``audio_url`` (set by the orchestrator from
    ``DubOptions.audio_url``).
    """

    def __init__(self) -> None:
        self.app_key = _required_env("VOLCENGINE_ASR_APP_KEY")
        self.resource_id = _required_env("VOLCENGINE_ASR_RESOURCE_ID")
        self.submit_url = os.environ.get(
            "VOLCENGINE_ASR_BASE_URL", _DEFAULT_SUBMIT_URL
        )
        self.query_url = os.environ.get(
            "VOLCENGINE_ASR_QUERY_URL", _DEFAULT_QUERY_URL
        )
        self.audio_url = ""  # set by orchestrator from DubOptions.audio_url
        self.audio = _Audio(self)

    @property
    def skip_chunking(self) -> bool:
        return bool(self.audio_url)

    def _base_headers(self) -> dict[str, str]:
        h = {
            "X-Api-Key": self.app_key,
            "X-Api-Resource-Id": self.resource_id,
        }
        return h

    def transcribe(
        self, file: tuple[str, BinaryIO], model: str, timeout: float | None
    ) -> SimpleNamespace:
        if not self.audio_url:
            raise RuntimeError(
                "audio_url is not set on the VolcengineTranscriptionClient. "
                "Pass --audio-url on the CLI or set DubOptions.audio_url."
            )

        task_id = str(uuid.uuid4())
        audio_format = _guess_format(self.audio_url)

        # ── 1. Submit ──────────────────────────────────────────
        submit_headers = {
            **self._base_headers(),
            "Content-Type": "application/json",
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
        }
        submit_payload = {
            "user": {"uid": "violin"},
            "audio": {"format": audio_format, "url": self.audio_url},
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
                "enable_speaker_info": False,
                "enable_channel_split": False,
            },
        }

        import json as _json
        print(f"      [ASR DEBUG] POST {self.submit_url}")
        print(f"      [ASR DEBUG] Headers: {_json.dumps(submit_headers, indent=2)}")
        print(f"      [ASR DEBUG] Body: {_json.dumps(submit_payload, ensure_ascii=False, indent=2)}")

        submit_resp = httpx.post(
            self.submit_url,
            headers=submit_headers,
            json=submit_payload,
            timeout=timeout,
        )
        if submit_resp.status_code != 200:
            print(f"      [ASR DEBUG] Response status: {submit_resp.status_code}")
            print(f"      [ASR DEBUG] Response headers: {dict(submit_resp.headers)}")
            print(f"      [ASR DEBUG] Response body: {submit_resp.text}")
        submit_resp.raise_for_status()

        status = submit_resp.headers.get("X-Api-Status-Code", "")
        if status != _STATUS_DONE:
            raise RuntimeError(
                f"Volcengine ASR V3 submit failed: "
                f"X-Api-Status-Code={status} "
                f"X-Api-Message={submit_resp.headers.get('X-Api-Message', '')}"
            )

        x_tt_logid = submit_resp.headers.get("X-Tt-Logid", "")

        # ── 2. Poll ────────────────────────────────────────────
        result = self._poll_result(task_id, x_tt_logid, timeout)
        return self._normalize(result)

    def _poll_result(
        self, task_id: str, x_tt_logid: str, timeout: float | None
    ) -> dict[str, Any]:
        """Poll query endpoint until the task completes or fails."""
        for _ in range(180):
            headers = {
                **self._base_headers(),
                "Content-Type": "application/json",
                "X-Api-Request-Id": task_id,
            }
            if x_tt_logid:
                headers["X-Tt-Logid"] = x_tt_logid

            response = httpx.post(
                self.query_url,
                headers=headers,
                json={},
                timeout=timeout,
            )
            response.raise_for_status()

            status = response.headers.get("X-Api-Status-Code", "")
            if status == _STATUS_DONE:
                return response.json()
            if status in _STATUS_PROCESSING:
                time.sleep(2)
                continue

            raise RuntimeError(
                f"Volcengine ASR V3 query failed: "
                f"X-Api-Status-Code={status} "
                f"X-Api-Message={response.headers.get('X-Api-Message', '')}"
            )

        raise TimeoutError("Volcengine ASR V3 polling timed out")

    def _normalize(self, data: dict[str, Any]) -> SimpleNamespace:
        """Convert V3 response to ``SimpleNamespace`` (same format as V1)."""
        import json as _json
        print(f"      [ASR DEBUG] Raw result keys: {list(data.keys())}")
        result = data.get("result") or {}
        print(f"      [ASR DEBUG] Result keys: {list(result.keys())}, "
              f"utterances count: {len(result.get('utterances') or [])}")
        # Print first utterance to see field names
        utterances = result.get("utterances") or []
        if utterances:
            print(f"      [ASR DEBUG] First utterance keys: {list(utterances[0].keys())}")
            print(f"      [ASR DEBUG] First utterance sample: {_json.dumps(utterances[0], ensure_ascii=False)}")
        raw_segments = (
            result.get("utterances")
            or result.get("segments")
            or result.get("utterance")
            or []
        )
        if not raw_segments and result.get("text"):
            raw_segments = [{
                "text": result["text"],
                "start_time": 0,
                "end_time": result.get("duration", 0),
            }]

        segments = []
        words = []
        for idx, item in enumerate(raw_segments):
            text = (item.get("text") or item.get("utterance") or "").strip()
            start = _seconds(item.get("start_time", item.get("start", 0)))
            end = _seconds(item.get("end_time", item.get("end", start)))
            segments.append(
                _as_obj(
                    id=idx, text=text, start=start, end=end, no_speech_prob=0.0
                )
            )
            for word in item.get("words") or []:
                w_text = (word.get("word") or word.get("text") or "").strip()
                if not w_text:
                    continue
                words.append(
                    _as_obj(
                        word=w_text,
                        start=_seconds(
                            word.get("start_time", word.get("start", start))
                        ),
                        end=_seconds(
                            word.get("end_time", word.get("end", end))
                        ),
                    )
                )

        return _as_obj(segments=segments, words=words)


def make_volcengine_transcription_client() -> VolcengineTranscriptionClient:
    return VolcengineTranscriptionClient()
