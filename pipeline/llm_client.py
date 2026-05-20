"""Factory for translation + transcription clients."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _parse_translation_config(cfg: dict[str, Any]) -> tuple[str, str]:
    entry = cfg["models"]["translation"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_translation_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_translation_config(cfg)
    return model


def get_translation_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_translation_config(cfg)
    return provider


def make_translation_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    provider, _ = _parse_translation_config(cfg)

    if provider == "openai":
        from openai import OpenAI
        api_key = openai_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)

    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", _DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)

    if provider == "together":
        from together import Together
        api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
        return Together(api_key=api_key)

    raise ValueError(f"Unsupported translation provider: {provider}")


# ── Chat (video Q&A) client ─────────────────────────────────


def _parse_chat_config(cfg: dict[str, Any]) -> tuple[str, str]:
    entry = cfg["models"]["chat"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_chat_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_chat_config(cfg)
    return provider


def get_chat_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_chat_config(cfg)
    return model


def make_chat_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    provider, _ = _parse_chat_config(cfg)

    if provider == "openai":
        from openai import OpenAI
        api_key = openai_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)

    if provider == "together":
        from together import Together
        api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
        return Together(api_key=api_key)

    raise ValueError(f"Unsupported chat provider: {provider}")


# ── Startup validation ──────────────────────────────────────

_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "together": ("TOGETHER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "volcengine": (),
}

_VOLCENGINE_ENV_KEYS = {
    "transcription": (
        "VOLCENGINE_ASR_APP_KEY",
        "VOLCENGINE_ASR_RESOURCE_ID",
    ),
    "tts": (
        "VOLCENGINE_TTS_API_KEY",
        "VOLCENGINE_TTS_RESOURCE_ID",
    ),
}


def _env_keys_for_provider(provider: str, stage: str | None = None) -> tuple[str, ...]:
    if provider == "cartesia":
        provider = "together"
    if provider == "volcengine" and stage in _VOLCENGINE_ENV_KEYS:
        return _VOLCENGINE_ENV_KEYS[stage]
    try:
        return _PROVIDER_ENV_KEYS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def required_env_keys(cfg: dict[str, Any]) -> set[str]:
    keys: set[str] = set()

    keys.update(_env_keys_for_provider(get_transcription_provider(cfg), "transcription"))
    keys.update(_env_keys_for_provider(get_translation_provider(cfg), "translation"))

    tts_entry = cfg["models"].get("tts")
    if isinstance(tts_entry, dict):
        tts_provider = tts_entry.get("provider", "together")
    else:
        tts_provider = "together"
    keys.update(_env_keys_for_provider(tts_provider, "tts"))

    return keys


def validate_env(cfg: dict[str, Any]) -> list[str]:
    return sorted(k for k in required_env_keys(cfg) if not os.environ.get(k))


def _parse_transcription_config(cfg: dict[str, Any]) -> tuple[str, str]:
    entry = cfg["models"]["transcription"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_transcription_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_transcription_config(cfg)
    return model


def get_transcription_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_transcription_config(cfg)
    return provider


def make_transcription_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    provider, _ = _parse_transcription_config(cfg)

    if provider == "openai":
        from openai import OpenAI
        api_key = openai_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)

    if provider == "volcengine":
        from .volcengine_asr import make_volcengine_transcription_client
        return make_volcengine_transcription_client()

    if provider == "together":
        from together import Together
        api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
        return Together(api_key=api_key)

    raise ValueError(f"Unsupported transcription provider: {provider}")