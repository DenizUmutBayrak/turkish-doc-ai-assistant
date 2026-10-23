"""Small standard-library client for Ollama's local chat endpoint."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.core.config import Settings, get_settings
from backend.services.answer_generator import AnswerGenerationError

MAX_REQUEST_BYTES = 128_000
MAX_RESPONSE_BYTES = 1_000_000


class OllamaClientError(AnswerGenerationError):
    """Safe, user-facing Ollama request failure."""


class OllamaConnectionError(OllamaClientError):
    """Ollama could not be reached."""


class OllamaTimeoutError(OllamaClientError):
    """Ollama did not respond before the configured timeout."""


class OllamaHTTPError(OllamaClientError):
    """Ollama returned a non-success HTTP response."""


class OllamaResponseError(OllamaClientError):
    """Ollama returned an invalid or empty response."""


class OllamaChatClient:
    """HTTP adapter for POST /api/chat; no Ollama SDK is required."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "think": self.settings.ollama_think,
            "options": {"num_predict": self.settings.ollama_num_predict},
        }
        try:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise OllamaResponseError("Ollama isteği hazırlanamadı.") from exc
        if len(payload) > MAX_REQUEST_BYTES:
            raise OllamaResponseError("Ollama isteği izin verilen boyutu aşıyor.")

        request = Request(
            f"{self.settings.ollama_base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.ollama_timeout_seconds) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise OllamaHTTPError(
                f"Ollama HTTP {exc.code} yanıtı verdi. Ollama'nın çalıştığını "
                "ve seçilen modelin yüklü olduğunu kontrol edin."
            ) from exc
        except (TimeoutError, ConnectionError) as exc:
            if isinstance(exc, TimeoutError):
                raise OllamaTimeoutError(
                    "Ollama yanıtı zaman aşımına uğradı. Timeout ayarını veya "
                    "yerel modelin çalışma süresini kontrol edin."
                ) from exc
            raise OllamaConnectionError(
                "Ollama'ya bağlanılamadı. Ollama'nın kurulu ve çalışıyor olduğunu kontrol edin."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError,)):
                raise OllamaTimeoutError(
                    "Ollama yanıtı zaman aşımına uğradı. Timeout ayarını veya "
                    "yerel modelin çalışma süresini kontrol edin."
                ) from exc
            raise OllamaConnectionError(
                "Ollama'ya bağlanılamadı. Ollama'nın kurulu ve çalışıyor olduğunu kontrol edin."
            ) from exc
        except OSError as exc:
            raise OllamaConnectionError(
                "Ollama'ya bağlanılamadı. Ollama'nın kurulu ve çalışıyor olduğunu kontrol edin."
            ) from exc

        if len(response_body) > MAX_RESPONSE_BYTES:
            raise OllamaResponseError("Ollama yanıtı izin verilen boyutu aşıyor.")
        try:
            data = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaResponseError("Ollama geçersiz JSON yanıtı verdi.") from exc
        if not isinstance(data, Mapping):
            raise OllamaResponseError("Ollama yanıtının biçimi beklenenden farklı.")
        message = data.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise OllamaResponseError("Ollama yanıtında asistan mesajı bulunamadı.")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama yanıt metninin biçimi beklenenden farklı.")
        if not content.strip():
            raise OllamaResponseError("Ollama boş yanıt döndürdü.")
        return content.strip()
