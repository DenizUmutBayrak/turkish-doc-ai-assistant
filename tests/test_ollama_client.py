"""HTTP adapter tests without a running Ollama server."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from backend.core.config import Settings
from backend.services import ollama_client
from backend.services.ollama_client import (
    MAX_REQUEST_BYTES,
    OllamaChatClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
)


def test_client_sends_non_streaming_chat_with_local_model_settings(monkeypatch) -> None:
    response = io.BytesIO(
        json.dumps(
            {"message": {"role": "assistant", "content": "Merhaba."}},
            ensure_ascii=False,
        ).encode("utf-8")
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(ollama_client, "urlopen", fake_urlopen)
    settings = Settings(
        _env_file=None,
        ollama_model="qwen3:1.7b",
        ollama_timeout_seconds=37,
    )

    answer = OllamaChatClient(settings).generate("Sistem.", "Soru.")

    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert answer == "Merhaba."
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 37
    assert payload["model"] == "qwen3:1.7b"
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 512
    assert payload["messages"][0] == {"role": "system", "content": "Sistem."}
    assert payload["messages"][1] == {"role": "user", "content": "Soru."}


def test_connection_refusal_has_actionable_error(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise URLError(ConnectionRefusedError("secret connection details"))

    monkeypatch.setattr(ollama_client, "urlopen", fail_urlopen)
    with pytest.raises(OllamaConnectionError, match="kurulu ve çalışıyor") as error:
        OllamaChatClient(Settings(_env_file=None)).generate("system", "user")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timeout"),
        URLError(TimeoutError("timeout")),
    ],
)
def test_timeout_has_actionable_error(monkeypatch, failure) -> None:
    def fail_urlopen(request, timeout):
        raise failure

    monkeypatch.setattr(ollama_client, "urlopen", fail_urlopen)
    with pytest.raises(OllamaTimeoutError, match="zaman aşımına"):
        OllamaChatClient(Settings(_env_file=None)).generate("system", "user")


def test_http_error_does_not_expose_response_body(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            500,
            "server error",
            hdrs=None,
            fp=io.BytesIO(b"private server response"),
        )

    monkeypatch.setattr(ollama_client, "urlopen", fail_urlopen)
    with pytest.raises(OllamaHTTPError, match="HTTP 500") as error:
        OllamaChatClient(Settings(_env_file=None)).generate("system", "user")
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "response_body",
    [
        "not json",
        "[]",
        '{"message": []}',
        '{"message": {"role": "user", "content": "hatalı rol"}}',
        '{"message": {"role": "assistant", "content": 23}}',
        '{"message": {"role": "assistant", "content": "   "}}',
    ],
)
def test_unexpected_or_empty_response_is_rejected(monkeypatch, response_body) -> None:
    monkeypatch.setattr(
        ollama_client,
        "urlopen",
        lambda request, timeout: io.BytesIO(response_body.encode("utf-8")),
    )
    with pytest.raises(OllamaResponseError):
        OllamaChatClient(Settings(_env_file=None)).generate("system", "user")


def test_request_body_limit_is_enforced_before_network(monkeypatch) -> None:
    def forbidden_urlopen(request, timeout):
        pytest.fail("boyut sınırı aşıldığında ağa istek gönderilmemeliydi")

    monkeypatch.setattr(ollama_client, "urlopen", forbidden_urlopen)
    with pytest.raises(OllamaResponseError, match="boyutu"):
        OllamaChatClient(Settings(_env_file=None)).generate("system", "x" * MAX_REQUEST_BYTES)
