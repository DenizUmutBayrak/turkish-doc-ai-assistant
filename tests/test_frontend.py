"""Streamlit script tests for local UI state and API failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).parents[1] / "frontend" / "app.py"


def _response(method: str, url: str, *, status_code: int, payload: dict[str, Any]):
    response = httpx.Response(status_code, json=payload)
    response.request = httpx.Request(method, url)
    return response


def test_ui_shows_actionable_message_when_backend_is_offline(monkeypatch) -> None:
    def offline_request(*_args, **_kwargs):
        raise httpx.ConnectError("private connection detail")

    monkeypatch.setattr(httpx, "request", offline_request)

    app = AppTest.from_file(str(APP_FILE)).run()

    assert not app.exception
    assert any("Backend API'ye bağlanılamadı" in error.value for error in app.error)
    assert any("Tekrar dene" in button.label for button in app.button)


def test_session_reset_deletes_backend_data_and_creates_fresh_session(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def mock_request(method: str, url: str, **_kwargs):
        calls.append((method, url))
        if method == "POST" and url.endswith("/api/v1/sessions"):
            count = sum(
                call_method == "POST" and call_url.endswith("/api/v1/sessions")
                for call_method, call_url in calls
            )
            return _response(
                method,
                url,
                status_code=201,
                payload={"session_id": f"session-{count}", "expires_in_seconds": 3600},
            )
        if method == "GET":
            return _response(method, url, status_code=200, payload={"status": "ok"})
        return _response(
            method,
            url,
            status_code=200,
            payload={"deleted": True, "message": "deleted"},
        )

    monkeypatch.setattr(httpx, "request", mock_request)
    app = AppTest.from_file(str(APP_FILE)).run()
    app.button[1].click().run()

    assert not app.exception
    assert ("DELETE", "http://localhost:8000/api/v1/sessions/session-1") in calls
    assert app.session_state["session_id"] == "session-2"
    assert app.session_state["documents"] == []
    assert app.session_state["messages"] == []


def test_failed_backend_delete_keeps_current_ui_session(monkeypatch) -> None:
    def mock_request(method: str, url: str, **_kwargs):
        if method == "POST" and url.endswith("/api/v1/sessions"):
            return _response(
                method,
                url,
                status_code=201,
                payload={"session_id": "session-stays", "expires_in_seconds": 3600},
            )
        if method == "GET":
            return _response(method, url, status_code=200, payload={"status": "ok"})
        return _response(
            method,
            url,
            status_code=503,
            payload={"detail": "Backend oturumu temizleyemedi."},
        )

    monkeypatch.setattr(httpx, "request", mock_request)
    app = AppTest.from_file(str(APP_FILE)).run()
    app.button[1].click().run()

    assert not app.exception
    assert app.session_state["session_id"] == "session-stays"
    assert any("Oturum temizlenemedi" in error.value for error in app.error)


def test_failed_pdf_indexing_does_not_add_a_document_or_fingerprint(monkeypatch) -> None:
    def mock_request(method: str, url: str, **_kwargs):
        if method == "POST" and url.endswith("/api/v1/sessions"):
            return _response(
                method,
                url,
                status_code=201,
                payload={"session_id": "session-upload", "expires_in_seconds": 3600},
            )
        if method == "GET":
            return _response(method, url, status_code=200, payload={"status": "ok"})
        return _response(
            method,
            url,
            status_code=503,
            payload={"detail": "Yerel embedding modeli kullanılamıyor."},
        )

    monkeypatch.setattr(httpx, "request", mock_request)
    app = AppTest.from_file(str(APP_FILE)).run()
    app.file_uploader[0].upload(
        "synthetic.pdf", b"synthetic document bytes", "application/pdf"
    ).run()
    upload_button = next(
        button for button in app.button if "PDF" in button.label and "yükle" in button.label
    )
    upload_button.click().run()

    assert not app.exception
    assert app.session_state["documents"] == []
    assert app.session_state["uploaded_fingerprints"] == set()
    assert any("Yerel embedding modeli kullanılamıyor" in error.value for error in app.error)
