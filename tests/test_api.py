"""Offline API tests use fake embeddings and fake answer generation."""

from collections.abc import Sequence
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend.api.main import create_app
from backend.core.config import Settings
from backend.services.embedding_service import EmbeddingService
from backend.services.rag_service import RagService, RagServiceError
from backend.services.retrieval import InMemoryVectorIndex
from frontend.app import ApiClient, ApiRequestError, _fingerprint, _new_uploads, _status_message


class FixedEmbedder:
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeGenerator:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            '{"answer":"Sentetik test yanıtı.","source_ids":["S1"],"insufficient_evidence":false}'
        )


def make_pdf(text: str = "Warranty lasts 18 months.") -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    page = writer.pages[0]
    content = BytesIO()
    # Keep the test fixture self-contained and use pypdf's text content helper.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(content)
    return content.getvalue()


def build_client(
    *,
    settings: Settings | None = None,
    rag_factory=None,
    index_factory=None,
) -> TestClient:
    active = settings or Settings(_env_file=None)

    def make_index(config: Settings):
        return InMemoryVectorIndex(
            embedding_service=EmbeddingService(backend=FixedEmbedder(), settings=config),
            settings=config,
        )

    def make_rag(index, config):
        return RagService(index, FakeGenerator(), config)

    return TestClient(
        create_app(
            active,
            index_factory=index_factory or make_index,
            rag_factory=rag_factory or make_rag,
        )
    )


def create_session(client: TestClient) -> str:
    response = client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def upload(client: TestClient, session_id: str, name: str, content: bytes | None = None):
    return client.post(
        f"/api/v1/sessions/{session_id}/documents",
        files={
            "file": (
                name,
                make_pdf() if content is None else content,
                "application/pdf",
            )
        },
    )


def test_health_does_not_create_session_or_load_model() -> None:
    def forbidden_factory(_settings):
        raise AssertionError("Health endpoint should not create an index.")

    client = build_client(index_factory=forbidden_factory)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "hazır olduğunu göstermez" in response.json()["message"]


def test_local_api_docs_and_openapi_are_enabled() -> None:
    client = build_client()

    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/health" in openapi.json()["paths"]
    assert "/api/v1/sessions/{session_id}/documents" in openapi.json()["paths"]


def test_upload_indexing_failure_logs_types_without_sensitive_error(caplog) -> None:
    class FailingIndex:
        def add_chunks(self, _chunks):
            raise RuntimeError("private document contents and secret config")

    client = build_client(index_factory=lambda _settings: FailingIndex())
    session_id = create_session(client)

    response = upload(client, session_id, "sentetik.pdf")

    assert response.status_code == 503
    assert response.json()["detail"] == "PDF yerel olarak indekslenemedi."
    assert "RuntimeError" in caplog.text
    assert "private document contents" not in caplog.text
    assert "secret config" not in caplog.text


def test_session_ids_are_unpredictable_and_independent() -> None:
    client = build_client()

    first = create_session(client)
    second = create_session(client)

    assert first != second
    assert len(first) >= 40


def test_missing_session_rejected_for_upload_and_question() -> None:
    client = build_client()

    upload_response = upload(client, "unknown-session", "ornek.pdf")
    question_response = client.post(
        "/api/v1/sessions/unknown-session/questions", json={"question": "Garanti kaç ay?"}
    )

    assert upload_response.status_code == question_response.status_code == 404
    assert "Oturum bulunamadı" in upload_response.json()["detail"]


def test_valid_pdf_is_parsed_indexed_and_summarized() -> None:
    client = build_client()
    session_id = create_session(client)

    response = upload(client, session_id, r"..private\garanti.pdf")

    assert response.status_code == 201
    assert response.json()["document"] == {
        "filename": "garanti.pdf",
        "page_count": 1,
        "chunk_count": 1,
    }
    assert response.json()["documents"] == [response.json()["document"]]


def test_upload_rejects_oversized_non_pdf_and_corrupt_files() -> None:
    client = build_client(settings=Settings(_env_file=None, max_upload_size_mb=1))
    session_id = create_session(client)

    oversized = upload(client, session_id, "buyuk.pdf", b"x" * (1024 * 1024 + 1))
    wrong_extension = upload(client, session_id, "notlar.txt", b"text")
    corrupt_pdf = upload(client, session_id, "bozuk.pdf", b"not a pdf")

    assert oversized.status_code == 413
    assert wrong_extension.status_code == 400
    assert corrupt_pdf.status_code == 400
    assert "PDF" in corrupt_pdf.json()["detail"]


def test_questions_use_only_the_current_session_index_and_return_sources() -> None:
    client = build_client()
    first = create_session(client)
    second = create_session(client)
    assert upload(client, first, "garanti.pdf").status_code == 201

    empty_session_question = client.post(
        f"/api/v1/sessions/{second}/questions", json={"question": "Garanti ne kadar?"}
    )
    assert empty_session_question.status_code == 409

    second_upload = upload(client, second, "iade.pdf", make_pdf("Returns allowed in 30 days."))
    assert second_upload.status_code == 201
    first_answer = client.post(
        f"/api/v1/sessions/{first}/questions", json={"question": "Garanti ne kadar?"}
    )
    second_answer = client.post(
        f"/api/v1/sessions/{second}/questions", json={"question": "İade süresi nedir?"}
    )

    assert first_answer.status_code == second_answer.status_code == 200
    assert first_answer.json()["sources"][0]["filename"] == "garanti.pdf"
    assert second_answer.json()["sources"][0]["filename"] == "iade.pdf"
    assert first_answer.json()["sources"][0]["page_number"] == 1
    assert first_answer.json()["answer"].startswith("Sentetik test yanıtı.")


def test_question_input_errors_are_safe_and_do_not_echo_user_text() -> None:
    client = build_client()
    session_id = create_session(client)
    upload(client, session_id, "ornek.pdf")

    response = client.post(f"/api/v1/sessions/{session_id}/questions", json={"question": "   "})

    assert response.status_code == 422
    assert "input" not in response.text
    assert "question" not in response.text


def test_ollama_failure_maps_to_controlled_service_unavailable() -> None:
    class FailingRag:
        def answer(self, _question: str):
            raise RagServiceError("C:\\private\\secret.pdf model internals")

    def failing_rag(_index, _settings):
        return FailingRag()

    client = build_client(rag_factory=failing_rag)
    session_id = create_session(client)
    upload(client, session_id, "ornek.pdf")

    response = client.post(
        f"/api/v1/sessions/{session_id}/questions", json={"question": "Belgede ne var?"}
    )

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_delete_clears_session_and_is_idempotent() -> None:
    client = build_client()
    session_id = create_session(client)
    upload(client, session_id, "ornek.pdf")

    first = client.delete(f"/api/v1/sessions/{session_id}")
    second = client.delete(f"/api/v1/sessions/{session_id}")

    assert first.status_code == second.status_code == 200
    assert first.json()["deleted"] is True
    assert second.json()["deleted"] is False
    assert (
        client.post(
            f"/api/v1/sessions/{session_id}/questions", json={"question": "Soru"}
        ).status_code
        == 404
    )


def test_session_store_expires_idle_sessions_and_clears_index() -> None:
    from backend.api.session_store import SessionStore

    now = [10.0]
    settings = Settings(_env_file=None, session_ttl_minutes=1)
    store = SessionStore(settings, clock=lambda: now[0])
    session = store.create()
    now[0] += 61

    assert store.get(session.session_id) is None


def test_default_session_index_receives_settings_as_configuration() -> None:
    from backend.api.session_store import SessionStore

    settings = Settings(_env_file=None, embedding_model="local-test-model")
    session = SessionStore(settings).create()

    assert session.index.settings is settings
    assert session.index.embedding_service.settings is settings


def test_frontend_api_errors_have_local_user_friendly_messages() -> None:
    assert "Backend" in _status_message(503)
    assert "Oturum" in _status_message(404)
    assert "sınırı" in _status_message(413)
    assert "OCR" in _status_message(422)


def test_frontend_deduplicates_selected_uploads_and_can_explicitly_repeat() -> None:
    from types import SimpleNamespace

    content = b"same synthetic PDF bytes"
    files = [SimpleNamespace(name="one.pdf", getvalue=lambda: content)] * 2

    fingerprint = _fingerprint(content)
    skipped = _new_uploads(files, {fingerprint}, allow_duplicates=False)
    repeated = _new_uploads(files[:1], {fingerprint}, allow_duplicates=True)

    assert skipped == []
    assert len(repeated) == 1
    assert repeated[0][1] == content


def test_api_client_connection_errors_are_actionable(monkeypatch) -> None:
    import httpx

    def fail_request(*_args, **_kwargs):
        raise httpx.ConnectError("private transport details")

    monkeypatch.setattr(httpx, "request", fail_request)

    try:
        ApiClient("http://127.0.0.1:8000").request("GET", "/health")
    except ApiRequestError as exc:
        assert exc.status_code is None
        assert "API'yi başlatıp tekrar deneyin" in str(exc)
        assert "private transport details" not in str(exc)
    else:
        raise AssertionError("Connection error should be mapped to a user-safe message.")
