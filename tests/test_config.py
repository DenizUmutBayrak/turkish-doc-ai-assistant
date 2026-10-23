"""Temel yapılandırma davranışının ağdan bağımsız kontrolleri."""

from backend.core.config import Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.api_base_url == "http://localhost:8000"
    assert settings.max_upload_size_mb == 20
    assert settings.max_pdf_pages == 300
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.embedding_device == "cpu"
    assert settings.embedding_query_prefix == "query:"
    assert settings.embedding_document_prefix == "passage:"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "qwen3:1.7b"
    assert settings.ollama_timeout_seconds == 120
    assert settings.ollama_think is False
    assert settings.ollama_allow_remote is False
    assert settings.rag_max_context_chars == 16000


def test_ollama_remote_address_requires_explicit_opt_in() -> None:
    try:
        Settings(_env_file=None, ollama_base_url="http://192.0.2.10:11434")
    except ValueError as exc:
        assert "OLLAMA_ALLOW_REMOTE=true" in str(exc)
    else:
        raise AssertionError("Uzak Ollama adresi varsayılan olarak reddedilmeliydi.")


def test_ollama_remote_address_can_be_explicitly_enabled() -> None:
    settings = Settings(
        _env_file=None,
        ollama_allow_remote=True,
        ollama_base_url="http://192.0.2.10:11434/",
    )
    assert settings.ollama_base_url == "http://192.0.2.10:11434"


def test_settings_read_environment_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("EMBEDDING_MODEL", "test/embedding")
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query-task:")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-only-secret"
    assert settings.openai_model == "test-model"
    assert settings.chunk_size == 1000
    assert settings.embedding_model == "test/embedding"
    assert settings.embedding_query_prefix == "query-task:"


def test_settings_reject_invalid_limits() -> None:
    try:
        Settings(_env_file=None, max_upload_size_mb=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Geçersiz dosya boyutu sınırı reddedilmeliydi.")


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    try:
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)
    except ValueError:
        pass
    else:
        raise AssertionError("Parça boyutuna eşit overlap reddedilmeliydi.")
