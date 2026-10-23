"""Ortam değişkenlerinden uygulama yapılandırmasını yükler."""

from functools import lru_cache
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Uygulama ayarları; gizli anahtarlar varsayılan olarak dışarı sızdırılmaz."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_query_prefix: str = "query:"
    embedding_document_prefix: str = "passage:"
    ollama_allow_remote: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout_seconds: int = Field(default=120, ge=1, le=600)
    ollama_think: bool = False
    ollama_num_predict: int = Field(default=512, ge=1, le=4096)
    rag_max_context_chars: int = Field(default=16000, ge=512, le=100000)
    rag_max_question_chars: int = Field(default=2000, ge=1, le=10000)
    api_base_url: str = "http://localhost:8000"
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)
    max_pdf_pages: int = Field(default=300, ge=1, le=2000)
    max_session_files: int = Field(default=10, ge=1, le=100)
    max_session_chunks: int = Field(default=2000, ge=1, le=20000)
    max_active_sessions: int = Field(default=100, ge=1, le=10000)
    session_ttl_minutes: int = Field(default=60, ge=1, le=10080)
    chunk_size: int = Field(default=800, ge=100, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str, info) -> str:
        """Remote Ollama hosts require an explicit opt-in setting."""

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL geçerli bir HTTP(S) adresi olmalıdır.")
        if parsed.username or parsed.password:
            raise ValueError("OLLAMA_BASE_URL içinde kullanıcı bilgisi kullanılamaz.")
        try:
            loopback = (
                parsed.hostname.lower() == "localhost" or ip_address(parsed.hostname).is_loopback
            )
        except ValueError:
            loopback = False
        if not loopback and not info.data.get("ollama_allow_remote", False):
            raise ValueError(
                "Uzak Ollama adresi için OLLAMA_ALLOW_REMOTE=true açıkça ayarlanmalıdır."
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Overlap, her parçanın boyutundan küçük olmalı ki ilerleme sağlansın."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük olmalıdır.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Tek bir yapılandırma nesnesi döndürür; testler önbelleği temizleyebilir."""

    return Settings()
