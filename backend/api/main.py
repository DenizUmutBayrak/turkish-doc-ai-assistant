"""FastAPI application for the local Turkish document assistant."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    DeleteSessionResponse,
    DocumentSummary,
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    SessionResponse,
    SourceResponse,
)
from backend.api.session_store import (
    SessionLimitError,
    SessionState,
    SessionStore,
    StoredDocument,
)
from backend.core.config import Settings, get_settings
from backend.services.chunker import chunk_pages
from backend.services.embedding_service import EmbeddingServiceError
from backend.services.pdf_loader import PdfProcessingError, load_pdf_pages
from backend.services.rag_service import RagInputError, RagService, RagServiceError
from backend.services.retrieval import InMemoryVectorIndex

IndexFactory = Callable[[Settings], InMemoryVectorIndex]
RagFactory = Callable[[InMemoryVectorIndex, Settings], RagService]
logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    index_factory: IndexFactory | None = None,
    rag_factory: RagFactory | None = None,
) -> FastAPI:
    """Build the API without eagerly loading any embedding or chat model."""
    active_settings = settings or get_settings()
    app = FastAPI(
        title="Türkçe Doküman AI Asistanı API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = active_settings
    app.state.sessions = SessionStore(active_settings, index_factory=index_factory)
    app.state.rag_factory = rag_factory or (
        lambda index, config: RagService(index=index, settings=config)
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "İstek biçimi geçersiz."})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "message": "API çalışıyor; yerel modellerin hazır olduğunu göstermez.",
        }

    @app.post("/api/v1/sessions", response_model=SessionResponse, status_code=201)
    def create_session(request: Request) -> SessionResponse:
        try:
            session = request.app.state.sessions.create()
        except SessionLimitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SessionResponse(
            session_id=session.session_id,
            expires_in_seconds=request.app.state.settings.session_ttl_minutes * 60,
        )

    @app.post(
        "/api/v1/sessions/{session_id}/documents",
        response_model=DocumentUploadResponse,
        status_code=201,
    )
    def upload_document(
        session_id: str,
        request: Request,
        file: UploadFile = File(...),
    ) -> DocumentUploadResponse:
        session = _get_session(request, session_id)
        settings: Settings = request.app.state.settings
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Yalnızca PDF dosyası yükleyebilirsiniz.")

        content = _read_upload(file, settings.max_upload_size_mb * 1024 * 1024)
        with session.lock:
            if len(session.documents) >= settings.max_session_files:
                raise HTTPException(
                    status_code=413, detail="Oturum başına dosya sınırına ulaşıldı."
                )
            try:
                pages = load_pdf_pages(content, filename=filename, settings=settings)
            except PdfProcessingError as exc:
                status_code = 413 if "sınırını aşıyor" in str(exc) else 400
                raise HTTPException(status_code=status_code, detail=str(exc)) from exc

            chunks = chunk_pages(pages, settings=settings)
            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail="PDF'den aranabilir metin çıkarılamadı; taranmış PDF'lerde OCR yoktur.",
                )
            if sum(document.chunk_count for document in session.documents) + len(chunks) > (
                settings.max_session_chunks
            ):
                raise HTTPException(
                    status_code=413, detail="Oturum başına parça sınırına ulaşıldı."
                )

            try:
                session.index.add_chunks(chunks)
            except EmbeddingServiceError as exc:
                _log_safe_failure("embedding_index", exc, chunk_count=len(chunks))
                raise HTTPException(
                    status_code=503,
                    detail="Yerel embedding modeli kullanılamıyor; model kurulumunu kontrol edin.",
                ) from exc
            except Exception as exc:
                _log_safe_failure("vector_index", exc, chunk_count=len(chunks))
                raise HTTPException(
                    status_code=503, detail="PDF yerel olarak indekslenemedi."
                ) from exc

            summary = DocumentSummary(
                filename=pages[0].filename,
                page_count=len(pages),
                chunk_count=len(chunks),
            )
            session.documents.append(
                StoredDocument(
                    filename=summary.filename,
                    page_count=summary.page_count,
                    chunk_count=summary.chunk_count,
                )
            )
            documents = [
                DocumentSummary(
                    filename=item.filename,
                    page_count=item.page_count,
                    chunk_count=item.chunk_count,
                )
                for item in session.documents
            ]
        return DocumentUploadResponse(document=summary, documents=documents)

    @app.post("/api/v1/sessions/{session_id}/questions", response_model=QuestionResponse)
    def ask_question(
        session_id: str,
        body: QuestionRequest,
        request: Request,
    ) -> QuestionResponse:
        session = _get_session(request, session_id)
        if not session.documents:
            raise HTTPException(status_code=409, detail="Önce bu oturuma en az bir PDF yükleyin.")
        try:
            rag: RagService = request.app.state.rag_factory(
                session.index, request.app.state.settings
            )
            with session.lock:
                answer = rag.answer(body.question)
        except RagInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RagServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Yerel yanıt hizmeti kullanılamıyor. Ollama'nın çalıştığını ve "
                    "seçilen modelin kurulu olduğunu kontrol edin."
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Yerel yanıt hizmeti şu anda kullanılamıyor."
            ) from exc
        return QuestionResponse(
            answer=answer.answer,
            sources=[
                SourceResponse(
                    source_id=source.source_id,
                    filename=source.filename,
                    page_number=source.page_number,
                    chunk_index=source.chunk_index,
                    similarity=source.similarity,
                )
                for source in answer.sources
            ],
        )

    @app.delete("/api/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
    def delete_session(session_id: str, request: Request) -> DeleteSessionResponse:
        deleted = request.app.state.sessions.delete(session_id)
        return DeleteSessionResponse(
            deleted=deleted,
            message=(
                "Oturum verileri bellekten temizlendi." if deleted else "Oturum zaten bulunmuyor."
            ),
        )

    return app


def _get_session(request: Request, session_id: str) -> SessionState:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı veya süresi doldu.")
    return session


def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most limit+1 bytes so oversized content is rejected during reading."""
    chunks: list[bytes] = []
    total = 0
    while total <= max_bytes:
        block = file.file.read(min(64 * 1024, max_bytes + 1 - total))
        if not block:
            break
        if not isinstance(block, bytes):
            raise HTTPException(status_code=400, detail="PDF ikili dosya biçiminde olmalıdır.")
        chunks.append(block)
        total += len(block)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="PDF dosya boyutu sınırını aşıyor.")
    if total == 0:
        raise HTTPException(status_code=400, detail="PDF dosyası boş.")
    return b"".join(chunks)


app = create_app()


def _log_safe_failure(stage: str, exc: Exception, *, chunk_count: int) -> None:
    """Log failure location and exception types without content or exception text."""
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    logger.error(
        "PDF indexing failed: stage=%s chunk_count=%d exception_chain=%s",
        stage,
        chunk_count,
        " <- ".join(chain),
    )
