"""Opt-in integration test; downloads the configured model on first execution."""

import os
from math import isfinite, sqrt

import pytest

from backend.core.config import Settings
from backend.models.documents import TextChunk
from backend.services.embedding_service import (
    EmbeddingInputTooLongError,
    EmbeddingService,
)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_MODEL_INTEGRATION") != "1",
    reason="Model indirme/çalıştırma için RUN_MODEL_INTEGRATION=1 ayarlayın.",
)
def test_real_e5_model_can_embed_query_and_document() -> None:
    service = EmbeddingService(settings=Settings())
    assert service.settings.embedding_device == "cpu"
    assert service.settings.embedding_query_prefix == "query:"
    assert service.settings.embedding_document_prefix == "passage:"
    chunk = TextChunk(
        text="İade işlemi ürünü teslim aldıktan sonra otuz gün içinde yapılabilir.",
        filename="ornek.pdf",
        page_number=1,
        chunk_index=0,
    )

    document_vector = service.embed_chunks([chunk])[0].vector
    query_vector = service.embed_query("Ürünü geri vermek için kaç günüm var?")
    model = service._backend._get_model()

    assert str(model.device).startswith("cpu")
    assert len(document_vector) == len(query_vector) == model.get_embedding_dimension() == 384
    assert all(isfinite(value) for value in (*document_vector, *query_vector))
    assert sqrt(sum(value * value for value in document_vector)) == pytest.approx(1.0, abs=1e-4)
    assert sqrt(sum(value * value for value in query_vector)) == pytest.approx(1.0, abs=1e-4)

    with pytest.raises(EmbeddingInputTooLongError, match="sessizce kesilmedi"):
        service.embed_query("güvenli içerik " * 600)
