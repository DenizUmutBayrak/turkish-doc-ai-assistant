"""Opt-in end-to-end RAG test against a real local Ollama model."""

import json
import os

import pytest

from backend.core.config import Settings
from backend.models.documents import TextChunk
from backend.services.rag_service import RagService
from backend.services.retrieval import InMemoryVectorIndex


@pytest.mark.ollama_integration
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Yerel Ollama testi için RUN_OLLAMA_INTEGRATION=1 ayarlayın.",
)
def test_real_ollama_rag_generates_answer_with_retrieved_source() -> None:
    """Exercise real local embeddings, retrieval, and configured Ollama generation."""
    settings = Settings(_env_file=None, embedding_device="cpu", top_k=2)
    chunks = [
        TextChunk(
            text=(
                "Sentetik test belgesi: Kuzey Yıldızı termosunun garanti süresi, "
                "satın alma tarihinden itibaren 18 aydır. Garanti başvurusu için "
                "ürünün seri numarası ve satın alma belgesi gerekir."
            ),
            filename="sentetik_garanti.pdf",
            page_number=4,
            chunk_index=0,
        ),
        TextChunk(
            text=(
                "Sentetik test belgesi: Mavi Rota sırt çantası, teslimden sonra "
                "30 gün içinde kullanılmamışsa iade edilebilir."
            ),
            filename="sentetik_iade.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]
    index = InMemoryVectorIndex(settings=settings)
    assert index.index_chunks(chunks) == len(chunks)

    question = "Kuzey Yıldızı termosu için ne kadar süre garanti veriliyor?"
    retrieved = index.search(question, top_k=settings.top_k)
    assert retrieved, "Gerçek embedding/retrieval en az bir sonuç döndürmeli."

    result = RagService(index=index, settings=settings).answer(question)

    assert result.answer.strip()
    assert result.sources, "Gerçek model yanıtı en az bir doğrulanmış kaynak taşımalı."
    retrieved_sources = {
        (item.filename, item.page_number, item.chunk_index, item.similarity) for item in retrieved
    }
    for source in result.sources:
        assert (
            source.filename,
            source.page_number,
            source.chunk_index,
            source.similarity,
        ) in retrieved_sources

    print(
        "REAL_OLLAMA_RAG="
        + json.dumps(
            {
                "model": settings.ollama_model,
                "answer_nonempty": bool(result.answer.strip()),
                "source_count": len(result.sources),
                "sources": [
                    {
                        "filename": source.filename,
                        "page": source.page_number,
                        "chunk_index": source.chunk_index,
                        "similarity": round(source.similarity, 4),
                    }
                    for source in result.sources
                ],
            },
            ensure_ascii=False,
        )
    )
