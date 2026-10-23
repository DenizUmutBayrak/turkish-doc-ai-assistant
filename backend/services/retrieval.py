"""Belge parçaları için süreç içi cosine similarity araması."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite, sqrt

from backend.core.config import Settings, get_settings
from backend.models.documents import SearchResult, TextChunk
from backend.services.embedding_service import (
    EmbeddedChunk,
    EmbeddingDimensionError,
    EmbeddingService,
    EmbeddingServiceError,
)


class InMemoryVectorIndex:
    """Tek belge kümesine ait küçük, kalıcı olmayan semantik arama indeksi.

    Oturumlar arasında veri paylaşımını önlemek için her oturum kendi indeks
    örneğini kullanmalıdır. İçerik yalnızca bu Python nesnesinde bellekte tutulur.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService(settings=self.settings)
        )
        self._items: list[EmbeddedChunk] = []
        self._dimension: int | None = None

    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        """İndeksi verilen chunk kümesiyle değiştirip parça sayısını döndürür."""

        embedded = self.embedding_service.embed_chunks(chunks)
        dimension = len(embedded[0].vector) if embedded else None
        if dimension is not None and any(len(item.vector) != dimension for item in embedded):
            raise EmbeddingDimensionError("İndekslenecek vektör boyutları uyuşmuyor.")
        self._items = list(embedded)
        self._dimension = dimension
        return len(self._items)

    def add_chunks(self, chunks: Sequence[TextChunk]) -> int:
        """Yeni chunk'ları ekler ve toplam indeks boyutunu döndürür."""

        embedded = self.embedding_service.embed_chunks(chunks)
        if not embedded:
            return len(self._items)
        new_dimension = len(embedded[0].vector)
        if any(len(item.vector) != new_dimension for item in embedded):
            raise EmbeddingDimensionError("Eklenecek vektör boyutları uyuşmuyor.")
        if self._dimension is not None and new_dimension != self._dimension:
            raise EmbeddingDimensionError(
                "Yeni embedding boyutu indeksle uyuşmuyor; aynı modelle indeksi yeniden oluşturun."
            )
        self._items.extend(embedded)
        self._dimension = new_dimension
        return len(self._items)

    def clear(self) -> None:
        """İndeksin bu örnekte tuttuğu belge metni ve vektörleri temizler."""

        self._items.clear()
        self._dimension = None

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """En benzer parçaları cosine similarity azalan sırada döndürür."""

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingServiceError("Arama sorgusu boş olamaz.")
        result_limit = self.settings.top_k if top_k is None else top_k
        if not isinstance(result_limit, int) or isinstance(result_limit, bool) or result_limit < 1:
            raise ValueError("top_k en az 1 olmalıdır.")
        if not self._items:
            return []

        query_vector = self.embedding_service.embed_query(query)
        if self._dimension is None or len(query_vector) != self._dimension:
            raise EmbeddingDimensionError(
                "Sorgu embedding boyutu belge indeksiyle uyuşmuyor; indeksi aynı modelle kurun."
            )

        results = [
            SearchResult(
                text=item.chunk.text,
                filename=item.chunk.filename,
                page_number=item.chunk.page_number,
                chunk_index=item.chunk.chunk_index,
                similarity=_cosine_similarity(query_vector, item.vector),
            )
            for item in self._items
        ]
        results.sort(
            key=lambda result: (
                -result.similarity,
                result.filename,
                result.page_number,
                result.chunk_index,
                result.text,
            )
        )
        return results[:result_limit]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Vektörlerin cosine similarity puanını hesaplar ve girdileri doğrular."""

    if not left or len(left) != len(right):
        raise EmbeddingDimensionError("Cosine similarity için vektör boyutları uyuşmuyor.")
    if not all(isfinite(value) for value in (*left, *right)):
        raise EmbeddingServiceError("Cosine similarity vektöründe geçersiz sayı var.")

    left_norm = sqrt(fsum(value * value for value in left))
    right_norm = sqrt(fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise EmbeddingServiceError("Cosine similarity sıfır vektörle hesaplanamaz.")

    score = fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))
