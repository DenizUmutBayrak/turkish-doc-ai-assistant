"""Deterministik vektörlerle bellek içi retrieval testleri."""

from collections.abc import Sequence

import pytest

from backend.core.config import Settings
from backend.models.documents import TextChunk
from backend.services.embedding_service import (
    EmbeddingDimensionError,
    EmbeddingService,
)
from backend.services.retrieval import InMemoryVectorIndex


class FixtureEmbedder:
    """Önceden belirlenmiş fixture vektörleri döndürür; anlamsal model değildir."""

    vectors = {
        "passage: Garanti süresi iki yıldır.": [1.0, 0.0, 0.0],
        "passage: İade otuz gün içinde yapılabilir.": [0.0, 1.0, 0.0],
        "passage: Teknik destek e-posta ile verilir.": [0.0, 0.0, 1.0],
        "query: Ürünü geri göndermek için kaç gün var?": [0.0, 1.0, 0.0],
        "query: Teknik yardıma nasıl ulaşırım?": [0.0, 0.0, 1.0],
        "query: Bilinmeyen konu": [1.0, 1.0, 0.0],
        "query: Farklı boyut": [1.0, 0.0],
    }

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.inputs.extend(texts)
        return [self.vectors[text] for text in texts]


def make_chunk(
    text: str,
    filename: str,
    page_number: int,
    chunk_index: int,
) -> TextChunk:
    return TextChunk(
        text=text,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
    )


@pytest.fixture
def fixture_index() -> tuple[InMemoryVectorIndex, FixtureEmbedder]:
    backend = FixtureEmbedder()
    service = EmbeddingService(backend=backend, settings=Settings(_env_file=None))
    index = InMemoryVectorIndex(service, Settings(_env_file=None, top_k=2))
    index.index_chunks(
        [
            make_chunk("Garanti süresi iki yıldır.", "garanti.pdf", 3, 0),
            make_chunk("İade otuz gün içinde yapılabilir.", "iade.pdf", 8, 1),
            make_chunk("Teknik destek e-posta ile verilir.", "destek.pdf", 12, 2),
        ]
    )
    return index, backend


def test_turkish_paraphrase_finds_matching_fixture_passage(
    fixture_index: tuple[InMemoryVectorIndex, FixtureEmbedder],
) -> None:
    index, backend = fixture_index

    result = index.search("Ürünü geri göndermek için kaç gün var?", top_k=1)

    assert len(result) == 1
    assert result[0].filename == "iade.pdf"
    assert result[0].page_number == 8
    assert result[0].text == "İade otuz gün içinde yapılabilir."
    assert backend.inputs[-1] == "query: Ürünü geri göndermek için kaç gün var?"


def test_results_sort_by_cosine_similarity_and_keep_sources(
    fixture_index: tuple[InMemoryVectorIndex, FixtureEmbedder],
) -> None:
    index, _ = fixture_index

    results = index.search("Teknik yardıma nasıl ulaşırım?", top_k=3)

    assert [result.filename for result in results] == ["destek.pdf", "garanti.pdf", "iade.pdf"]
    assert [result.page_number for result in results] == [12, 3, 8]
    assert [result.similarity for result in results] == pytest.approx([1.0, 0.0, 0.0])
    assert all(-1.0 <= result.similarity <= 1.0 for result in results)


def test_top_k_is_applied_and_ties_are_deterministic(
    fixture_index: tuple[InMemoryVectorIndex, FixtureEmbedder],
) -> None:
    index, _ = fixture_index

    results = index.search("Bilinmeyen konu", top_k=2)

    assert [result.filename for result in results] == ["garanti.pdf", "iade.pdf"]
    assert [result.similarity for result in results] == pytest.approx([2**-0.5, 2**-0.5])


def test_empty_index_returns_empty_without_loading_embeddings() -> None:
    backend = FixtureEmbedder()
    index = InMemoryVectorIndex(
        EmbeddingService(backend, Settings(_env_file=None)),
        Settings(_env_file=None),
    )

    assert index.search("Teknik yardıma nasıl ulaşırım?") == []
    assert backend.inputs == []


def test_empty_query_top_k_and_dimension_mismatch_are_safe(
    fixture_index: tuple[InMemoryVectorIndex, FixtureEmbedder],
) -> None:
    index, _ = fixture_index

    with pytest.raises(ValueError, match="sorgusu boş"):
        index.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        index.search("Teknik yardıma nasıl ulaşırım?", top_k=0)
    with pytest.raises(EmbeddingDimensionError, match="boyutu"):
        index.search("Farklı boyut")


def test_separate_indexes_do_not_share_document_chunks() -> None:
    backend_a = FixtureEmbedder()
    index_a = InMemoryVectorIndex(
        EmbeddingService(backend_a, Settings(_env_file=None)),
        Settings(_env_file=None),
    )
    index_a.index_chunks([make_chunk("İade otuz gün içinde yapılabilir.", "a.pdf", 1, 0)])
    backend_b = FixtureEmbedder()
    index_b = InMemoryVectorIndex(
        EmbeddingService(backend_b, Settings(_env_file=None)),
        Settings(_env_file=None),
    )

    assert index_a.search("Ürünü geri göndermek için kaç gün var?", 1)[0].filename == "a.pdf"
    assert index_b.search("Ürünü geri göndermek için kaç gün var?", 1) == []


def test_clear_removes_index_data(
    fixture_index: tuple[InMemoryVectorIndex, FixtureEmbedder],
) -> None:
    index, _ = fixture_index

    index.clear()

    assert index.search("Teknik yardıma nasıl ulaşırım?") == []


def test_zero_vectors_are_rejected_for_cosine_similarity() -> None:
    backend = FixtureEmbedder()
    backend.vectors["passage: Garanti süresi iki yıldır."] = [0.0, 0.0, 0.0]
    service = EmbeddingService(backend, Settings(_env_file=None))
    index = InMemoryVectorIndex(service, Settings(_env_file=None))
    index.index_chunks([make_chunk("Garanti süresi iki yıldır.", "garanti.pdf", 3, 0)])

    with pytest.raises(ValueError, match="sıfır vektör"):
        index.search("Teknik yardıma nasıl ulaşırım?")
