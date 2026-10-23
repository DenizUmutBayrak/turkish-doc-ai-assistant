"""Offline test doubles for the embedding service; no model download is needed."""

import sys
from collections.abc import Sequence
from types import ModuleType

import pytest

from backend.core.config import Settings
from backend.models.documents import TextChunk
from backend.services.embedding_service import (
    EmbeddingDimensionError,
    EmbeddingInputTooLongError,
    EmbeddingModelLoadError,
    EmbeddingService,
    EmbeddingServiceError,
    SentenceTransformerBackend,
    _load_sentence_transformer,
)


class FakeEmbedder:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = list(vectors)
        self.inputs: list[str] = []

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.inputs.extend(texts)
        return self.vectors


def make_chunk(text: str = "Garanti iki yıldır.", index: int = 0) -> TextChunk:
    return TextChunk(
        text=text,
        filename="urun_rehberi.pdf",
        page_number=7,
        chunk_index=index,
    )


def test_embeds_chunks_with_passage_prefix_and_preserves_metadata() -> None:
    backend = FakeEmbedder([[1.0, 0.0, 0.0]])
    service = EmbeddingService(backend=backend, settings=Settings(_env_file=None))
    chunk = make_chunk()

    embedded = service.embed_chunks([chunk])

    assert backend.inputs == ["passage: Garanti iki yıldır."]
    assert embedded == [type(embedded[0])(chunk=chunk, vector=(1.0, 0.0, 0.0))]


def test_embeds_query_with_query_prefix() -> None:
    backend = FakeEmbedder([[0.0, 1.0]])
    service = EmbeddingService(backend=backend, settings=Settings(_env_file=None))

    vector = service.embed_query(" İade süresi kaç gün? ")

    assert backend.inputs == ["query: İade süresi kaç gün?"]
    assert vector == (0.0, 1.0)


def test_empty_chunk_batch_does_not_call_backend() -> None:
    backend = FakeEmbedder([])
    service = EmbeddingService(backend=backend, settings=Settings(_env_file=None))

    assert service.embed_chunks([]) == []
    assert backend.inputs == []


def test_default_service_defers_model_loading_until_first_embedding() -> None:
    service = EmbeddingService(settings=Settings(_env_file=None))

    assert service._backend._model is None


@pytest.mark.parametrize("query", ["", "  ", None])
def test_empty_or_invalid_query_is_rejected(query: str | None) -> None:
    service = EmbeddingService(backend=FakeEmbedder([]), settings=Settings(_env_file=None))

    with pytest.raises(EmbeddingServiceError, match="boş"):
        service.embed_query(query)  # type: ignore[arg-type]


def test_whitespace_only_chunk_is_rejected() -> None:
    service = EmbeddingService(backend=FakeEmbedder([]), settings=Settings(_env_file=None))

    with pytest.raises(EmbeddingServiceError, match="Boş belge"):
        service.embed_chunks([make_chunk(text="   ")])


def test_embedding_count_and_vector_dimensions_are_validated() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(EmbeddingDimensionError, match="2 metin"):
        EmbeddingService(FakeEmbedder([[1.0, 0.0]]), settings).embed_chunks(
            [make_chunk(), make_chunk(index=1)]
        )

    with pytest.raises(EmbeddingDimensionError, match="boyutları"):
        EmbeddingService(FakeEmbedder([[1.0, 0.0], [1.0, 0.0, 0.0]]), settings).embed_chunks(
            [make_chunk(), make_chunk(index=1)]
        )


def test_embedding_dimension_must_match_between_calls() -> None:
    class ChangesDimension:
        def __init__(self) -> None:
            self.call = 0

        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            self.call += 1
            return [[1.0, 0.0]] if self.call == 1 else [[1.0, 0.0, 0.0]]

    service = EmbeddingService(ChangesDimension(), Settings(_env_file=None))
    service.embed_chunks([make_chunk()])

    with pytest.raises(EmbeddingDimensionError, match="önceki boyut"):
        service.embed_query("soru")


def test_non_finite_vectors_are_rejected() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(EmbeddingDimensionError, match="geçersiz sayı"):
        EmbeddingService(FakeEmbedder([[float("nan")]]), settings).embed_chunks([make_chunk()])


def test_model_load_failure_is_reported_without_network_access() -> None:
    def fail_loading(_model_name: str, _device: str) -> object:
        raise RuntimeError("internal load details")

    backend = SentenceTransformerBackend("fake/model", model_loader=fail_loading)

    with pytest.raises(EmbeddingModelLoadError, match="yüklenemedi"):
        backend.encode(["query: deneme"])


def test_default_model_loader_uses_only_cached_model_files(monkeypatch) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs) -> None:
            calls["model_name"] = model_name
            calls.update(kwargs)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    _load_sentence_transformer.cache_clear()
    try:
        _load_sentence_transformer("local/model", "cpu")
    finally:
        _load_sentence_transformer.cache_clear()

    assert calls == {"model_name": "local/model", "device": "cpu", "local_files_only": True}


def test_model_is_loaded_once_per_backend_instance() -> None:
    class FakeModel:
        max_seq_length = 20
        tokenizer = staticmethod(lambda text, **_kwargs: {"input_ids": text.split()})

        @staticmethod
        def encode(texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    loads = 0

    def load_model(_name: str, _device: str) -> FakeModel:
        nonlocal loads
        loads += 1
        return FakeModel()

    backend = SentenceTransformerBackend("fake/model", model_loader=load_model)

    backend.encode(["query: one"])
    backend.encode(["passage: two"])

    assert loads == 1


def test_long_input_is_rejected_before_model_can_truncate_it() -> None:
    class FakeModel:
        max_seq_length = 3
        tokenizer = staticmethod(lambda _text, **_kwargs: {"input_ids": [1, 2, 3, 4]})

        @staticmethod
        def encode(*_args, **_kwargs):
            raise AssertionError("Uzun girdi modele gönderilmemeliydi.")

    backend = SentenceTransformerBackend(
        "fake/model", model_loader=lambda _name, _device: FakeModel()
    )

    with pytest.raises(EmbeddingInputTooLongError, match="sessizce kesilmedi"):
        backend.encode(["query: çok uzun"])
