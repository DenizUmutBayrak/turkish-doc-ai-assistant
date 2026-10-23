"""Yerel embedding üretimi için küçük, sağlayıcıdan bağımsız arayüz."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from typing import Any, Protocol

from backend.core.config import Settings, get_settings
from backend.models.documents import TextChunk


class EmbeddingServiceError(ValueError):
    """Embedding girdisi veya model çalıştırması güvenli biçimde tamamlanamadı."""


class EmbeddingModelLoadError(EmbeddingServiceError):
    """Yerel model bağımlılığı veya model dosyaları yüklenemedi."""


class EmbeddingInputTooLongError(EmbeddingServiceError):
    """Metin, modelin desteklediği token sınırını aşıyor."""


class EmbeddingDimensionError(EmbeddingServiceError):
    """Embedding vektörlerinin sayısı veya boyutları beklenenle uyuşmuyor."""


class TextEmbedder(Protocol):
    """Embedding altyapısının uygulaması gereken tek işlem."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Her giriş metni için aynı boyutta bir vektör döndürür."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """Vektör üretildikten sonra da kaynak metadata'sını taşıyan chunk."""

    chunk: TextChunk
    vector: tuple[float, ...]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str) -> Any:
    """İlk istekte modeli yükler; aynı model/cihaz çifti tekrar kullanılabilir."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingModelLoadError(
            "Yerel embedding bağımlılıkları kurulu değil. "
            'Model desteği için `python -m pip install -e ".[ml]"` çalıştırın.'
        ) from exc

    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception as exc:
        raise EmbeddingModelLoadError(
            "Yerel embedding modeli cache içinde bulunamadı veya yüklenemedi. "
            "README'deki model indirme adımlarını uygulayın."
        ) from exc


class SentenceTransformerBackend:
    """sentence-transformers kütüphanesini servisten ayıran yerel adapter."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        model_loader: Any = _load_sentence_transformer,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model_loader = model_loader
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self._model_loader(self.model_name, self.device)
            except EmbeddingModelLoadError:
                raise
            except Exception as exc:
                raise EmbeddingModelLoadError(
                    "Yerel embedding modeli yüklenemedi. Model adını ve yerel kurulum "
                    "koşullarını kontrol edin."
                ) from exc
        return self._model

    @staticmethod
    def _check_token_limits(model: Any, texts: Sequence[str]) -> None:
        try:
            max_tokens = model.max_seq_length
        except Exception as exc:
            raise EmbeddingServiceError(
                "Embedding modelinin token sınırı doğrulanamadı; metin güvenle işlenemiyor."
            ) from exc
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise EmbeddingServiceError(
                "Embedding modelinin token sınırı doğrulanamadı; metin güvenle işlenemiyor."
            )

        try:
            for text in texts:
                tokenized = model.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
                token_count = len(tokenized["input_ids"])
                if token_count > max_tokens:
                    raise EmbeddingInputTooLongError(
                        f"Metin {token_count} token içeriyor; model sınırı {max_tokens}. "
                        "Chunk boyutunu küçültün. Metin sessizce kesilmedi."
                    )
        except EmbeddingInputTooLongError:
            raise
        except Exception as exc:
            raise EmbeddingServiceError("Metnin model token sınırı güvenle doğrulanamadı.") from exc

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        model = self._get_model()
        self._check_token_limits(model, texts)
        try:
            return model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingServiceError("Embedding vektörleri oluşturulamadı.") from exc


class EmbeddingService:
    """E5 belge/sorgu öneklerini ve vektör doğrulamasını yöneten servis."""

    def __init__(
        self,
        backend: TextEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._backend: TextEmbedder = (
            backend
            if backend is not None
            else SentenceTransformerBackend(
                model_name=self.settings.embedding_model,
                device=self.settings.embedding_device,
                batch_size=self.settings.embedding_batch_size,
            )
        )
        self._dimension: int | None = None

    @staticmethod
    def _with_prefix(prefix: str, text: str) -> str:
        normalized_prefix = prefix.strip()
        return f"{normalized_prefix} {text}" if normalized_prefix else text

    def _encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        try:
            raw_vectors = self._backend.encode(texts)
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            raise EmbeddingServiceError("Embedding vektörleri oluşturulamadı.") from exc

        if hasattr(raw_vectors, "tolist"):
            raw_vectors = raw_vectors.tolist()
        if isinstance(raw_vectors, (str, bytes)):
            raise EmbeddingDimensionError("Model geçersiz embedding biçimi döndürdü.")
        try:
            rows = list(raw_vectors)
        except TypeError as exc:
            raise EmbeddingDimensionError("Model embedding listesi döndürmedi.") from exc
        if len(rows) != len(texts):
            raise EmbeddingDimensionError(
                f"{len(texts)} metin için {len(rows)} embedding döndürüldü."
            )

        vectors: list[tuple[float, ...]] = []
        dimension: int | None = None
        for row in rows:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (str, bytes)):
                raise EmbeddingDimensionError("Model geçersiz vektör biçimi döndürdü.")
            try:
                vector = tuple(float(value) for value in row)
            except (TypeError, ValueError) as exc:
                raise EmbeddingDimensionError("Embedding vektörü sayısal değil.") from exc
            if not vector:
                raise EmbeddingDimensionError("Embedding vektörü boş.")
            if not all(isfinite(value) for value in vector):
                raise EmbeddingDimensionError("Embedding vektöründe geçersiz sayı var.")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingDimensionError("Embedding vektör boyutları birbiriyle uyuşmuyor.")
            vectors.append(vector)

        if dimension is None:
            raise EmbeddingDimensionError("Model boş embedding sonucu döndürdü.")
        if self._dimension is not None and dimension != self._dimension:
            raise EmbeddingDimensionError(
                f"Embedding boyutu {dimension}; önceki boyut {self._dimension} ile uyuşmuyor."
            )
        self._dimension = dimension
        return vectors

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        """Belge parçalarını E5 passage önekiyle toplu biçimde vektörleştirir."""

        if not chunks:
            return []
        for chunk in chunks:
            if not chunk.text.strip():
                raise EmbeddingServiceError("Boş belge parçası embedding'e dönüştürülemez.")

        vectors = self._encode(
            [
                self._with_prefix(self.settings.embedding_document_prefix, chunk.text)
                for chunk in chunks
            ]
        )
        return [EmbeddedChunk(chunk=chunk, vector=vector) for chunk, vector in zip(chunks, vectors)]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Arama sorgusunu E5 query önekiyle vektörleştirir."""

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingServiceError("Arama sorgusu boş olamaz.")
        text = self._with_prefix(self.settings.embedding_query_prefix, query.strip())
        return self._encode([text])[0]
