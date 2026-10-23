"""Sayfa metinlerini kaynak bilgisini koruyarak karakter tabanlı böler."""

from backend.core.config import Settings, get_settings
from backend.models.documents import PageText, TextChunk


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Tek metni, örtüşmeli sabit karakter pencerelerine ayırır.

    Dilimlerin ardışık birleştirilmesi overlap nedeniyle tekrar içerir; buna
    karşılık özgün metindeki her karakter en az bir parçada bulunur.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size sıfırdan büyük olmalıdır.")
    if overlap < 0:
        raise ValueError("overlap negatif olamaz.")
    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size değerinden küçük olmalıdır.")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap

    return chunks


def chunk_pages(pages: list[PageText], settings: Settings | None = None) -> list[TextChunk]:
    """Her sayfayı ayrı parçalara böler ve kaynak metadata'sını korur."""

    active_settings = settings or get_settings()
    # Settings modeli de bu ilişkiyi doğrular; burada açık kontrol, bu fonksiyonun
    # güvenli ilerleme garantisini ayarlardan bağımsız olarak korur.
    if active_settings.chunk_overlap >= active_settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük olmalıdır.")

    result: list[TextChunk] = []
    chunk_index = 0
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=active_settings.chunk_size,
            overlap=active_settings.chunk_overlap,
        )
        for text in page_chunks:
            result.append(
                TextChunk(
                    text=text,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    return result
