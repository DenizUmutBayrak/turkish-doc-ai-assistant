"""PDF dosyalarından sayfa metni çıkarma."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pypdf import PdfReader

from backend.core.config import Settings, get_settings
from backend.models.documents import PageText


class PdfProcessingError(ValueError):
    """PDF kullanıcı girdisi işlenemediğinde gösterilecek güvenli hata."""


PdfSource = bytes | bytearray | memoryview | str | Path | BinaryIO


def _display_filename(filename: str | None, source: PdfSource) -> str:
    if filename is None:
        if isinstance(source, (str, Path)):
            filename = Path(source).name
        else:
            filename = "belge.pdf"

    safe_name = PurePosixPath(filename.replace("\\", "/")).name
    return safe_name or "belge.pdf"


def _read_pdf_bytes(source: PdfSource, max_bytes: int) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise PdfProcessingError("PDF dosyasına erişilemedi.") from exc
        if file_size == 0:
            raise PdfProcessingError("PDF dosyası boş.")
        if file_size > max_bytes:
            raise PdfProcessingError("PDF izin verilen dosya boyutu sınırını aşıyor.")
        try:
            with path.open("rb") as pdf_file:
                content = pdf_file.read(max_bytes + 1)
        except OSError as exc:
            raise PdfProcessingError("PDF dosyası okunamadı.") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        content = bytes(source)
    else:
        try:
            content = source.read(max_bytes + 1)
        except (OSError, ValueError) as exc:
            raise PdfProcessingError("Yüklenen PDF okunamadı.") from exc

    if not isinstance(content, bytes):
        raise PdfProcessingError("PDF ikili dosya biçiminde olmalıdır.")
    if not content:
        raise PdfProcessingError("PDF dosyası boş.")
    if len(content) > max_bytes:
        raise PdfProcessingError("PDF izin verilen dosya boyutu sınırını aşıyor.")
    return content


def load_pdf_pages(
    source: PdfSource,
    filename: str | None = None,
    settings: Settings | None = None,
) -> list[PageText]:
    """PDF sayfalarını metin ve kaynak metadatasıyla döndürür.

    ``source`` dosya yolu, bytes veya ikili dosya benzeri nesne olabilir. Dosya
    adı yalnızca kaynak etiketi olarak kullanılır; dosya yolu oluşturmak için
    kullanılmaz. Boş sayfalar, sayfa numarası korunarak boş metinle döner.
    """

    active_settings = settings or get_settings()
    safe_filename = _display_filename(filename, source)
    max_bytes = active_settings.max_upload_size_mb * 1024 * 1024
    pdf_bytes = _read_pdf_bytes(source, max_bytes)

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=True)
    except Exception as exc:
        raise PdfProcessingError("PDF bozuk veya okunamıyor.") from exc

    try:
        if reader.is_encrypted:
            raise PdfProcessingError("Şifreli PDF dosyaları bu uygulamada desteklenmiyor.")

        page_count = len(reader.pages)
    except PdfProcessingError:
        raise
    except Exception as exc:
        raise PdfProcessingError("PDF sayfaları okunamıyor.") from exc
    if page_count == 0:
        raise PdfProcessingError("PDF dosyasında sayfa bulunamadı.")
    if page_count > active_settings.max_pdf_pages:
        raise PdfProcessingError("PDF izin verilen sayfa sayısı sınırını aşıyor.")

    extracted_pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PdfProcessingError(
                f"PDF'nin {page_number}. sayfasındaki metin okunamadı."
            ) from exc
        extracted_pages.append(PageText(filename=safe_filename, page_number=page_number, text=text))

    return extracted_pages
