"""PDF metin çıkarma için küçük, bellekte üretilen PDF'lerle testler."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.core.config import Settings
from backend.models.documents import PageText
from backend.services.pdf_loader import PdfProcessingError, load_pdf_pages


def make_pdf(*page_texts: str, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        if text:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(content)
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extracts_text_from_pdf_page_and_sanitizes_display_name() -> None:
    pages = load_pdf_pages(make_pdf("Merhaba dunya"), filename=r"..\ozel\elkitabi.pdf")

    assert len(pages) == 1
    assert pages[0].text == "Merhaba dunya"
    assert pages[0].filename == "elkitabi.pdf"
    assert pages[0].page_number == 1


def test_page_numbers_are_one_based_and_empty_pages_are_preserved() -> None:
    pages = load_pdf_pages(make_pdf("Ilk sayfa", "", "Son sayfa"), filename="belge.pdf")

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "Ilk sayfa"),
        (2, ""),
        (3, "Son sayfa"),
    ]


def test_rejects_empty_corrupt_and_encrypted_pdfs() -> None:
    with pytest.raises(PdfProcessingError, match="boş"):
        load_pdf_pages(b"")
    with pytest.raises(PdfProcessingError, match="bozuk"):
        load_pdf_pages(b"this is not a pdf")
    with pytest.raises(PdfProcessingError, match="Şifreli"):
        load_pdf_pages(make_pdf("gizli", password="parola"))


def test_enforces_upload_size_and_page_count_limits() -> None:
    small_limit = Settings(_env_file=None, max_upload_size_mb=1)
    with pytest.raises(PdfProcessingError, match="boyutu"):
        load_pdf_pages(b"%PDF " + b"x" * (1024 * 1024), settings=small_limit)

    one_page_limit = Settings(_env_file=None, max_pdf_pages=1)
    with pytest.raises(PdfProcessingError, match="sayfa sayısı"):
        load_pdf_pages(make_pdf("bir", "iki"), settings=one_page_limit)


def test_enforces_size_limit_for_path_inputs(tmp_path) -> None:
    pdf_path = tmp_path / "buyuk.pdf"
    pdf_path.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(PdfProcessingError, match="boyutu"):
        load_pdf_pages(pdf_path, settings=Settings(_env_file=None, max_upload_size_mb=1))


def test_loads_pdf_from_path(tmp_path) -> None:
    pdf_path = tmp_path / "notlar.pdf"
    pdf_path.write_bytes(make_pdf("Dosyadan okundu"))

    pages = load_pdf_pages(pdf_path)

    assert pages[0].filename == "notlar.pdf"
    assert pages[0].text == "Dosyadan okundu"


def test_page_text_requires_valid_source_metadata() -> None:
    with pytest.raises(ValueError):
        PageText(filename="", page_number=0, text="metin")
