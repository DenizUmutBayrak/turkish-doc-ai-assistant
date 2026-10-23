"""Karakter tabanlı chunking için deterministik birim testleri."""

import pytest

from backend.core.config import Settings
from backend.models.documents import PageText
from backend.services.chunker import chunk_pages, chunk_text


def test_splits_text_into_expected_chunk_lengths() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]
    assert all(len(chunk) <= 4 for chunk in chunks)


def test_neighboring_chunks_have_requested_overlap_without_losing_characters() -> None:
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
    rebuilt = chunks[0] + "".join(chunk[2:] for chunk in chunks[1:])
    assert rebuilt == text


def test_empty_and_short_text_are_handled() -> None:
    assert chunk_text("", chunk_size=5, overlap=2) == []
    assert chunk_text("abc", chunk_size=5, overlap=2) == ["abc"]


def test_page_metadata_is_preserved_and_pages_are_not_joined() -> None:
    pages = [
        PageText(filename="rehber.pdf", page_number=3, text="a" * 220),
        PageText(filename="rehber.pdf", page_number=4, text="xy"),
    ]
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)

    chunks = chunk_pages(pages, settings=settings)

    assert [(len(chunk.text), chunk.page_number) for chunk in chunks] == [
        (100, 3),
        (100, 3),
        (60, 3),
        (2, 4),
    ]
    assert all(chunk.filename == "rehber.pdf" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_settings_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("metin", chunk_size=chunk_size, overlap=overlap)


def test_configuration_rejects_overlap_that_prevents_progress() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
