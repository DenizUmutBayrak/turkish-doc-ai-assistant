"""PDF sayfalarını ve belge parçalarını temsil eden veri modelleri."""

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    """PDF içindeki tek bir sayfanın çıkarılmış metni ve kaynak bilgisi."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str


class TextChunk(BaseModel):
    """Arama için hazırlanmış metin parçası ve kaynak bilgisi."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class SearchResult(BaseModel):
    """Semantik arama sonucu; similarity bir olasılık veya doğruluk yüzdesi değildir."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    similarity: float = Field(ge=-1.0, le=1.0)


class AnswerSource(BaseModel):
    """Validated retrieval source cited by an answer."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    similarity: float = Field(ge=-1.0, le=1.0)


class RagAnswer(BaseModel):
    """Generated answer and only the retrieval sources validated for it."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: tuple[AnswerSource, ...] = ()
