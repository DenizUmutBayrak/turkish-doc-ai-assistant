"""Stable request and response schemas for the local API."""

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    expires_in_seconds: int


class DocumentSummary(BaseModel):
    filename: str
    page_count: int
    chunk_count: int


class DocumentUploadResponse(BaseModel):
    document: DocumentSummary
    documents: list[DocumentSummary]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)


class SourceResponse(BaseModel):
    source_id: str
    filename: str
    page_number: int
    chunk_index: int
    similarity: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class DeleteSessionResponse(BaseModel):
    deleted: bool
    message: str
