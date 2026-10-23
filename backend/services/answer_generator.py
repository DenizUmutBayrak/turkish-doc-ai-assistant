"""Provider-neutral contract for a single non-streaming answer generation."""

from typing import Protocol


class AnswerGenerationError(RuntimeError):
    """Provider-neutral generation failure with a sanitized message."""


class AnswerGenerator(Protocol):
    """Minimal interface consumed by the RAG service."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one answer from the supplied instructions and prompt."""
