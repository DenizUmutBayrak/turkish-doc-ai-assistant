"""Per-session in-memory document indexes with expiry and isolation."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from backend.core.config import Settings
from backend.services.retrieval import InMemoryVectorIndex


@dataclass(frozen=True)
class StoredDocument:
    """Non-content metadata kept for the current in-memory session."""

    filename: str
    page_count: int
    chunk_count: int


@dataclass
class SessionState:
    """Mutable data owned by exactly one unpredictable session identifier."""

    session_id: str
    index: InMemoryVectorIndex
    created_at: float
    last_access: float
    documents: list[StoredDocument] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class SessionLimitError(RuntimeError):
    """The configured maximum number of active in-memory sessions was reached."""


class SessionStore:
    """Thread-safe session registry; documents are never persisted to disk."""

    def __init__(
        self,
        settings: Settings,
        index_factory: Callable[[Settings], InMemoryVectorIndex] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._index_factory = index_factory or (lambda config: InMemoryVectorIndex(settings=config))
        self._clock = clock
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def create(self) -> SessionState:
        with self._lock:
            self._purge_expired_locked()
            if len(self._sessions) >= self.settings.max_active_sessions:
                raise SessionLimitError(
                    "Etkin oturum sınırına ulaşıldı; daha sonra tekrar deneyin."
                )
            now = self._clock()
            session_id = secrets.token_urlsafe(32)
            session = SessionState(
                session_id=session_id,
                index=self._index_factory(self.settings),
                created_at=now,
                last_access=now,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_access = self._clock()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        with session.lock:
            session.index.clear()
            session.documents.clear()
        return True

    def _purge_expired_locked(self) -> None:
        expires_after = self.settings.session_ttl_minutes * 60
        now = self._clock()
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_access >= expires_after
        ]
        expired = [self._sessions.pop(session_id) for session_id in expired_ids]
        for session in expired:
            with session.lock:
                session.index.clear()
                session.documents.clear()
