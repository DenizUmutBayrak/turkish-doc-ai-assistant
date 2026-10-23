"""Streamlit UI for the local Turkish document assistant."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any

import httpx
import streamlit as st

from backend.core.config import Settings


class ApiRequestError(RuntimeError):
    """Safe API error that can be shown to the local user."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    """Small HTTP wrapper; session identifiers and messages remain in UI state."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                timeout=httpx.Timeout(180.0, connect=5.0),
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ApiRequestError(
                None,
                "Backend API'ye bağlanılamadı. Ayrı terminalde API'yi başlatıp tekrar deneyin.",
            ) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.is_success:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            message = detail if isinstance(detail, str) else _status_message(response.status_code)
            raise ApiRequestError(response.status_code, message)
        if not isinstance(payload, dict):
            raise ApiRequestError(response.status_code, "Backend beklenmeyen bir yanıt verdi.")
        return payload


def _status_message(status_code: int) -> str:
    messages = {
        404: "Oturum bulunamadı veya süresi doldu. Yeni bir oturum başlatın.",
        413: "Yükleme sınırı aşıldı. Daha küçük bir PDF seçin veya oturumu temizleyin.",
        422: "PDF'den aranabilir metin çıkarılamadı. Taranmış PDF'lerde OCR henüz yoktur.",
        503: (
            "Yerel hizmet kullanılamıyor. Backend'i, embedding modelini ve Ollama'yı kontrol edin."
        ),
    }
    return messages.get(status_code, "İstek tamamlanamadı. Girdiyi kontrol edip tekrar deneyin.")


def _api_client() -> ApiClient:
    base_url = os.getenv("API_BASE_URL") or Settings().api_base_url
    return ApiClient(base_url)


def _fingerprint(content: bytes) -> str:
    """Identify duplicate uploads without retaining document content."""
    return hashlib.sha256(content).hexdigest()


def _new_uploads(
    uploaded_files: Sequence[Any],
    uploaded_fingerprints: set[str],
    *,
    allow_duplicates: bool,
) -> list[tuple[Any, bytes, str]]:
    candidates = []
    seen_in_selection: set[str] = set()
    for uploaded in uploaded_files:
        content = uploaded.getvalue()
        fingerprint = _fingerprint(content)
        if fingerprint in seen_in_selection:
            continue
        seen_in_selection.add(fingerprint)
        if allow_duplicates or fingerprint not in uploaded_fingerprints:
            candidates.append((uploaded, content, fingerprint))
    return candidates


def _initialize_state(client: ApiClient) -> bool:
    if "session_id" in st.session_state:
        st.session_state.setdefault("message_sequence", len(st.session_state.get("messages", [])))
        return True
    try:
        payload = client.request("POST", "/api/v1/sessions")
    except ApiRequestError as exc:
        st.error(str(exc))
        st.info("Backend çalıştıktan sonra **Tekrar dene** düğmesini kullanın.")
        if st.button("Tekrar dene", key="retry_session"):
            st.rerun()
        return False

    st.session_state.session_id = payload["session_id"]
    st.session_state.documents = []
    st.session_state.messages = []
    st.session_state.uploaded_fingerprints = set()
    st.session_state.message_sequence = 0
    st.session_state.api_status = None
    return True


def _check_api(client: ApiClient, *, force: bool = False) -> bool:
    if not force and st.session_state.get("api_status") is not None:
        return st.session_state.api_status
    try:
        client.request("GET", "/health")
    except ApiRequestError:
        st.session_state.api_status = False
    else:
        st.session_state.api_status = True
    return st.session_state.api_status


def _reset_session(client: ApiClient) -> bool:
    """Delete the backend session before clearing any corresponding UI state."""
    session_id = st.session_state.get("session_id")
    if session_id:
        try:
            client.request("DELETE", f"/api/v1/sessions/{session_id}")
        except ApiRequestError as exc:
            st.error(f"Oturum temizlenemedi: {exc}")
            return False

    for key in (
        "session_id",
        "documents",
        "messages",
        "uploaded_fingerprints",
        "message_sequence",
        "api_status",
        "upload_statuses",
    ):
        st.session_state.pop(key, None)
    return True


def _render_styles() -> None:
    """Apply static, local styles only; no user content enters this HTML."""
    st.markdown(
        """
        <style>
          :root {
            --app-bg: #f8f7f4;
            --sidebar-bg: #f1f0ec;
            --ink: #182230;
            --heading: #152238;
            --muted: #667085;
            --accent: #4f46e5;
            --user-surface: #eef0fa;
            --line: #e5e4df;
          }
          .stApp {
            background: var(--app-bg);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
          }
          .block-container { max-width: 1120px; padding-top: 1.75rem; padding-bottom: 3rem; }
          section[data-testid="stSidebar"] { background: var(--sidebar-bg); }
          h1, h2, h3 { color: var(--heading); letter-spacing: -0.025em; }
          h1 {
            font-size: clamp(1.65rem, 2.6vw, 2.05rem);
            line-height: 1.22;
            overflow-wrap: anywhere;
          }
          h2, h3 { line-height: 1.3; }
          p, .stMarkdown { line-height: 1.65; }
          [data-testid="stCaptionContainer"] p { color: var(--muted); }
          .stButton button, .stFormSubmitButton button {
            min-height: 2.65rem;
            border-radius: 10px;
            font-weight: 600;
            transition: border-color 120ms ease, background-color 120ms ease;
          }
          .stButton button:focus-visible, .stFormSubmitButton button:focus-visible,
          textarea:focus-visible, input:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
          }
          div[data-testid="stChatMessage"] {
            padding: 0.85rem 1rem;
            margin: 0.65rem 0;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: #fff;
          }
          div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--user-surface);
            border-color: #dce0f3;
          }
          div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            max-width: 78ch;
            line-height: 1.7;
            overflow-wrap: anywhere;
          }
          .stExpander { border-color: var(--line); border-radius: 12px; }
          div[data-testid="stChatInput"] textarea { border-radius: 12px; }
          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
              scroll-behavior: auto !important;
              transition: none !important;
            }
          }
          @media (max-width: 700px) {
            .block-container { padding: 1.1rem 0.9rem 1.75rem; }
            div[data-testid="stChatMessage"] { padding: 0.7rem 0.75rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(client: ApiClient) -> None:
    settings = Settings()
    with st.sidebar:
        st.markdown("### Oturum")
        connected = _check_api(client)
        if connected:
            st.success("API bağlantısı etkin", icon="🟢")
            st.caption("Bu durum yalnızca FastAPI'nin erişilebilir olduğunu gösterir.")
        else:
            st.error("API bağlantısı yok", icon="🔴")
            st.caption("Backend'i `127.0.0.1:8000` adresinde başlatıp yeniden deneyin.")
        if st.button("API bağlantısını denetle", use_container_width=True):
            _check_api(client, force=True)
            st.rerun()

        documents = st.session_state.get("documents", [])
        st.caption(
            f"{len(documents)} / {settings.max_session_files} belge indekslendi. "
            "Dosya ayrıntıları ana alanda."
        )

        st.divider()
        st.caption(
            f"PDF başına en fazla {settings.max_upload_size_mb} MB. "
            "Belgeler yerel API oturumunun belleğinde tutulur."
        )
        st.caption(
            "Soru geçmişi yalnızca bu ekranda gösterilir; önceki sorular yeni yanıtlara "
            "bağlam olarak eklenmez."
        )
        if st.button("Oturumu temizle", type="secondary", use_container_width=True):
            if _reset_session(client):
                st.rerun()


def _render_documents(client: ApiClient) -> None:
    settings = Settings()
    st.subheader("Belgeler")
    st.caption(
        f"PDF sürükleyip bırakın veya seçin. PDF başına sınır {settings.max_upload_size_mb} MB; "
        f"oturum başına en fazla {settings.max_session_files} dosya."
    )
    uploaded_files = st.file_uploader(
        "PDF dosyaları",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=settings.max_upload_size_mb,
        label_visibility="collapsed",
        key="pdf_uploader",
        help="Metin içeren PDF'ler desteklenir. Taranmış belgeler için OCR henüz yoktur.",
    )
    allow_duplicates = st.checkbox(
        "Aynı PDF'yi bu oturuma yeniden indekslememe izin ver",
        help="İçerik parmak izi yalnızca Streamlit oturumunda tutulur; PDF metni kaydedilmez.",
    )
    candidates = _new_uploads(
        uploaded_files or [],
        st.session_state.uploaded_fingerprints,
        allow_duplicates=allow_duplicates,
    )
    if uploaded_files and not candidates:
        st.info("Seçilen PDF'ler bu oturumda zaten indekslendi.")

    if candidates and st.button(
        f"{len(candidates)} PDF'yi yükle ve indeksle",
        type="primary",
        key="upload_pdfs",
        disabled=not st.session_state.get("api_status", False),
    ):
        succeeded = 0
        for uploaded, content, fingerprint in candidates:
            with st.spinner(f"{uploaded.name} yerel olarak işlenip indeksleniyor…"):
                try:
                    payload = client.request(
                        "POST",
                        f"/api/v1/sessions/{st.session_state.session_id}/documents",
                        files={"file": (uploaded.name, content, "application/pdf")},
                    )
                except ApiRequestError as exc:
                    st.error(f"{uploaded.name}: {exc}")
                    st.caption(
                        "Teknik ayrıntı gerekiyorsa backend terminalindeki güvenli logları "
                        "inceleyin."
                    )
                    break
            st.session_state.documents = payload["documents"]
            st.session_state.uploaded_fingerprints.add(fingerprint)
            succeeded += 1
        if succeeded:
            st.success(f"{succeeded} PDF başarıyla indekslendi.")
            st.rerun()

    if st.session_state.get("documents"):
        with st.expander("İndekslenmiş belgeler", expanded=True):
            for document in st.session_state.documents:
                st.text(
                    f"{document['filename']} · {document['page_count']} sayfa · "
                    f"{document['chunk_count']} parça · İndekslendi"
                )


def _render_sources(sources: Sequence[dict[str, Any]]) -> None:
    if not sources:
        st.caption("Bu yanıt için doğrulanmış belge kaynağı bulunamadı.")
        return
    with st.expander(f"Kaynaklar · {len(sources)}", expanded=False):
        for source in sources:
            st.caption(
                f"{source['source_id']} · Sayfa {source['page_number']} · "
                f"Parça {source['chunk_index']}"
            )
            st.text(source["filename"])
            st.caption(
                f"Benzerlik {source['similarity']:.3f}. Bu değer sıralama ölçüsüdür; "
                "doğruluk yüzdesi değildir."
            )


def _append_message(message: dict[str, Any]) -> None:
    message["id"] = st.session_state.message_sequence
    st.session_state.message_sequence += 1
    st.session_state.messages.append(message)
    st.session_state.messages = st.session_state.messages[-50:]


def _render_chat(client: ApiClient) -> None:
    st.subheader("Belgeye sorun")
    if not st.session_state.documents:
        st.info(
            "Sohbete başlamak için önce **Belgeler** bölümünden metin içeren bir PDF yükleyin. "
            "Soruların yanıtı belgenin içeriğine bağlıdır."
        )
        st.caption(
            "Örnek soru fikirleri: garanti süresi nedir? İade için hangi adımlar izlenir? "
            "Belge hangi teknik destek kanalını öneriyor?"
        )
        return

    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            if message.get("error"):
                st.error(message["error"])
                if st.button("Bu soruyu tekrar dene", key=f"retry_{message['id']}"):
                    st.session_state.pending_question = message["question"]
                    st.rerun()
            else:
                st.markdown(message["answer"] or "Yanıt oluşturulamadı.")
                _render_sources(message["sources"])

    pending_question = st.session_state.pop("pending_question", None)
    question = pending_question or st.chat_input(
        "Belgeleriniz hakkında Türkçe bir soru sorun…", max_chars=2000
    )
    if not question:
        return
    if not question.strip():
        st.warning("Lütfen boş olmayan bir soru yazın.")
        return
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Belgeler aranıyor, yerel model yanıt hazırlıyor…"):
            try:
                payload = client.request(
                    "POST",
                    f"/api/v1/sessions/{st.session_state.session_id}/questions",
                    json={"question": question},
                )
            except ApiRequestError as exc:
                st.error(str(exc))
                st.caption(
                    "Yanıt oluşturulamadı. Sorun sürerse backend ve Ollama durumunu kontrol edin."
                )
                message = {
                    "question": question,
                    "answer": "",
                    "sources": [],
                    "error": str(exc),
                }
                _append_message(message)
            else:
                message = {
                    "question": question,
                    "answer": payload["answer"],
                    "sources": payload["sources"],
                }
                _append_message(message)
                st.markdown(message["answer"] or "Yanıt oluşturulamadı.")
                _render_sources(message["sources"])


def main() -> None:
    st.set_page_config(
        page_title="Türkçe Doküman AI Asistanı",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _render_styles()
    client = _api_client()
    st.markdown("# Türkçe Doküman AI Asistanı")
    st.markdown(
        "PDF belgelerinize Türkçe sorular sorun. Yanıtları kaynaklarıyla birlikte inceleyin."
    )
    st.caption(
        "Ücretsiz ve yerel çalışma · PDF metni ve indeks bu oturum boyunca bellekte tutulur."
    )

    if not _initialize_state(client):
        return
    _render_sidebar(client)
    _render_documents(client)
    st.divider()
    _render_chat(client)


if __name__ == "__main__":
    main()
