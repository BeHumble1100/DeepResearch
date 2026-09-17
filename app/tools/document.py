"""Async document opening, parsing, and local parsed-text storage."""

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Protocol

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.research.schemas import DocumentRef


class DocumentOpener(Protocol):
    async def open(self, *, url: str) -> DocumentRef: ...


class DocumentOpenError(RuntimeError):
    """Raised when an opened document cannot be fetched or parsed."""


class LocalDocumentStore:
    """Stores parsed text outside ResearchState for later document-local retrieval."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write_text(self, *, document_id: str, content: str) -> str:
        directory = self._root / document_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "content.txt"
        temporary = directory / "content.txt.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
        return str(destination)

    def read_text(self, *, local_path: str) -> str:
        return Path(local_path).read_text(encoding="utf-8")


class HttpDocumentOpener:
    """Fetches HTML/PDF documents and persists complete parsed text locally."""

    _HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
    _PDF_CONTENT_TYPE = "application/pdf"
    _REQUEST_HEADERS = {
        "User-Agent": "DeepResearch/1.0 (evidence-driven research agent)",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
    }

    def __init__(
        self,
        settings: Settings,
        *,
        store: LocalDocumentStore | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or LocalDocumentStore(settings.document_store_dir)
        self._transport = transport

    async def open(self, *, url: str) -> DocumentRef:
        timeout = httpx.Timeout(self._settings.document_timeout_seconds)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                transport=self._transport,
                headers=self._REQUEST_HEADERS,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.TimeoutException as error:
            raise DocumentOpenError(f"Document request timed out: {url}") from error
        except httpx.HTTPStatusError as error:
            raise DocumentOpenError(
                f"Document request returned HTTP {error.response.status_code}: {url}"
            ) from error
        except httpx.HTTPError as error:
            raise DocumentOpenError(f"Document request failed: {url}") from error

        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        is_pdf = content_type == self._PDF_CONTENT_TYPE or response.content.startswith(b"%PDF-")
        if is_pdf:
            title, content = await asyncio.to_thread(_extract_pdf, response.content)
            normalized_content_type = self._PDF_CONTENT_TYPE
        elif content_type in self._HTML_CONTENT_TYPES:
            title, content = await asyncio.to_thread(_extract_html, response.text)
            normalized_content_type = "text/html"
        else:
            raise DocumentOpenError(f"Unsupported document content type: {content_type or 'missing'}")

        if not content.strip():
            raise DocumentOpenError(f"Document contains no extractable text: {response.url}")

        final_url = str(response.url)
        document_id = hashlib.sha256(final_url.encode("utf-8")).hexdigest()
        local_path = await asyncio.to_thread(
            self._store.write_text,
            document_id=document_id,
            content=content,
        )
        return DocumentRef(
            id=document_id,
            url=final_url,
            title=title,
            content_type=normalized_content_type,
            local_path=local_path,
        )


def _extract_html(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    for element in soup(["script", "style", "noscript", "template", "nav", "header", "footer", "aside"]):
        element.decompose()
    content = trafilatura.extract(str(soup), include_links=False, include_images=False)
    if content:
        return title, content.strip()

    text = soup.get_text("\n", strip=True)
    return title, text


def _extract_pdf(content: bytes) -> tuple[str | None, str]:
    try:
        reader = PdfReader(BytesIO(content))
        metadata_title = reader.metadata.title if reader.metadata else None
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except PdfReadError as error:
        raise DocumentOpenError("PDF could not be parsed.") from error
    return metadata_title, text
