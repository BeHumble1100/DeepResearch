import asyncio
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.tools import document
from app.tools.document import DocumentOpenError, HttpDocumentOpener, LocalDocumentStore


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        searxng_base_url="http://localhost:8080",
        document_timeout_seconds=1,
        document_store_dir=tmp_path / "documents",
    )


def test_html_document_is_parsed_and_saved_for_later_retrieval(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == "DeepResearch/1.0 (evidence-driven research agent)"
        assert "text/html" in request.headers["accept"]
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
                <html><head><title>Example title</title></head>
                <body><nav>Navigation</nav><main><p>Useful document body.</p></main></body></html>
            """,
        )

    opener = HttpDocumentOpener(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    reference = asyncio.run(opener.open(url="https://example.com/article"))

    assert reference.content_type == "text/html"
    assert reference.title == "Example title"
    assert reference.local_path is not None
    saved_content = Path(reference.local_path).read_text(encoding="utf-8")
    assert "Useful document body." in saved_content
    assert "Navigation" not in saved_content


def test_pdf_document_is_parsed_and_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "PDF body text"

    class FakeMetadata:
        title = "PDF title"

    class FakePdfReader:
        def __init__(self, stream: object) -> None:
            self.metadata = FakeMetadata()
            self.pages = [FakePage()]

    monkeypatch.setattr(document, "PdfReader", FakePdfReader)
    opener = HttpDocumentOpener(
        make_settings(tmp_path),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=b"%PDF-mock-content",
            )
        ),
    )

    reference = asyncio.run(opener.open(url="https://example.com/document.pdf"))

    assert reference.content_type == "application/pdf"
    assert reference.title == "PDF title"
    assert reference.local_path is not None
    assert Path(reference.local_path).read_text(encoding="utf-8") == "PDF body text"


def test_document_opener_reports_timeout(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    opener = HttpDocumentOpener(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DocumentOpenError, match="timed out"):
        asyncio.run(opener.open(url="https://example.com/article"))


def test_document_opener_reports_http_status(tmp_path: Path) -> None:
    opener = HttpDocumentOpener(
        make_settings(tmp_path),
        transport=httpx.MockTransport(lambda request: httpx.Response(403)),
    )

    with pytest.raises(DocumentOpenError, match="HTTP 403"):
        asyncio.run(opener.open(url="https://example.com/article"))


def test_document_opener_rejects_unsupported_content_type(tmp_path: Path) -> None:
    opener = HttpDocumentOpener(
        make_settings(tmp_path),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "image/png"})
        ),
    )

    with pytest.raises(DocumentOpenError, match="Unsupported"):
        asyncio.run(opener.open(url="https://example.com/image.png"))


def test_local_document_store_reads_saved_text(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    path = store.write_text(document_id="document", content="Reusable document text")

    assert store.read_text(local_path=path) == "Reusable document text"
