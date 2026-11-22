import io
import json

import pytest

from backend.ingestion.chunker import chunk_sections
from backend.ingestion.loaders import UnsupportedFileType, extract_sections


def test_text_and_markdown():
    assert extract_sections("a.txt", b"hello world")[0]["text"] == "hello world"
    assert "Title" in extract_sections("a.md", b"# Title\nbody")[0]["text"]


def test_csv_and_json():
    assert "a | b" in extract_sections("a.csv", b"a,b\n1,2\n")[0]["text"]
    assert '"k": 1' in extract_sections("a.json", json.dumps({"k": 1}).encode())[0]["text"]


def test_html_strips_scripts():
    text = extract_sections("a.html", b"<html><script>evil()</script><p>Visible</p></html>")[0]["text"]
    assert "Visible" in text and "evil" not in text


def test_docx_roundtrip():
    import docx

    d = docx.Document()
    d.add_paragraph("Refund terms apply.")
    buf = io.BytesIO()
    d.save(buf)
    assert "Refund terms" in extract_sections("a.docx", buf.getvalue())[0]["text"]


def test_xlsx_roundtrip():
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.append(["Region", "Revenue"])
    wb.active.append(["EMEA", 42])
    buf = io.BytesIO()
    wb.save(buf)
    assert "EMEA | 42" in extract_sections("a.xlsx", buf.getvalue())[0]["text"]


def test_unsupported():
    with pytest.raises(UnsupportedFileType):
        extract_sections("a.exe", b"")


def test_chunker_keeps_pages():
    chunks = chunk_sections([{"text": "word " * 600, "page": 3}], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1 and all(c["page"] == 3 for c in chunks)


def test_chunker_starts_new_chunk_at_each_heading():
    text = "# Title\nAn introduction paragraph.\n\n## Section 1 — A\nFirst section text.\n\n## Section 2 — B\nSecond section text."
    chunks = chunk_sections([{"text": text, "page": None}], chunk_size=1000, chunk_overlap=0)
    assert [c["text"].splitlines()[0] for c in chunks] == ["# Title", "## Section 1 — A", "## Section 2 — B"]
