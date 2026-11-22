"""Text extraction for supported formats.

Each loader returns a list of ``{"text": str, "page": int | None}`` sections. PDFs and
slide decks keep page/slide numbers so answers can cite them.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from pathlib import Path

Section = dict


def load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append({"text": text, "page": i})
    return out


def load_docx(data: bytes) -> list[Section]:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [{"text": "\n".join(parts), "page": None}] if parts else []


def load_xlsx(data: bytes) -> list[Section]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(vals):
                rows.append(" | ".join(vals))
        if rows:
            out.append({"text": f"Sheet: {ws.title}\n" + "\n".join(rows), "page": None})
    return out


def load_pptx(data: bytes) -> list[Section]:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = [sh.text_frame.text for sh in slide.shapes
                 if getattr(sh, "has_text_frame", False) and sh.text_frame.text.strip()]
        if texts:
            out.append({"text": "\n".join(texts), "page": i})
    return out


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def load_text(data: bytes) -> list[Section]:
    text = _decode(data).strip()
    return [{"text": text, "page": None}] if text else []


def load_csv(data: bytes) -> list[Section]:
    reader = csv.reader(io.StringIO(_decode(data)))
    rows = [" | ".join(r) for r in reader if any(c.strip() for c in r)]
    return [{"text": "\n".join(rows), "page": None}] if rows else []


def load_json(data: bytes) -> list[Section]:
    try:
        obj = json.loads(_decode(data))
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        text = _decode(data)
    return [{"text": text, "page": None}] if text.strip() else []


def load_html(data: bytes) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(data), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return [{"text": text, "page": None}] if text else []


LOADERS: dict[str, Callable[[bytes], list[Section]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_xlsx,
    ".pptx": load_pptx,
    ".csv": load_csv,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".json": load_json,
    ".html": load_html,
    ".htm": load_html,
}

SUPPORTED_EXTENSIONS = sorted(LOADERS)


class UnsupportedFileType(ValueError):
    pass


def extract_sections(filename: str, data: bytes) -> list[Section]:
    ext = Path(filename).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    return loader(data)
