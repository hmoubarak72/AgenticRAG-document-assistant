"""Split extracted sections into overlapping chunks, preserving page numbers.

Markdown-style headings start a new chunk, so each document section is retrieved and
cited on its own instead of being merged with its neighbors.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING = re.compile(r"(?m)^(?=#{1,4} )")


def _heading_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in _HEADING.split(text)]
    return [b for b in blocks if b]


def chunk_sections(sections: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for section in sections:
        for block in _heading_blocks(section["text"]):
            for piece in splitter.split_text(block):
                piece = piece.strip()
                if len(piece) >= 20:
                    chunks.append({"text": piece, "page": section.get("page")})
    return chunks
