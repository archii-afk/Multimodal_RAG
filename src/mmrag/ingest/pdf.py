"""Page-bounded PDF text ingestion with exact provenance offsets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz

from mmrag.model import DEFAULT_CONFIDENCE, IngestBatch, Modality, NodeDraft, NodeKind, SourceDraft


def ingest_pdf(
    path: str | Path,
    *,
    source_ref: str,
    chunk_tokens: int = 650,
    overlap_tokens: int = 80,
) -> IngestBatch:
    """Extract PDF text into overlapping chunks that never cross pages."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("require chunk_tokens > overlap_tokens >= 0")

    source_sha = _sha256(pdf_path)
    nodes: list[NodeDraft] = [
        SourceDraft(
            ref=source_ref,
            kind=NodeKind.SOURCE,
            modality=Modality.DOCUMENT,
            content=pdf_path.stem,
            path=str(pdf_path),
            mime_type="application/pdf",
            sha256=source_sha,
            confidence=1.0,
            provenance={"extractor": "pymupdf"},
        )
    ]
    warnings: list[str] = []
    chunk_index = 0
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_text = page.get_text("text")
            page_chunks = _page_chunks(page_text, chunk_tokens, overlap_tokens)
            if not page_chunks:
                warnings.append(f"PDF page {page_number} contains no extractable text")
            for char_start, char_end, token_count in page_chunks:
                nodes.append(
                    NodeDraft(
                        ref=f"pdf:{chunk_index:04d}",
                        kind=NodeKind.PDF_CHUNK,
                        modality=Modality.DOCUMENT,
                        content=page_text[char_start:char_end],
                        source_ref=source_ref,
                        page=page_number,
                        confidence=DEFAULT_CONFIDENCE["pdf"],
                        provenance={
                            "extractor": "pymupdf",
                            "path": str(pdf_path),
                            "sha256": source_sha,
                            "page_char_start": char_start,
                            "page_char_end": char_end,
                            "token_count": token_count,
                            "chunk_tokens": chunk_tokens,
                            "overlap_tokens": overlap_tokens,
                        },
                    )
                )
                chunk_index += 1
    return IngestBatch(tuple(nodes), warnings=tuple(warnings))


def _page_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[tuple[int, int, int]]:
    tokens = list(re.finditer(r"\S+", text))
    chunks: list[tuple[int, int, int]] = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + chunk_tokens, len(tokens))
        chunks.append(
            (
                tokens[start_index].start(),
                tokens[end_index - 1].end(),
                end_index - start_index,
            )
        )
        if end_index == len(tokens):
            break
        start_index = end_index - overlap_tokens
    return chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
