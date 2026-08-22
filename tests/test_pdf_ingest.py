from __future__ import annotations

import fitz

from mmrag.ingest.pdf import ingest_pdf
from mmrag.model import Modality, NodeKind


def _make_pdf(path):
    document = fitz.open()
    for page_number in (1, 2):
        page = document.new_page()
        words = " ".join(f"page{page_number}-word{index}" for index in range(12))
        page.insert_textbox(fitz.Rect(40, 40, 550, 800), words, fontsize=10)
    document.save(path)
    document.close()


def test_ingest_pdf_keeps_chunks_within_pages_and_records_offsets(tmp_path):
    pdf_path = tmp_path / "architecture.pdf"
    _make_pdf(pdf_path)

    batch = ingest_pdf(
        pdf_path,
        source_ref="source:pdf",
        chunk_tokens=5,
        overlap_tokens=2,
    )

    assert batch.nodes[0].kind is NodeKind.SOURCE
    assert batch.nodes[0].modality is Modality.DOCUMENT
    chunks = [node for node in batch.nodes if node.kind is NodeKind.PDF_CHUNK]
    assert {node.page for node in chunks} == {1, 2}
    assert len(chunks) == 8
    with fitz.open(pdf_path) as document:
        page_text = {index + 1: page.get_text("text") for index, page in enumerate(document)}
    for chunk in chunks:
        start = chunk.provenance["page_char_start"]
        end = chunk.provenance["page_char_end"]
        assert chunk.content == page_text[chunk.page][start:end]
        assert f"page{chunk.page}-" in chunk.content
        assert f"page{3 - chunk.page}-" not in chunk.content
        assert chunk.provenance["token_count"] <= 5
