"""
ingestion.py - OCR/native extraction and structure-aware chunking.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

from app.config import get_settings
from app.models import Chunk, OCRPage
from app.ocr import OCRProcessor

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019])"
    r"|(?<=\n)\s*\n+",
    re.UNICODE,
)
_SECTION_RE = re.compile(
    r"^\s*(?:section|article|clause|schedule|exhibit|annexure)?\s*"
    r"(?:\d+(?:\.\d+)*|[A-Z])[\).:-]?\s+[A-Z][A-Za-z0-9 ,&/()-]{3,120}$",
    re.IGNORECASE,
)
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def clean_ocr_text(text: str) -> str:
    """Clean OCR/native text while preserving paragraphs and clause boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z0-9(])", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _section_title(text: str, previous: str | None = None) -> str | None:
    for line in text.splitlines():
        candidate = line.strip()
        if 4 <= len(candidate) <= 140 and _SECTION_RE.match(candidate):
            return candidate
    return previous


def _build_chunks(
    sentences: list[str],
    document: str,
    page: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    start_chunk_index: int,
    section_title: str | None,
    ocr_confidence: float | None,
    extraction_method: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_idx = start_chunk_index

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = _approx_tokens(sent)

        if current_tokens + sent_tokens <= chunk_size_tokens:
            current_sentences.append(sent)
            current_tokens += sent_tokens
            i += 1
            continue

        if current_sentences:
            text = " ".join(current_sentences).strip()
            if text:
                chunks.append(Chunk(
                    document=document,
                    page=page,
                    chunk_index=chunk_idx,
                    text=text,
                    token_count=current_tokens,
                    section_title=section_title,
                    ocr_confidence=ocr_confidence,
                    extraction_method=extraction_method,
                ))
                chunk_idx += 1

            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                t = _approx_tokens(s)
                if overlap_tokens + t <= chunk_overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += t
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens
        else:
            chunks.append(Chunk(
                document=document,
                page=page,
                chunk_index=chunk_idx,
                text=sent.strip(),
                token_count=sent_tokens,
                section_title=section_title,
                ocr_confidence=ocr_confidence,
                extraction_method=extraction_method,
            ))
            chunk_idx += 1
            i += 1

    if current_sentences:
        text = " ".join(current_sentences).strip()
        if text:
            chunks.append(Chunk(
                document=document,
                page=page,
                chunk_index=chunk_idx,
                text=text,
                token_count=current_tokens,
                section_title=section_title,
                ocr_confidence=ocr_confidence,
                extraction_method=extraction_method,
            ))

    return chunks


def extract_pages(path: Path | str) -> list[OCRPage]:
    settings = get_settings()
    processor = OCRProcessor(
        min_native_chars_per_page=settings.min_native_chars_per_page,
        render_dpi=settings.ocr_render_dpi,
    )
    return processor.extract_document(path)


def parse_pdf(
    pdf_path: Path | str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[Chunk]:
    """Parse a PDF or image file and return chunks with provenance metadata."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

    document_name = pdf_path.name
    all_chunks: list[Chunk] = []
    global_chunk_idx = 0
    last_section: str | None = None

    logger.info("Parsing document with OCR fallback: %s", document_name)
    pages = extract_pages(pdf_path)
    for ocr_page in pages:
        raw_text = clean_ocr_text(ocr_page.raw_text)
        if not raw_text:
            logger.debug("Page %d is empty - skipping", ocr_page.page_number)
            continue

        last_section = _section_title(raw_text, last_section)
        sentences = _split_into_sentences(raw_text)
        if not sentences:
            continue

        page_chunks = _build_chunks(
            sentences=sentences,
            document=document_name,
            page=ocr_page.page_number,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            start_chunk_index=global_chunk_idx,
            section_title=last_section,
            ocr_confidence=ocr_page.ocr_confidence,
            extraction_method=ocr_page.extraction_method,
        )
        all_chunks.extend(page_chunks)
        global_chunk_idx += len(page_chunks)

    logger.info("Parsed '%s' -> %d pages -> %d chunks", document_name, len(pages), len(all_chunks))
    return all_chunks


def iter_pdfs(directory: Path | str) -> Generator[Path, None, None]:
    """Yield supported document paths in a directory (non-recursive)."""
    directory = Path(directory)
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            yield f