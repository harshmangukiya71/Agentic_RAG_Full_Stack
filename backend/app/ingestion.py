"""
ingestion.py - OCR/native extraction and structure-aware chunking.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

import numpy as np

from app.config import get_settings
from app.embeddings import EmbeddingModel
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


def _split_long_sentences(sentences: list[str], chunk_size_tokens: int) -> list[str]:
    """Keep pathological long spans from creating oversized chunks."""
    max_chars = max(chunk_size_tokens * _CHARS_PER_TOKEN, 1)
    split: list[str] = []
    for sentence in sentences:
        if _approx_tokens(sentence) <= chunk_size_tokens:
            split.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        current_chars = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_chars + extra > max_chars:
                split.append(" ".join(current))
                current = [word]
                current_chars = len(word)
            else:
                current.append(word)
                current_chars += extra
        if current:
            split.append(" ".join(current))
    return split


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


def _make_chunk(
    sentences: list[str],
    document: str,
    page: int,
    chunk_index: int,
    section_title: str | None,
    ocr_confidence: float | None,
    extraction_method: str,
) -> Chunk | None:
    text = " ".join(sentences).strip()
    if not text:
        return None
    return Chunk(
        document=document,
        page=page,
        chunk_index=chunk_index,
        text=text,
        token_count=_approx_tokens(text),
        section_title=section_title,
        ocr_confidence=ocr_confidence,
        extraction_method=extraction_method,
    )


def _trailing_overlap(sentences: list[str], chunk_overlap_tokens: int) -> list[str]:
    overlap: list[str] = []
    tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = _approx_tokens(sentence)
        if tokens + sentence_tokens > chunk_overlap_tokens:
            break
        overlap.insert(0, sentence)
        tokens += sentence_tokens
    return overlap


def _merge_tiny_groups(
    groups: list[list[str]],
    min_tokens: int,
    chunk_size_tokens: int,
) -> list[list[str]]:
    merged: list[list[str]] = []
    for group in groups:
        group_tokens = sum(_approx_tokens(s) for s in group)
        if (
            merged
            and group_tokens < min_tokens
            and sum(_approx_tokens(s) for s in merged[-1]) + group_tokens
            <= chunk_size_tokens
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)

    if len(merged) > 1 and sum(_approx_tokens(s) for s in merged[-1]) < min_tokens:
        last = merged.pop()
        if sum(_approx_tokens(s) for s in merged[-1] + last) <= chunk_size_tokens:
            merged[-1].extend(last)
        else:
            merged.append(last)
    return merged


def _build_semantic_chunks(
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
    settings = get_settings()
    sentences = _split_long_sentences(sentences, chunk_size_tokens)
    if len(sentences) <= 1:
        return _build_chunks(
            sentences, document, page, chunk_size_tokens, chunk_overlap_tokens,
            start_chunk_index, section_title, ocr_confidence, extraction_method,
        )

    embeddings = EmbeddingModel.get().embed_documents(sentences)
    if embeddings.size == 0 or len(embeddings) != len(sentences):
        raise ValueError("sentence embedding count mismatch")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    current_tokens = _approx_tokens(sentences[0])
    rolling_similarity = float(adjacent[0]) if len(adjacent) else 1.0

    for idx in range(1, len(sentences)):
        sentence = sentences[idx]
        sentence_tokens = _approx_tokens(sentence)
        previous_similarity = float(adjacent[idx - 1]) if idx - 1 < len(adjacent) else rolling_similarity
        similarity_drop = rolling_similarity - previous_similarity
        should_split = (
            current_tokens + sentence_tokens > chunk_size_tokens
            or (
                current_tokens >= settings.semantic_chunk_min_tokens
                and similarity_drop >= settings.semantic_chunk_similarity_drop
            )
        )

        if should_split and current:
            groups.append(current)
            current = _trailing_overlap(current, chunk_overlap_tokens)
            current_tokens = sum(_approx_tokens(s) for s in current)

        current.append(sentence)
        current_tokens += sentence_tokens
        rolling_similarity = (rolling_similarity * 0.75) + (previous_similarity * 0.25)

    if current:
        groups.append(current)

    groups = _merge_tiny_groups(
        groups,
        min_tokens=settings.semantic_chunk_min_tokens,
        chunk_size_tokens=chunk_size_tokens,
    )

    chunks: list[Chunk] = []
    chunk_idx = start_chunk_index
    for group in groups:
        chunk = _make_chunk(
            group, document, page, chunk_idx, section_title,
            ocr_confidence, extraction_method,
        )
        if chunk:
            chunks.append(chunk)
            chunk_idx += 1
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
    settings = get_settings()
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

        if settings.semantic_chunking_enabled:
            try:
                page_chunks = _build_semantic_chunks(
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
            except Exception as exc:
                logger.warning(
                    "Semantic chunking failed on %s page %d; using fixed chunks: %s",
                    document_name, ocr_page.page_number, exc,
                )
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
        else:
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
