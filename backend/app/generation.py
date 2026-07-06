"""
generation.py — LLM generation via Gemini with hallucination mitigation.

Hallucination mitigation (multi-layered):
  1. Context-only system prompt: LLM is explicitly forbidden from using external knowledge.
  2. LLM Fallback: If no relevant chunks are retrieved and question seems general,
     fall back to a direct LLM answer (clearly labelled as such).
  3. Confidence scoring: Computed as weighted combination of:
       • Top chunk re-ranking score (semantic relevance)
       • Token overlap between answer and retrieved context (faithfulness proxy)
     If computed confidence < 0.35 we append a caveat to the answer.
  4. Source citation enforcement: sources are pulled from the retrieved chunks
     deterministically, not from LLM hallucination.

Works with: any uploaded PDF — legal, resume, story, technical, medical, etc.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from openai import OpenAI

from app.config import get_settings
from app.models import QueryResponse, ReasoningOutput, RetrievedChunk, SourceReference

logger = logging.getLogger(__name__)


def _is_llm_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "quota",
            "resource_exhausted",
            "too many requests",
        )
    )


def _format_number(value) -> str:
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    return [
        SourceReference(
            document=chunk.document,
            page=chunk.page,
            chunk=chunk.chunk,
            chunk_index=chunk.chunk_index,
            entities=chunk.entity_matches,
        )
        for chunk in chunks
    ]


def _deterministic_reasoning_answer(
    reasoning_output: ReasoningOutput | None,
    context_chunks: list[RetrievedChunk],
) -> QueryResponse | None:
    if not reasoning_output:
        return None

    if reasoning_output.rankings:
        lines = ["Ranked results from extracted document evidence:"]
        for item in reasoning_output.rankings:
            entity = item.get("entity") or item.get("text", "Item")
            metric = item.get("metric") or "value"
            value = item.get("value", item.get("year", ""))
            suffix = ""
            if item.get("chunk_document") and item.get("chunk_page"):
                suffix = f" ({item['chunk_document']}, page {item['chunk_page']})"
            lines.append(
                f"{item.get('rank', len(lines))}. {entity}: "
                f"{_format_number(value)} {metric}{suffix}"
            )
        return QueryResponse(
            answer="\n".join(lines),
            sources=_build_sources(context_chunks),
            confidence=0.75 if reasoning_output.evidence_sufficient else 0.45,
        )

    if reasoning_output.calculations:
        lines = ["Calculated results from extracted document evidence:"]
        for calc in reasoning_output.calculations:
            if calc.get("operation") == "numeric_filter":
                records = calc.get("records", [])
                if not records:
                    lines.append("No records matched the numeric filter.")
                else:
                    metric = calc.get("metric", "value")
                    for record in records:
                        lines.append(
                            f"- {record['entity']}: "
                            f"{_format_number(record['value'])} {metric}"
                        )
            elif calc.get("operation") == "summary_stats":
                metric = calc.get("metric", "metric")
                lines.append(
                    f"{metric}: sum={_format_number(calc.get('sum'))}, "
                    f"average={_format_number(calc.get('average'))}, "
                    f"min={_format_number(calc.get('min'))}, "
                    f"max={_format_number(calc.get('max'))}, "
                    f"count={_format_number(calc.get('count'))}"
                )
            elif "result" in calc:
                lines.append(
                    f"{calc.get('metric', calc.get('operation', 'result'))}: "
                    f"{_format_number(calc['result'])}"
                )
        return QueryResponse(
            answer="\n".join(lines),
            sources=_build_sources(context_chunks),
            confidence=0.7 if reasoning_output.evidence_sufficient else 0.4,
        )

    return None

# ── System prompt (RAG mode) ─────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful, precise document assistant. Your job is to answer questions
based strictly on the provided document excerpts. Follow these rules:

RULES:
1. Answer ONLY from the provided context. Do NOT use external knowledge or make assumptions.
2. If the context does not contain the answer, say exactly: "The information requested was not found in the provided document excerpts."
3. Be specific and accurate: include exact names, numbers, dates, and details when present.
4. Match your tone to the document type — factual for reports, friendly for stories, precise for contracts.
5. Do NOT fabricate, infer, or extrapolate any information not explicitly present in the context.
"""

# ── System prompt (LLM fallback mode) ────────────────────────────────────────
_FALLBACK_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Answer the user's question
clearly and accurately using your general knowledge. Be concise, factual, and friendly.
If the question is ambiguous, state your assumptions briefly."""

_CONTEXT_TEMPLATE = """{structured_context}DOCUMENT EXCERPTS:
{context}

{history_block}

USER QUESTION: {question}

Answer based solely on the excerpts above. Use the conversation history only to
understand references in the question; do not treat it as document evidence."""

_HISTORY_TEMPLATE = """RECENT CONVERSATION HISTORY:
{history}"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i}] Document: {chunk.document} | Page: {chunk.page}\n"
            f"{chunk.chunk}"
        )
    return "\n\n---\n\n".join(parts)


def _build_structured_context(reasoning_output: ReasoningOutput | None) -> str:
    if not reasoning_output:
        return ""

    if not (reasoning_output.calculations or reasoning_output.rankings):
        return ""

    return (
        "REASONING RESULTS (pre-computed from document evidence):\n"
        f"{reasoning_output.summary}\n\n"
        "Use these pre-computed results as primary evidence. "
        "Cross-check against the excerpts below.\n\n"
    )


def _token_overlap_score(answer: str, context: str) -> float:
    """
    Compute token-level recall: fraction of unique answer tokens found in context.
    Serves as a faithfulness proxy — high overlap means answer is grounded in context.
    """
    def tokenize(text: str) -> set[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = set(text.split())
        # Remove stopwords (minimal list)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                     "to", "and", "or", "that", "this", "it", "for", "on", "with",
                     "as", "at", "by", "from", "be", "has", "have", "had", "not"}
        return tokens - stopwords

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def _compute_confidence(
    top_relevance_score: float,
    token_overlap: float,
    settings,
) -> float:
    """
    Confidence = weighted combination of retrieval quality + answer faithfulness.

      • top_relevance_score (0–1): how semantically relevant the top chunk is.
      • token_overlap (0–1): how much of the answer appears verbatim in context.

    The combined score is clipped to [0, 1].
    """
    raw = (0.6 * top_relevance_score) + (0.4 * token_overlap)
    scaled = min(1.0, raw * settings.confidence_scale_factor)
    return round(scaled, 3)


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    top_k_context: int | None = None,
    conversation_history: str = "",
    reasoning_output: ReasoningOutput | None = None,
) -> QueryResponse:
    """
    Generate an answer grounded in retrieved_chunks using Gemini.

    Steps:
      1. Check if top chunk relevance >= threshold; refuse if not.
      2. Build context from top-k retrieved chunks.
      3. Call Gemini (OpenAI-compatible API).
      4. Compute confidence score.
      5. Return structured QueryResponse.
    """
    settings = get_settings()
    k = top_k_context or settings.final_context_k
    context_chunks = retrieved_chunks[:k]

    # ── Step 1: Refusal guard (only if zero chunks came through) ─────────────
    # Note: the empty-store case is handled upstream in pipeline.py before
    # generate_answer() is called. If we reach here, we always have some chunks.
    # Low relevance scores still get sent to the LLM — it will correctly say
    # "not found" if the context doesn't contain the answer.
    top_score = context_chunks[0].relevance_score if context_chunks else 0.0
    logger.info(
        "Generating answer: %d chunks, top relevance=%.4f",
        len(context_chunks), top_score,
    )
    if not context_chunks:
        # Should not normally reach here (pipeline guards against this)
        return QueryResponse(
            answer="No relevant content was retrieved. Please upload a document first.",
            sources=[],
            confidence=0.0,
        )


    # ── Step 2: Build context ─────────────────────────────────────────────
    context_text = _build_context(context_chunks)
    structured_context = _build_structured_context(reasoning_output)
    history_block = _HISTORY_TEMPLATE.format(history=conversation_history) if conversation_history else ""
    user_content = _CONTEXT_TEMPLATE.format(
        structured_context=structured_context,
        context=context_text,
        history_block=history_block,
        question=question,
    )

    # ── Step 3: Call Gemini ───────────────────────────────────────────────
    client = OpenAI(
        base_url=settings.gemini_base_url,
        api_key=settings.gemini_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    try:
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=settings.gemini_max_tokens,
            temperature=settings.gemini_temperature,
            stream=False,
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("Gemini API call failed: %s", exc)
        deterministic = _deterministic_reasoning_answer(
            reasoning_output, context_chunks
        )
        if deterministic:
            if _is_llm_capacity_error(exc):
                deterministic.answer += (
                    "\n\nNote: the LLM provider is currently rate-limited, "
                    "so this answer was generated from the local reasoning "
                    "extractor instead of the final LLM step."
                )
            return deterministic
        if _is_llm_capacity_error(exc):
            return QueryResponse(
                answer=(
                    "The LLM provider is currently rate-limited or out of quota, "
                    "so I could not generate the final answer. Please retry after "
                    "the quota window resets."
                ),
                sources=_build_sources(context_chunks),
                confidence=0.0,
            )
        raise

    # ── Step 4: Confidence scoring ────────────────────────────────────────
    top_relevance = context_chunks[0].relevance_score
    overlap = _token_overlap_score(raw_answer, structured_context + context_text)
    confidence = _compute_confidence(top_relevance, overlap, settings)

    # Append caveat if confidence is low
    if confidence < 0.35 and "not found" not in raw_answer.lower():
        raw_answer += (
            "\n\n⚠️ *Low confidence: the retrieved context may not fully cover this question. "
            "Please verify against the original document.*"
        )

    # ── Step 5: Build structured response ────────────────────────────────
    sources = _build_sources(context_chunks)

    return QueryResponse(
        answer=raw_answer.strip(),
        sources=sources,
        confidence=confidence,
    )


def generate_llm_fallback(question: str, conversation_history: str = "") -> QueryResponse:
    """
    Answer a question directly from LLM general knowledge (no document context).

    Used when:
      - No documents are ingested, OR
      - Retrieved chunks have very low relevance AND question seems general.

    The response is clearly labelled so the user knows it's not grounded in their docs.
    """
    settings = get_settings()
    client = OpenAI(
        base_url=settings.gemini_base_url,
        api_key=settings.gemini_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    logger.info("LLM fallback: answering '%s' from general knowledge", question[:80])

    try:
        user_content = question
        if conversation_history:
            user_content = (
                "RECENT CONVERSATION HISTORY:\n"
                f"{conversation_history}\n\n"
                f"USER QUESTION: {question}"
            )

        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[
                {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=settings.gemini_max_tokens,
            temperature=0.4,   # slightly warmer for general Q&A
            stream=False,
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("LLM fallback API call failed: %s", exc)
        raise

    # Prepend a clear indicator that this is NOT from the uploaded documents
    labelled = (
        "🌐 **General Knowledge Answer** *(No relevant content found in your documents — "
        "answering from general knowledge)*\n\n"
        + raw_answer.strip()
    )

    return QueryResponse(
        answer=labelled,
        sources=[],
        confidence=0.5,   # medium confidence — LLM-only, not grounded
    )
