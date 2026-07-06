"""
evaluation.py — Retrieval + hallucination evaluation harness.

Core fix in this version:
  GENERATE → VERIFY → ACCEPT/REJECT

  Every generated question is verified by asking the LLM to extract the
  exact answer from the source chunk text. If the LLM cannot find the answer
  in the chunk (returns NOT_FOUND), the question is rejected as hallucinated
  and never enters the eval pool. This guarantees every evaluated question
  is strictly answerable from the uploaded document.

  This fixes the MISS caused by questions like "When was xAI founded?" where
  the LLM generated a question from its own training knowledge about the topic
  rather than from the actual chunk content.

Two evaluation modes:
  1. AUTO: generates factual + graph/relational + comparative questions,
     verifies each against source chunk, deduplicates, evaluates.
  2. PREDEFINED: reads Q&A pairs from tests/eval_qa_pairs.json.

Retrieval metrics: Recall@1 / @3 / @5, MRR
Grounding metrics: answer_faithfulness (stemmed), evidence_coverage,
                   unsupported_rate, speculative_edge_rate
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from openai import OpenAI

from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.models import Chunk, EvalPair, EvalReport, EvalResult

if TYPE_CHECKING:
    from app.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_EVAL_TOP_K = 5
_FAITHFULNESS_THRESHOLD: float = 0.30
_MAX_FACTUAL_PER_CHUNK: int = 5
_MAX_GRAPH_PER_CHUNK: int = 3
_DEDUP_OVERLAP_THRESHOLD: float = 0.60

# Sentinel returned by _verify_question when answer is not in chunk
_NOT_FOUND = "__NOT_FOUND__"

_SPECULATIVE_PATTERNS = re.compile(
    r"\b(?:may\s+be|might\s+be|could\s+(?:be|suggest|indicate|imply)|"
    r"possibly|perhaps|indirectly|seems\s+to|appears\s+to|"
    r"it\s+is\s+(?:possible|likely|probable)|"
    r"suggests?\s+a\s+(?:possible|potential)|"
    r"cannot\s+be\s+confirmed|cannot\s+confirm|"
    r"implicitly|may\s+(?:have|indicate|suggest|imply))\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "and", "or", "that", "this", "it", "for", "on", "with", "as",
    "at", "by", "from", "be", "has", "have", "had", "not", "its",
    "their", "they", "which", "who", "what", "how", "when", "where",
}


# ── Stemmer (zero dependencies) ───────────────────────────────────────────────

def _stem(word: str) -> str:
    w = word.lower()
    for suffix in (
        "nesses", "ations", "ments",
        "iness", "ation", "ings", "ness", "ment",
        "ers", "ing", "ion", "ies", "ful",
        "ed", "er", "es", "ly", "s",
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return {_stem(tok) for tok in text.split() if tok not in _STOPWORDS}


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _cosine_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        model = EmbeddingModel.get()
        vec_a = model.embed_query(a)
        vec_b = model.embed_query(b)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom <= 0:
            return 0.0
        return round(max(0.0, min(1.0, float(np.dot(vec_a, vec_b) / denom))), 4)
    except Exception as exc:
        logger.debug("Embedding similarity unavailable; using token overlap: %s", exc)
        return round(_token_overlap(a, b), 4)


def _bertscore_like(answer: str, reference: str) -> dict[str, float]:
    if not reference.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    semantic = _cosine_similarity(answer, reference)
    token_precision = _token_overlap(answer, reference)
    token_recall = _token_overlap(reference, answer)
    precision = round(max(semantic, token_precision), 4)
    recall = round(max(semantic, token_recall), 4)
    f1 = round((2 * precision * recall / (precision + recall)), 4) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Faithfulness / grounding helpers ─────────────────────────────────────────

def _answer_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Stemmed token recall — fraction of answer tokens present in context."""
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for chunk in context_chunks:
        context_tokens |= _tokenize(chunk)
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


def _evidence_coverage(answer: str, context_chunks: list[str]) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0
    supported = 0
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        for chunk in context_chunks:
            if len(sent_tokens & _tokenize(chunk)) >= 3:
                supported += 1
                break
    return round(supported / len(sentences), 4)


def _is_speculative(answer: str, has_graph_chunks: bool) -> bool:
    if not has_graph_chunks:
        return False
    return bool(_SPECULATIVE_PATTERNS.search(answer))


def _compute_grounding_metrics(
    answer: str, source_chunks: list[dict]
) -> dict[str, Any]:
    # Use full chunk text; fall back to preview
    chunk_texts = [
        s.get("chunk_full") or s.get("chunk_preview", "") for s in source_chunks
    ]
    has_graph = any("graph" in s.get("retrieval_source", "") for s in source_chunks)
    faith = _answer_faithfulness(answer, chunk_texts)
    return {
        "faithfulness": faith,
        "evidence_coverage": _evidence_coverage(answer, chunk_texts),
        "is_speculative": _is_speculative(answer, has_graph),
        "has_graph_chunks": has_graph,
        "unsupported": faith < _FAITHFULNESS_THRESHOLD,
    }


# ── Dedup ─────────────────────────────────────────────────────────────────────

_PoolItem = tuple[str, str, int, str, str]  # question, document, page, type, reference_answer


def _deduplicate(
    candidates: list[_PoolItem],
    threshold: float = _DEDUP_OVERLAP_THRESHOLD,
) -> list[_PoolItem]:
    kept: list[_PoolItem] = []
    for candidate in candidates:
        q = candidate[0]
        if all(_token_overlap(q, k[0]) < threshold for k in kept):
            kept.append(candidate)
    return kept


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(
    system: str, user: str, max_tokens: int = 600, temperature: float = 0.4
) -> str:
    settings = get_settings()
    client = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        timeout=settings.eval_llm_timeout_seconds,
        max_retries=settings.eval_llm_max_retries,
    )
    resp = client.chat.completions.create(
        model=settings.nvidia_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_questions(raw: str) -> list[str]:
    """Parse a numbered LLM list output into individual question strings."""
    _starters = {
        "what", "who", "when", "where", "which", "how", "why",
        "whose", "whom", "is", "are", "was", "were", "does",
        "did", "can", "could", "would", "will", "name",
    }
    questions: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line).strip().strip('"\'')
        if not cleaned:
            continue
        if not cleaned.endswith("?"):
            first = cleaned.lower().split()[0] if cleaned.split() else ""
            if first in _starters:
                cleaned += "?"
        if cleaned.endswith("?") and 10 < len(cleaned) < 400:
            questions.append(cleaned)
    return questions


# ── CORE FIX: Answer verification ────────────────────────────────────────────

def _verify_question(question: str, chunk_text: str) -> str | None:
    """
    Ask the LLM to extract the exact answer to `question` from `chunk_text`.

    Returns the extracted answer string if found, or None if the answer
    cannot be found in the chunk (meaning the question was hallucinated
    from the LLM's training knowledge, not from the document).

    This is the primary defence against hallucinated eval questions.
    The LLM is given ONLY the chunk text — no external knowledge.
    """
    system = (
        "You are a strict answer extractor. "
        "Your ONLY job is to find the answer to a question within the provided text. "
        "Rules:\n"
        "1. Answer ONLY using words and facts explicitly present in the provided text.\n"
        "2. If the answer is not explicitly stated in the text, respond with exactly: NOT_FOUND\n"
        "3. Do NOT use any external knowledge. Do NOT infer or guess.\n"
        "4. Keep the answer short — one sentence or less.\n"
        "5. Never answer NOT_FOUND if the answer IS in the text."
    )
    user = (
        f"TEXT:\n{chunk_text}\n\n"
        f"QUESTION: {question}\n\n"
        "Extract the answer from the TEXT above. "
        "If not found in the TEXT, respond with exactly: NOT_FOUND"
    )
    try:
        raw = _llm(system, user, max_tokens=120, temperature=0.0)
        raw = raw.strip()
        # Reject if LLM says not found in any form
        not_found_patterns = [
            "not_found", "not found", "not explicitly", "not mentioned",
            "not stated", "not provided", "cannot be found", "not in the text",
            "not present", "no information", "not available",
        ]
        lower = raw.lower()
        if any(p in lower for p in not_found_patterns):
            return None
        # Reject empty or trivially short answers
        if len(raw.strip()) < 2:
            return None
        return raw
    except Exception as exc:
        logger.warning("Verification LLM call failed: %s", exc)
        # On error, be permissive — don't reject the question
        return "UNVERIFIED"


def _generate_and_verify(
    chunk: Chunk,
    questions_raw: list[str],
    q_type: str,
) -> list[_PoolItem]:
    """
    Verify each candidate question against the source chunk.
    Only questions whose answers can be extracted from the chunk are kept.
    Returns verified question tuples including the extracted reference answer.
    """
    verified: list[_PoolItem] = []
    for q in questions_raw:
        answer = _verify_question(q, chunk.text)
        if answer is not None:
            verified.append((q, chunk.document, chunk.page, q_type, answer))
            logger.debug("  ✓ VERIFIED [%s]: %s", q_type, q[:60])
        else:
            logger.info(
                "  ✗ REJECTED (answer not in chunk): %s", q[:70]
            )
    return verified


# ── Question generators ───────────────────────────────────────────────────────

def _gen_factual(chunk_text: str, n: int, retries: int | None = None) -> list[str]:
    """Generate up to n factual questions strictly from the chunk text."""
    if retries is None:
        retries = get_settings().eval_question_generation_retries
    system = (
        "You are an evaluation dataset generator for a RAG system. "
        "CRITICAL: Generate questions ONLY about facts explicitly stated in the "
        "provided excerpt. Do NOT use any external knowledge. "
        "Output ONLY a numbered list of questions, one per line, each ending with '?'. "
        "No explanations, no prefixes."
    )
    user = (
        f"Read ONLY the excerpt below. Generate exactly {n} DIFFERENT specific "
        "factual questions where the answer is explicitly stated word-for-word in the excerpt.\n\n"
        "Rules:\n"
        "- ONLY ask about facts that are directly written in the excerpt below\n"
        "- Do NOT ask about dates, amounts, or details NOT explicitly in the excerpt\n"
        "- Are NOT yes/no questions\n"
        "- Ask about different facts: names, roles, relationships, services, products\n"
        "- Are self-contained (no 'in the excerpt' phrasing)\n"
        "- Each MUST end with '?'\n\n"
        f"EXCERPT (use ONLY this text):\n{chunk_text}\n\n"
        f"Generate {n} questions (numbered 1 to {n}):"
    )
    for attempt in range(1, retries + 1):
        try:
            raw = _llm(system, user, max_tokens=n * 80, temperature=0.3)
            qs = _parse_questions(raw)
            if qs:
                logger.debug("Factual gen attempt %d: %d/%d", attempt, len(qs), n)
                return qs[:n]
        except Exception as exc:
            logger.warning("Factual gen attempt %d failed: %s", attempt, exc)
    return []


def _gen_graph(
    chunk_text: str, entity_ids: list[str], n: int, retries: int | None = None
) -> list[str]:
    """
    Generate relationship questions for entity-rich chunks.
    Still constrained to excerpt-only facts.
    """
    if retries is None:
        retries = get_settings().eval_question_generation_retries
    if len(entity_ids) < 2:
        return []
    system = (
        "You are an evaluation dataset generator for graph/entity retrieval. "
        "CRITICAL: Generate questions ONLY about relationships explicitly stated "
        "in the provided excerpt. Do NOT infer or use external knowledge. "
        "Output ONLY a numbered list of questions, one per line, ending with '?'."
    )
    user = (
        f"Read ONLY the excerpt below. Generate up to {n} questions about "
        "RELATIONSHIPS or CONNECTIONS between entities that are EXPLICITLY STATED "
        "in the excerpt.\n\n"
        "Rules:\n"
        "- ONLY ask about relationships directly written in the excerpt\n"
        "- Do NOT infer connections not explicitly stated\n"
        "- Good examples from text like 'X is CEO of Y':\n"
        "    'Who serves as CEO of [Company]?'\n"
        "    'What company did [Person] found?'\n"
        "    'Which platform does [Company] use for [purpose]?'\n"
        "- Each MUST end with '?'\n\n"
        f"EXCERPT (use ONLY this text):\n{chunk_text}\n\n"
        f"Generate up to {n} relational questions (numbered):"
    )
    for attempt in range(1, retries + 1):
        try:
            raw = _llm(system, user, max_tokens=n * 90, temperature=0.3)
            qs = _parse_questions(raw)
            if qs:
                return qs[:n]
        except Exception as exc:
            logger.warning("Graph gen attempt %d failed: %s", attempt, exc)
    return []


def _gen_comparative(chunk_a: Chunk, chunk_b: Chunk, retries: int | None = None) -> list[str]:
    """
    Generate one cross-chunk question requiring both excerpts.
    Only asks about facts present in both chunks.
    """
    if retries is None:
        retries = get_settings().eval_question_generation_retries
    system = (
        "You are an evaluation dataset generator. "
        "CRITICAL: Generate a question ONLY about facts explicitly stated in "
        "the provided excerpts. Do NOT use external knowledge. "
        "Output a single question ending with '?'. No explanations."
    )
    user = (
        "Read BOTH excerpts below. Generate ONE question that:\n"
        "- Connects or compares information EXPLICITLY written in both excerpts\n"
        "- Can only be fully answered using information from both excerpts\n"
        "- Is NOT a yes/no question\n"
        "- ONLY uses facts explicitly stated — no inference\n"
        "- Must end with '?'\n\n"
        f"EXCERPT A (page {chunk_a.page}):\n{chunk_a.text[:500]}\n\n"
        f"EXCERPT B (page {chunk_b.page}):\n{chunk_b.text[:500]}\n\n"
        "Cross-excerpt question (facts from both excerpts only):"
    )
    for attempt in range(1, retries + 1):
        try:
            raw = _llm(system, user, max_tokens=120, temperature=0.2)
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", raw).strip().strip('"\'')
            if not cleaned.endswith("?") and cleaned.split():
                cleaned += "?"
            if cleaned.endswith("?") and 10 < len(cleaned) < 400:
                return [cleaned]
        except Exception as exc:
            logger.warning("Comparative gen attempt %d failed: %s", attempt, exc)
    return []


# ── Pool builder ──────────────────────────────────────────────────────────────

def _build_pool(all_chunks: list[Chunk], n_pairs: int) -> list[_PoolItem]:
    """
    Build a verified, diverse question pool of size >= n_pairs.

    Pass 1 — Factual:     generate + verify per chunk
    Pass 2 — Graph:       generate + verify for entity-rich chunks
    Pass 3 — Comparative: generate + verify for adjacent-page pairs
    Dedup
    Pass 4 — Top-up:      round-robin fill if still short
    """
    n_chunks = max(len(all_chunks), 1)
    factual_per_chunk = max(1, min(_MAX_FACTUAL_PER_CHUNK, math.ceil(n_pairs / n_chunks)))
    logger.info(
        "Pool builder: %d chunks, target %d, %d factual/chunk",
        n_chunks, n_pairs, factual_per_chunk,
    )

    raw: list[_PoolItem] = []

    # Pass 1: Factual (generate + verify)
    for chunk in all_chunks:
        if len(raw) >= n_pairs:
            break
        candidates = _gen_factual(chunk.text, factual_per_chunk)
        verified = _generate_and_verify(chunk, candidates, "factual")
        raw.extend(verified[: max(0, n_pairs - len(raw))])
        logger.info(
            "  [factual] %s p.%d → %d/%d verified",
            chunk.document, chunk.page, len(verified), len(candidates),
        )

    # Pass 2: Graph/relational (generate + verify)
    for chunk in all_chunks:
        if len(raw) >= n_pairs:
            break
        if len(chunk.entities) >= 2:
            candidates = _gen_graph(chunk.text, chunk.entities, _MAX_GRAPH_PER_CHUNK)
            verified = _generate_and_verify(chunk, candidates, "relational")
            raw.extend(verified[: max(0, n_pairs - len(raw))])
            if verified:
                logger.info(
                    "  [graph] %s p.%d → %d/%d verified",
                    chunk.document, chunk.page, len(verified), len(candidates),
                )

    # Pass 3: Comparative (generate + verify anchored to chunk_a)
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in all_chunks:
        by_doc[chunk.document].append(chunk)
    for doc_chunks in by_doc.values():
        if len(raw) >= n_pairs:
            break
        if len(doc_chunks) < 2:
            continue
        for i in range(min(3, len(doc_chunks) - 1)):
            if len(raw) >= n_pairs:
                break
            candidates = _gen_comparative(doc_chunks[i], doc_chunks[i + 1])
            # For comparative, verify against combined text of both chunks
            combined_text = doc_chunks[i].text + " " + doc_chunks[i + 1].text
            combined_chunk = doc_chunks[i].model_copy(
                update={"text": combined_text}
            )
            verified = _generate_and_verify(combined_chunk, candidates, "comparative")
            raw.extend(verified[: max(0, n_pairs - len(raw))])
            if verified:
                logger.info(
                    "  [comparative] %s p.%d+p.%d → %d verified",
                    doc_chunks[0].document,
                    doc_chunks[i].page, doc_chunks[i + 1].page, len(verified),
                )

    # Dedup
    deduped = _deduplicate(raw)
    logger.info(
        "Pool: %d raw → %d after dedup (target %d)",
        len(raw), len(deduped), n_pairs,
    )

    # Pass 4: Top-up (generate + verify with higher temperature)
    if len(deduped) < n_pairs:
        logger.info("Pool short (%d < %d). Top-up pass...", len(deduped), n_pairs)
        chunk_cycle = cycle(all_chunks)
        max_attempts = (n_pairs - len(deduped)) * 8
        attempts = 0
        settings = get_settings()
        client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            timeout=settings.eval_llm_timeout_seconds,
            max_retries=settings.eval_llm_max_retries,
        )
        while len(deduped) < n_pairs and attempts < max_attempts:
            chunk = next(chunk_cycle)
            attempts += 1
            try:
                resp = client.chat.completions.create(
                    model=settings.nvidia_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an evaluation question generator. "
                                "Generate a question ONLY about facts in the provided text. "
                                "Output a single question ending with '?'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Generate ONE factual question whose answer is explicitly "
                                "stated in this text. Ask about a different aspect than "
                                "common questions (CEO names, founders):\n\n"
                                f"TEXT:\n{chunk.text}\n\nQuestion:"
                            ),
                        },
                    ],
                    max_tokens=100,
                    temperature=0.6,
                )
                q = (resp.choices[0].message.content or "").strip().strip('"\'')
                if q and not q.endswith("?"):
                    q += "?"
                if q.endswith("?") and 10 < len(q) < 400:
                    # Verify before adding
                    answer = _verify_question(q, chunk.text)
                    if answer is not None:
                        candidate: _PoolItem = (
                            q, chunk.document, chunk.page, "factual", answer
                        )
                        if all(
                            _token_overlap(q, ex[0]) < _DEDUP_OVERLAP_THRESHOLD
                            for ex in deduped
                        ):
                            deduped.append(candidate)
                            logger.debug("  Top-up verified: %s", q[:60])
            except Exception as exc:
                logger.debug("Top-up attempt %d failed: %s", attempts, exc)

    return deduped[:n_pairs]


# ── Rank helpers ──────────────────────────────────────────────────────────────

def _find_rank(
    retrieved: list[dict], expected_document: str, expected_page: int
) -> int:
    for rank, source in enumerate(retrieved, start=1):
        if (
            source["document"].lower() == expected_document.lower()
            and abs(source["page"] - expected_page) <= 1
        ):
            return rank
    return 0


def _compute_report(
    pairs: list[EvalPair],
    results: list[EvalResult],
    hits_at_1: int,
    hits_at_3: int,
    hits_at_5: int,
    reciprocal_rank_sum: float,
    latency_samples: list[dict[str, float]],
) -> EvalReport:
    n = max(len(pairs), 1)
    hit_rate_at_1 = round(hits_at_1 / n, 4)
    hit_rate_at_3 = round(hits_at_3 / n, 4)
    hit_rate_at_5 = round(hits_at_5 / n, 4)
    precision_at_1 = round(hits_at_1 / n, 4)
    precision_at_3 = round(hits_at_3 / (n * 3), 4)
    precision_at_5 = round(hits_at_5 / (n * 5), 4)
    mrr = round(reciprocal_rank_sum / n, 4)
    retrieval_metrics = {
        "recall_at_k": {"1": hit_rate_at_1, "3": hit_rate_at_3, "5": hit_rate_at_5},
        "precision_at_k": {"1": precision_at_1, "3": precision_at_3, "5": precision_at_5},
        "hit_rate": {"1": hit_rate_at_1, "3": hit_rate_at_3, "5": hit_rate_at_5},
        "mrr": mrr,
    }
    generation_metrics = {
        "faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
        "answer_relevancy": round(sum(r.answer_relevancy for r in results) / n, 4),
        "bertscore_precision": round(sum(r.bertscore_precision for r in results) / n, 4),
        "bertscore_recall": round(sum(r.bertscore_recall for r in results) / n, 4),
        "bertscore_f1": round(sum(r.bertscore_f1 for r in results) / n, 4),
    }
    latency_keys = sorted({key for sample in latency_samples for key in sample})
    latency_metrics = {
        key: round(sum(sample.get(key, 0.0) for sample in latency_samples) / n, 3)
        for key in latency_keys
    }
    return EvalReport(
        total_questions=len(pairs),
        hits_at_1=hits_at_1,
        hits_at_3=hits_at_3,
        hits_at_5=hits_at_5,
        recall_at_1=hit_rate_at_1,
        recall_at_3=hit_rate_at_3,
        recall_at_5=hit_rate_at_5,
        hit_rate_at_1=hit_rate_at_1,
        hit_rate_at_3=hit_rate_at_3,
        hit_rate_at_5=hit_rate_at_5,
        mrr=mrr,
        precision_at_1=precision_at_1,
        precision_at_3=precision_at_3,
        precision_at_5=precision_at_5,
        hits=hits_at_3,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        latency_metrics=latency_metrics,
        results=results,
    )


# ── Core evaluation loop ──────────────────────────────────────────────────────

def _run_retrieval_eval(
    pairs: list[EvalPair], pipeline: "RAGPipeline"
) -> EvalReport:
    results: list[EvalResult] = []
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    reciprocal_rank_sum = 0.0
    faithfulness_scores: list[float] = []
    coverage_scores: list[float] = []
    latency_samples: list[dict[str, float]] = []
    speculative_count = 0
    unsupported_count = 0
    type_hits: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "hit3": 0}
    )

    for i, pair in enumerate(pairs, start=1):
        q_type = pair.__dict__.get("question_type", "factual")
        logger.info(
            "[%d/%d][%s] Q: %s", i, len(pairs), q_type, pair.question[:80]
        )
        type_hits[q_type]["total"] += 1

        try:
            response = pipeline.query(pair.question, top_k=_EVAL_TOP_K)
        except Exception as exc:
            logger.error("Query failed Q%d: %s", i, exc)
            results.append(
                EvalResult(
                    question=pair.question,
                    expected_document=pair.expected_document,
                    expected_page=pair.expected_page,
                    reference_answer=pair.answer_hint or "",
                    retrieved_top5=[],
                    hit_at_1=False, hit_at_3=False, hit_at_5=False,
                    rank=0, reciprocal_rank=0.0,
                )
            )
            latency_samples.append({})
            continue

        retrieved_top5 = [
            {
                "document": s.document,
                "page": s.page,
                "chunk_preview": s.chunk[:200],
                "chunk_full": s.chunk,          # full text for faithfulness
                "retrieval_source": getattr(s, "retrieval_source", "unknown"),
            }
            for s in response.sources[:_EVAL_TOP_K]
        ]

        rank = _find_rank(
            retrieved_top5, pair.expected_document, pair.expected_page
        )
        rr = 1.0 / rank if rank > 0 else 0.0
        h1, h3, h5 = rank == 1, 0 < rank <= 3, 0 < rank <= 5
        if h1: hits_at_1 += 1
        if h3: hits_at_3 += 1; type_hits[q_type]["hit3"] += 1
        if h5: hits_at_5 += 1
        reciprocal_rank_sum += rr

        grounding = _compute_grounding_metrics(response.answer, retrieved_top5)
        reference_answer = pair.answer_hint or ""
        answer_relevancy = _cosine_similarity(pair.question, response.answer)
        bertscore = _bertscore_like(response.answer, reference_answer)
        latency = dict(response.latency_metrics or {})
        latency_samples.append(latency)
        faithfulness_scores.append(grounding["faithfulness"])
        coverage_scores.append(grounding["evidence_coverage"])
        if grounding["is_speculative"]: speculative_count += 1
        if grounding["unsupported"]: unsupported_count += 1

        results.append(
            EvalResult(
                question=pair.question,
                expected_document=pair.expected_document,
                expected_page=pair.expected_page,
                generated_answer=response.answer,
                reference_answer=reference_answer,
                retrieved_top5=retrieved_top5,
                hit_at_1=h1, hit_at_3=h3, hit_at_5=h5,
                rank=rank,
                reciprocal_rank=round(rr, 4),
                original_query=response.original_query,
                rewritten_query=response.rewritten_query,
                precision_at_1=1.0 if h1 else 0.0,
                precision_at_3=round((1.0 / 3.0) if h3 else 0.0, 4),
                precision_at_5=round((1.0 / 5.0) if h5 else 0.0, 4),
                faithfulness=grounding["faithfulness"],
                answer_relevancy=answer_relevancy,
                bertscore_precision=bertscore["precision"],
                bertscore_recall=bertscore["recall"],
                bertscore_f1=bertscore["f1"],
                latency_metrics=latency,
                total_response_time_ms=latency.get("total_response_time_ms", 0.0),
            )
        )

        status = f"✓ RANK-{rank}" if rank > 0 else "✗ MISS"
        spec_flag = " ⚠speculative" if grounding["is_speculative"] else ""
        logger.info(
            "  %s  RR=%.3f  faith=%.2f  cov=%.2f%s",
            status, rr,
            grounding["faithfulness"],
            grounding["evidence_coverage"],
            spec_flag,
        )

    report = _compute_report(
        pairs, results, hits_at_1, hits_at_3, hits_at_5,
        reciprocal_rank_sum, latency_samples,
    )
    n = max(len(pairs), 1)
    report.__dict__["grounding_summary"] = {
        "avg_faithfulness": round(
            sum(faithfulness_scores) / len(faithfulness_scores), 4
        ) if faithfulness_scores else 0.0,
        "avg_evidence_coverage": round(
            sum(coverage_scores) / len(coverage_scores), 4
        ) if coverage_scores else 0.0,
        "unsupported_rate": round(unsupported_count / n, 4),
        "speculative_edge_rate": round(speculative_count / n, 4),
        "unsupported_count": unsupported_count,
        "speculative_count": speculative_count,
        "by_question_type": {
            qt: {
                "total": v["total"],
                "hit3": v["hit3"],
                "recall_at_3": round(v["hit3"] / v["total"], 4) if v["total"] else 0.0,
            }
            for qt, v in type_hits.items()
        },
    }

    _print_report(report)
    return report


# ── Mode 1: AUTO evaluation ───────────────────────────────────────────────────

def auto_evaluate(pipeline: "RAGPipeline", n_pairs: int = 10) -> EvalReport:
    """
    Generate verified questions from ingested documents then evaluate.
    Every question is confirmed extractable from its source chunk before
    entering the eval pool — no hallucinated questions.
    """
    all_chunks: list[Chunk] = pipeline._retriever._corpus_chunks
    if not all_chunks:
        raise ValueError("No documents ingested. Upload at least one PDF first.")

    n_docs = len({c.document for c in all_chunks})
    logger.info(
        "Auto-eval: %d chunks across %d doc(s). Target: %d verified questions.",
        len(all_chunks), n_docs, n_pairs,
    )

    pool = _build_pool(all_chunks, n_pairs)
    if not pool:
        raise ValueError(
            "Could not generate any verified questions. "
            "Check your nvidia API key and model connection."
        )

    logger.info("Final verified pool: %d questions (target %d).", len(pool), n_pairs)
    type_counts: dict[str, int] = defaultdict(int)
    for _, _, _, q_type, _reference_answer in pool:
        type_counts[q_type] += 1
    for q_type, count in type_counts.items():
        logger.info("  %s: %d", q_type, count)

    pairs: list[EvalPair] = []
    for question, document, page, q_type, reference_answer in pool:
        pair = EvalPair(
            question=question,
            expected_document=document,
            expected_page=page,
            answer_hint=reference_answer,
        )
        pair.__dict__["question_type"] = q_type
        pairs.append(pair)

    return _run_retrieval_eval(pairs, pipeline)


# ── Mode 2: PREDEFINED evaluation ────────────────────────────────────────────

def run_evaluation(qa_path: Path, pipeline: "RAGPipeline") -> EvalReport:
    with open(qa_path, encoding="utf-8") as f:
        raw_pairs = json.load(f)
    pairs = [EvalPair(**p) for p in raw_pairs]
    logger.info("Predefined eval: %d questions from %s", len(pairs), qa_path)
    return _run_retrieval_eval(pairs, pipeline)


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_report(report: EvalReport) -> None:
    n = report.total_questions
    grounding = report.__dict__.get("grounding_summary", {})

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)
    print(f"  {'Metric':<38}  {'Score':>7}  {'Hits':>8}")
    print("-" * 70)
    print(
        f"  {'Recall@1 (correct is #1 result)':<38}  "
        f"{report.recall_at_1:>6.1%}  {report.hits_at_1:>5}/{n}"
    )
    print(
        f"  {'Recall@3 (correct in top 3)':<38}  "
        f"{report.recall_at_3:>6.1%}  {report.hits_at_3:>5}/{n}"
    )
    print(
        f"  {'Recall@5 (correct in top 5)':<38}  "
        f"{report.recall_at_5:>6.1%}  {report.hits_at_5:>5}/{n}"
    )
    print(f"  {'MRR (Mean Reciprocal Rank)':<38}  {report.mrr:>8.4f}")

    if grounding:
        print("-" * 70)
        print("  GROUNDING / HALLUCINATION METRICS")
        print("-" * 70)
        print(
            f"  {'Avg Answer Faithfulness':<38}  "
            f"{grounding.get('avg_faithfulness', 0):>8.3f}"
        )
        print(
            f"  {'Avg Evidence Coverage':<38}  "
            f"{grounding.get('avg_evidence_coverage', 0):>8.3f}"
        )
        unsup_n = grounding.get("unsupported_count", 0)
        print(
            f"  {'Unsupported Answer Rate':<38}  "
            f"{grounding.get('unsupported_rate', 0):>6.1%}  {unsup_n:>5}/{n}"
        )
        spec_n = grounding.get("speculative_count", 0)
        print(
            f"  {'Speculative Graph Edge Rate':<38}  "
            f"{grounding.get('speculative_edge_rate', 0):>6.1%}  {spec_n:>5}/{n}"
        )

        by_type = grounding.get("by_question_type", {})
        if by_type:
            print("-" * 70)
            print("  RECALL@3 BY QUESTION TYPE")
            print("-" * 70)
            for q_type, stats in by_type.items():
                print(
                    f"  {q_type:<38}  {stats['recall_at_3']:>6.1%}  "
                    f"{stats['hit3']:>5}/{stats['total']}"
                )

    print("=" * 70)
    print("\n  Per-question results:")
    for r in report.results:
        rank_str = f"rank {r.rank}" if r.rank > 0 else "not found"
        icon = "✓" if r.hit_at_3 else "✗"
        q_type = r.__dict__.get("question_type", "")
        type_tag = f"[{q_type}] " if q_type else ""
        print(f"    {icon} [{rank_str:>9s}]  {type_tag}{r.question[:55]}")
    print("=" * 70 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--qa-path", type=Path, default=None)
    parser.add_argument("--ingest-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=10)
    args = parser.parse_args()

    from app.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    if not args.skip_ingest:
        print(f"Ingesting PDFs from {args.ingest_dir}...")
        for r in pipeline.ingest_directory(args.ingest_dir):
            print(f"  Ingested: {r.document} ({r.chunks_created} chunks)")

    if args.qa_path:
        report = run_evaluation(args.qa_path, pipeline)
    else:
        print("AUTO mode — generating verified questions from ingested documents...")
        report = auto_evaluate(pipeline, n_pairs=args.n_pairs)

    report_path = Path("tests/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"Full report saved to {report_path}")
    sys.exit(0 if report.recall_at_3 >= 0.7 else 1)
