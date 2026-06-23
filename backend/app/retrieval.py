"""
retrieval.py — BM25 + dense + graph retrieval with RRF and cross-encoder reranking.

Changes from previous version:
  • Added _stem() — zero-dependency suffix stripper so BM25 matches
    morphological variants: "founders"/"founded"/"founding" all → "found",
    "invested"/"investing" → "invest", etc. Fixes the Recall@1 miss on
    "Who are the founders of Google?" style queries.
  • _tokenize() now stems every token before indexing and querying.
  • _tokenize_query_with_prefix() also stems before prefix expansion so
    prefix matching works on root forms.
  • Evidence filter and all other logic unchanged from previous version.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_GRAPH_EDGE_MIN_CONFIDENCE: float = 0.70
_RRF_NOISE_FLOOR: float = 0.005

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
        )
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ── Lightweight zero-dependency stemmer ───────────────────────────────────────

def _stem(word: str) -> str:
    """
    Strip common English suffixes to get an approximate root form.
    Zero external dependencies — no NLTK, no snowball.

    Handles the most common retrieval mismatches:
      founders/founded/founding → found
      invested/investing/investment → invest
      develops/developer/development → develop
      companies/company → compani / compan  (still matches)
      serves/served/serving → serv
    """
    w = word.lower()
    # Longest suffixes first to avoid over-stripping
    for suffix in (
        "nesses", "ations", "ments",
        "iness", "ation", "ments", "ings", "ness", "ment",
        "ers", "ing", "ion", "ies", "ful",
        "ed", "er", "es", "ly",
        "s",
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def _tokenize(text: str) -> list[str]:
    """Lowercase-split then stem every token."""
    return [_stem(tok) for tok in text.lower().split()]


def _tokenize_query_with_prefix(query: str, corpus_vocab: set[str]) -> list[str]:
    """
    Stem query tokens then expand any that are missing from the stemmed
    corpus vocab via prefix matching on the stemmed vocab.
    """
    tokens = [_stem(tok) for tok in query.lower().split()]
    expanded: list[str] = []
    for tok in tokens:
        if len(tok) > 2 and tok not in corpus_vocab:
            matches = [w for w in corpus_vocab if w.startswith(tok) and w != tok]
            expanded.extend(matches[:5] or [tok])
        else:
            expanded.append(tok)
    return expanded


def _reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk], k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
    entity_map: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            uid = f"{chunk.document}::{chunk.chunk_index}"
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            chunk_map[uid] = chunk
            entity_map.setdefault(uid, set()).update(chunk.entity_matches)

    fused: list[RetrievedChunk] = []
    for uid in sorted(scores, key=lambda u: scores[u], reverse=True):
        c = chunk_map[uid]
        fused.append(
            RetrievedChunk(
                document=c.document,
                page=c.page,
                chunk_index=c.chunk_index,
                chunk=c.chunk,
                relevance_score=round(scores[uid], 6),
                retrieval_source=c.retrieval_source,
                entity_matches=sorted(entity_map.get(uid, set())),
            )
        )
    return fused


def _filter_low_evidence_chunks(
    fused: list[RetrievedChunk],
    query_entity_ids: set[str],
) -> list[RetrievedChunk]:
    """
    Drop chunks that are BOTH below the RRF noise floor AND have zero entity
    overlap with the query.  Chunks with entity signal are always kept.
    """
    kept: list[RetrievedChunk] = []
    dropped = 0
    for chunk in fused:
        has_entity_signal = bool(query_entity_ids & set(chunk.entity_matches))
        if chunk.relevance_score >= _RRF_NOISE_FLOOR or has_entity_signal:
            kept.append(chunk)
        else:
            dropped += 1
    if dropped:
        logger.debug(
            "Evidence filter dropped %d low-signal chunks", dropped
        )
    return kept


class HybridRetriever:
    def __init__(self) -> None:
        self._corpus_chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        self._corpus_vocab: set[str] = set()
        self.last_timings: dict[str, float] = {}

    def _elapsed_ms(self, started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def rebuild_bm25(self, all_chunks: list[Chunk]) -> None:
        self._corpus_chunks = all_chunks
        # Index using stemmed tokens so query stems match
        self._tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
        self._corpus_vocab = {
            tok for tokens in self._tokenized_corpus for tok in tokens
        }
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        logger.info(
            "BM25 index rebuilt with %d chunks, vocab size: %d",
            len(all_chunks), len(self._corpus_vocab),
        )

    def bm25_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._corpus_chunks:
            return []

        # Stem the query tokens before scoring — matches stemmed index
        tokenized_query = (
            _tokenize_query_with_prefix(query, self._corpus_vocab)
            if self._corpus_vocab
            else _tokenize(query)
        )
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        max_score = max_score or 1.0

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self._corpus_chunks[idx]
            norm_score = float(scores[idx]) / max_score
            if norm_score > 0.001:
                results.append(
                    RetrievedChunk(
                        document=chunk.document,
                        page=chunk.page,
                        chunk_index=chunk.chunk_index,
                        chunk=chunk.text,
                        relevance_score=round(norm_score, 4),
                        retrieval_source="bm25",
                        entity_matches=chunk.entities,
                    )
                )
        return results

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        encoder = _get_cross_encoder()
        ce_scores = encoder.predict([(query, c.chunk) for c in candidates])
        scored = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)

        reranked: list[RetrievedChunk] = []
        for ce_score, chunk in scored[:top_k]:
            sigmoid_score = 1.0 / (1.0 + math.exp(-float(ce_score)))
            reranked.append(
                RetrievedChunk(
                    document=chunk.document,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    chunk=chunk.chunk,
                    relevance_score=round(sigmoid_score, 4),
                    retrieval_source=chunk.retrieval_source,
                    entity_matches=chunk.entity_matches,
                )
            )
        return reranked

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        vector_store,
        graph_store=None,
        top_k_final: int = 5,
        graph_results: list[RetrievedChunk] | None = None,
        query_entity_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid pipeline:
          BM25 (stemmed) + Dense → RRF → evidence filter →
          graph boost → top-20 → cross-encoder rerank → top_k_final
        """
        settings = get_settings()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        bm25_results = self.bm25_retrieve(query, top_k=settings.bm25_top_k)
        timings["bm25_retrieval_time_ms"] = self._elapsed_ms(started)

        started = time.perf_counter()
        dense_results = vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=settings.dense_top_k,
        )
        timings["dense_retrieval_time_ms"] = self._elapsed_ms(started)
        graph_results = graph_results or []

        started = time.perf_counter()
        fused = _reciprocal_rank_fusion(bm25_results, dense_results, graph_results)
        fused = _filter_low_evidence_chunks(fused, query_entity_ids or set())
        timings["rrf_fusion_time_ms"] = self._elapsed_ms(started)

        started = time.perf_counter()
        if graph_results:
            graph_keys = {(c.document, c.chunk_index) for c in graph_results}
            max_boost = settings.graph_boost_weight
            fused = sorted(
                [
                    chunk.model_copy(
                        update={
                            "relevance_score": round(
                                min(1.0, chunk.relevance_score + (
                                    max_boost
                                    if (chunk.document, chunk.chunk_index) in graph_keys
                                    else 0.0
                                )),
                                6,
                            ),
                            "retrieval_source": (
                                "graph_hybrid"
                                if (chunk.document, chunk.chunk_index) in graph_keys
                                else chunk.retrieval_source
                            ),
                        }
                    )
                    for chunk in fused
                ],
                key=lambda c: c.relevance_score,
                reverse=True,
            )
        timings["graph_boosting_time_ms"] = self._elapsed_ms(started)

        started = time.perf_counter()
        reranked = self.rerank(query, fused[:20], top_k=settings.rerank_top_k)
        timings["cross_encoder_rerank_time_ms"] = self._elapsed_ms(started)
        timings["reranking_time_ms"] = timings["cross_encoder_rerank_time_ms"]
        self.last_timings = timings
        logger.debug(
            "Hybrid: bm25=%d dense=%d graph=%d fused=%d reranked=%d",
            len(bm25_results), len(dense_results),
            len(graph_results), len(fused), len(reranked),
        )
        return reranked[:top_k_final]

    # ── Strategy-based retrieval (agentic routing) ────────────────────────────

    # Maps strategy names → which sources to include
    _STRATEGY_SOURCES: dict[str, dict] = {
        "exact_vector": {"bm25": True, "dense": True, "graph": False},
        "exact_graph_metadata": {"bm25": True, "dense": False, "graph": True},
        "dense_heavy": {"bm25": True, "dense": True, "graph": True},
        "graph_vector": {"bm25": False, "dense": True, "graph": True},
        "graph_vector_exact": {"bm25": True, "dense": True, "graph": True},
        "expanded": {"bm25": True, "dense": True, "graph": True},
        "iterative": {"bm25": True, "dense": True, "graph": True},
        "iterative_graph": {"bm25": True, "dense": True, "graph": True},
        "hybrid": {"bm25": True, "dense": True, "graph": True},
    }

    def retrieve_with_strategy(
        self,
        query: str,
        query_embedding: np.ndarray,
        vector_store,
        strategy: str = "hybrid",
        top_k: int = 5,
        graph_results: list[RetrievedChunk] | None = None,
        query_entity_ids: set[str] | None = None,
        query_type: str | None = None,
        query_scope: str = "LOCAL",
    ) -> list[RetrievedChunk]:
        """
        Dynamic retrieval routing based on query classification.

        Strategy names map to different source combinations and retrieval
        depths.  Falls back to full hybrid if the strategy is unknown.
        """
        settings = get_settings()
        sources = self._STRATEGY_SOURCES.get(strategy, self._STRATEGY_SOURCES["hybrid"])
        timings: dict[str, float] = {}

        logger.info(
            "Strategy retrieval: strategy=%s top_k=%d sources=%s",
            strategy, top_k, sources,
        )

        # ── Gather results from enabled sources ──────────────────────────────
        bm25_results: list[RetrievedChunk] = []
        dense_results: list[RetrievedChunk] = []
        g_results: list[RetrievedChunk] = graph_results or []

        if sources["bm25"]:
            # Scale BM25 top_k proportionally to the requested top_k
            bm25_multiplier = 4 if strategy == "exact_graph_metadata" else 2
            bm25_k = max(settings.bm25_top_k, top_k * bm25_multiplier)
            started = time.perf_counter()
            bm25_results = self.bm25_retrieve(query, top_k=bm25_k)
            timings["bm25_retrieval_time_ms"] = self._elapsed_ms(started)
        else:
            timings["bm25_retrieval_time_ms"] = 0.0

        if sources["dense"]:
            dense_multiplier = 4 if strategy == "dense_heavy" else 2
            dense_k = max(settings.dense_top_k, top_k * dense_multiplier)
            started = time.perf_counter()
            dense_results = vector_store.query(
                query_embedding=query_embedding.tolist(),
                top_k=dense_k,
            )
            timings["dense_retrieval_time_ms"] = self._elapsed_ms(started)
        else:
            timings["dense_retrieval_time_ms"] = 0.0

        if not sources["graph"]:
            g_results = []

        # ── Fuse ─────────────────────────────────────────────────────────────
        started = time.perf_counter()
        fused = _reciprocal_rank_fusion(bm25_results, dense_results, g_results)
        fused = _filter_low_evidence_chunks(fused, query_entity_ids or set())
        timings["rrf_fusion_time_ms"] = self._elapsed_ms(started)

        # ── Graph boost ──────────────────────────────────────────────────────
        started = time.perf_counter()
        if g_results:
            graph_keys = {(c.document, c.chunk_index) for c in g_results}
            max_boost = settings.graph_boost_weight
            fused = sorted(
                [
                    chunk.model_copy(
                        update={
                            "relevance_score": round(
                                min(1.0, chunk.relevance_score + (
                                    max_boost
                                    if (chunk.document, chunk.chunk_index) in graph_keys
                                    else 0.0
                                )),
                                6,
                            ),
                            "retrieval_source": (
                                "graph_hybrid"
                                if (chunk.document, chunk.chunk_index) in graph_keys
                                else chunk.retrieval_source
                            ),
                        }
                    )
                    for chunk in fused
                ],
                key=lambda c: c.relevance_score,
                reverse=True,
            )
        timings["graph_boosting_time_ms"] = self._elapsed_ms(started)

        # Keep local lookups fast; widen only for bounded corpus-wide reasoning.
        if (
            query_scope == "GLOBAL"
            and query_type in {"RANKING", "AGGREGATION", "COUNTING", "NUMERICAL_FILTER"}
        ):
            rerank_window = min(100, top_k * 4)
        elif strategy == "dense_heavy":
            rerank_window = min(100, top_k * 4)
        else:
            rerank_window = 20
        rerank_k = min(rerank_window, len(fused))
        started = time.perf_counter()
        reranked = self.rerank(query, fused[:rerank_window], top_k=rerank_k)
        timings["cross_encoder_rerank_time_ms"] = self._elapsed_ms(started)
        timings["reranking_time_ms"] = timings["cross_encoder_rerank_time_ms"]
        self.last_timings = timings

        logger.debug(
            "Strategy hybrid: bm25=%d dense=%d graph=%d fused=%d reranked=%d → top_k=%d",
            len(bm25_results), len(dense_results),
            len(g_results), len(fused), len(reranked), top_k,
        )
        return reranked[:top_k]
