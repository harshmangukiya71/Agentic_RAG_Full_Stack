"""
query_agent.py — Query Planning Agent for the Agentic RAG pipeline.

Classifies user queries into semantic categories and determines the
optimal retrieval strategy + parameters.

Categories:
  LOOKUP       — Simple fact retrieval ("What is X?")
  COMPARISON   — Comparing two or more items ("Compare A and B")
  RANKING      — Ordering by a criterion ("Rank all products by sales")
  AGGREGATION  — Numerical aggregation ("What is total revenue?")
  COUNTING     — Counting items ("How many employees?")
  MULTI_HOP    — Requires chaining facts ("Who manages the team that built X?")
  RELATIONSHIP — Entity relationships ("What is the relationship between A and B?")
  TEMPORAL     — Time-based reasoning ("What changed between 2022 and 2023?")
  ANALYTICAL   — Deep analysis ("Analyse the risk factors")

All classification is done via lightweight regex + keyword heuristics.
No LLM call is needed — keeps latency near-zero.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

from app.models import QueryClassification

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    ENTITY_LOOKUP = "ENTITY_LOOKUP"
    SEMANTIC_LOOKUP = "SEMANTIC_LOOKUP"
    NUMERICAL_FILTER = "NUMERICAL_FILTER"
    LOOKUP = "LOOKUP"
    COMPARISON = "COMPARISON"
    RANKING = "RANKING"
    AGGREGATION = "AGGREGATION"
    COUNTING = "COUNTING"
    MULTI_HOP = "MULTI_HOP"
    RELATIONSHIP = "RELATIONSHIP"
    TEMPORAL = "TEMPORAL"
    ANALYTICAL = "ANALYTICAL"


# ── Pattern sets (compiled once) ──────────────────────────────────────────────

_ENTITY_LOOKUP_RE = re.compile(
    r"\b(?:[A-Z]{1,10}-\d+|"
    r"(?:contract|invoice|employee|department)\s+(?:id\s*)?[A-Z]{1,10}-\d+)\b",
    re.IGNORECASE,
)

_SEMANTIC_LOOKUP_START_RE = re.compile(
    r"^\s*(?:which|what|identify|find)\b",
    re.IGNORECASE,
)

_SEMANTIC_BUSINESS_RE = re.compile(
    r"\b(?:initiative|venture|effort|program|activity|project|"
    r"earning|earnings|earned|revenue|income|generated|generate|"
    r"produced|produce|sales)\b",
    re.IGNORECASE,
)

_NUMERICAL_FILTER_RE = re.compile(
    r"\b(?:greater\s+than|less\s+than|more\s+than|fewer\s+than|"
    r"at\s+least|at\s+most|no\s+less\s+than|no\s+more\s+than|"
    r"above|below|under|over|between)\b|>=|<=|>|<|=",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(?:compar(?:e|ison|ing)|differ(?:ence|ent|s)|versus|vs\.?|"
    r"contrast|distinguish|same\s+as|similar\s+to|"
    r"higher\s+than|lower\s+than|more\s+than|less\s+than|"
    r"between\s+\w+\s+and)\b",
    re.IGNORECASE,
)

_RANKING_RE = re.compile(
    r"\b(?:rank(?:ing|ed)?|top\s+\d+|bottom\s+\d+|best|worst|"
    r"highest|lowest|largest|smallest|most|least|"
    r"sort(?:ed)?\s+by|order(?:ed)?\s+by|descending|ascending)\b",
    re.IGNORECASE,
)

_AGGREGATION_RE = re.compile(
    r"\b(?:total|sum|average|mean|aggregate|overall|combined|"
    r"net|gross|cumulative|add\s+up|sum\s+of|"
    r"list\s+all|show\s+all|display\s+all|enumerate|"
    r"list|show\s+every|every)\b",
    re.IGNORECASE,
)

_COUNTING_RE = re.compile(
    r"\b(?:how\s+many|count(?:ing|ed)?|number\s+of|"
    r"quantity|total\s+number|amount\s+of)\b",
    re.IGNORECASE,
)

_GLOBAL_SCOPE_RE = re.compile(
    r"\b(?:all|every|overall|entire|total|aggregate|"
    r"list|show\s+all|display\s+all|enumerate|"
    r"highest|lowest|most|least|best|worst|"
    r"top\s+\d+|bottom\s+\d+)\b",
    re.IGNORECASE,
)

_SUPERLATIVE_RE = re.compile(
    r"\b(?:highest|lowest|most|least|best|worst)\b",
    re.IGNORECASE,
)

_MULTI_ITEM_RE = re.compile(
    r"\b(?:all|every|top\s+\d+|bottom\s+\d+|list|rank)\b",
    re.IGNORECASE,
)

_MULTI_HOP_RE = re.compile(
    r"\b(?:who\s+(?:\w+\s+){1,4}(?:that|which|where)|"
    r"what\s+(?:\w+\s+){1,4}(?:that|which|where)|"
    r"which\s+.+\b(?:belong(?:s)?\s+to|managed\s+by|owned\s+by|"
    r"associated\s+with|linked\s+to|connected\s+to|related\s+to)\b.+|"
    r"\b(?:belong(?:s)?\s+to|managed\s+by|owned\s+by|associated\s+with|"
    r"linked\s+to|connected\s+to|related\s+to)\b.+\b(?:managed\s+by|owned\s+by|belongs\s+to)\b|"
    r"find.*?(?:then|and\s+then|next)|"
    r"based\s+on.*?(?:what|which|who)|"
    r"according\s+to.*?(?:what|how)|"
    r"using\s+the\s+.*?(?:find|determine|calculate))\b",
    re.IGNORECASE,
)

_RELATIONSHIP_RE = re.compile(
    r"\b(?:relationship|related\s+to|connected\s+to|"
    r"association|linked\s+to|affiliated|"
    r"between\s+\w+\s+and\s+\w+|"
    r"how\s+(?:is|are)\s+\w+\s+(?:related|connected)|"
    r"what\s+(?:is|are)\s+the\s+(?:link|connection|relation))\b",
    re.IGNORECASE,
)

_TEMPORAL_RE = re.compile(
    r"\b(?:when|before|after|during|since|until|"
    r"timeline|chronolog|history\s+of|"
    r"over\s+time|year[-\s]over[-\s]year|quarter\s+over\s+quarter|"
    r"changed?\s+(?:in|from|since|between)|"
    r"trend|growth|decline|evolution|progression|"
    r"\d{4}\s*(?:to|through|-)\s*\d{4})\b",
    re.IGNORECASE,
)

_ANALYTICAL_RE = re.compile(
    r"\b(?:analy[sz]e|assess|evaluat|investigat|examin|review|"
    r"implications?|impact|significance|"
    r"strengths?\s+and\s+weakness|pros?\s+and\s+cons?|"
    r"risk\s+factor|opportunity|challenge|"
    r"why\s+(?:did|does|is|are|was|were)|"
    r"explain\s+(?:the|why|how))\b",
    re.IGNORECASE,
)


# ── Strategy → top_k mapping ─────────────────────────────────────────────────

_STRATEGY_MAP: dict[QueryType, dict] = {
    QueryType.ENTITY_LOOKUP: {
        "retrieval_strategy": "exact_graph_metadata",
        "top_k": 10,
        "reasoning_hints": ["extract_entity_id", "exact_match_lookup"],
    },
    QueryType.SEMANTIC_LOOKUP: {
        "retrieval_strategy": "dense_heavy",
        "top_k": 15,
        "reasoning_hints": ["semantic_match", "entity_resolution"],
    },
    QueryType.NUMERICAL_FILTER: {
        "retrieval_strategy": "expanded",
        "top_k": 50,
        "reasoning_hints": ["extract_numbers", "apply_numeric_filter"],
    },
    QueryType.LOOKUP: {
        "retrieval_strategy": "exact_vector",
        "top_k": 5,
        "reasoning_hints": [],
    },
    QueryType.COMPARISON: {
        "retrieval_strategy": "graph_vector",
        "top_k": 20,
        "reasoning_hints": ["extract_entities", "compare_attributes"],
    },
    QueryType.RANKING: {
        "retrieval_strategy": "graph_vector_exact",
        "top_k": 50,
        "reasoning_hints": ["extract_entities", "sort_by_criterion"],
    },
    QueryType.AGGREGATION: {
        "retrieval_strategy": "expanded",
        "top_k": 50,
        "reasoning_hints": ["extract_numbers", "aggregate", "collect_all_entities"],
    },
    QueryType.COUNTING: {
        "retrieval_strategy": "expanded",
        "top_k": 30,
        "reasoning_hints": ["extract_entities", "count"],
    },
    QueryType.MULTI_HOP: {
        "retrieval_strategy": "iterative_graph",
        "top_k": 30,
        "reasoning_hints": ["graph_traversal", "chain_facts"],
    },
    QueryType.RELATIONSHIP: {
        "retrieval_strategy": "graph_vector",
        "top_k": 20,
        "reasoning_hints": ["extract_entities", "resolve_relationships"],
    },
    QueryType.TEMPORAL: {
        "retrieval_strategy": "graph_vector_exact",
        "top_k": 30,
        "reasoning_hints": ["extract_dates", "order_chronologically"],
    },
    QueryType.ANALYTICAL: {
        "retrieval_strategy": "expanded",
        "top_k": 30,
        "reasoning_hints": ["synthesize", "evaluate_evidence"],
    },
}


# ── Public API ────────────────────────────────────────────────────────────────


class QueryPlanningAgent:
    """
    Zero-latency query classifier.

    Uses cascading regex patterns (most specific first) to determine the
    semantic category of a user query, then maps it to a retrieval strategy
    and top_k value.
    """

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a user query and return retrieval parameters.

        Returns a QueryClassification with:
          - query_type: one of the QueryType enum values
          - retrieval_strategy: routing key for HybridRetriever
          - top_k: suggested retrieval depth
          - reasoning_hints: list of reasoning operations to perform
        """
        query_type = self._detect_type(query)
        strategy = dict(_STRATEGY_MAP[query_type])
        reasoning_hints = list(strategy["reasoning_hints"])

        if (
            query_type == QueryType.RANKING
            and _SUPERLATIVE_RE.search(query)
            and not _MULTI_ITEM_RE.search(query)
        ):
            strategy["top_k"] = 20
            reasoning_hints.append("find_top_1")

        classification = QueryClassification(
            query_type=query_type.value,
            query_scope=self._detect_scope(query),
            retrieval_strategy=strategy["retrieval_strategy"],
            top_k=strategy["top_k"],
            reasoning_hints=reasoning_hints,
        )

        logger.info(
            "Query classified: type=%s scope=%s strategy=%s top_k=%d | %r",
            classification.query_type,
            classification.query_scope,
            classification.retrieval_strategy,
            classification.top_k,
            query[:80],
        )
        return classification

    def _detect_scope(self, query: str) -> str:
        """Detect corpus-wide intent with O(query_length) regex matching."""
        return "GLOBAL" if _GLOBAL_SCOPE_RE.search(query.strip()) else "LOCAL"

    def _detect_type(self, query: str) -> QueryType:
        """
        Cascading classification — most specific patterns first.

        Order matters:  COUNTING before AGGREGATION (because "how many" is
        more specific than "total").  MULTI_HOP after other types because
        its patterns are broader.
        """
        q = query.strip()

        # Most specific first
        if _MULTI_HOP_RE.search(q):
            return QueryType.MULTI_HOP

        if _NUMERICAL_FILTER_RE.search(q):
            return QueryType.NUMERICAL_FILTER

        if _ENTITY_LOOKUP_RE.search(q):
            return QueryType.ENTITY_LOOKUP

        if _SEMANTIC_LOOKUP_START_RE.search(q) and _SEMANTIC_BUSINESS_RE.search(q):
            return QueryType.SEMANTIC_LOOKUP

        if _COUNTING_RE.search(q):
            return QueryType.COUNTING

        if _RANKING_RE.search(q):
            return QueryType.RANKING

        if _COMPARISON_RE.search(q):
            return QueryType.COMPARISON

        if _AGGREGATION_RE.search(q):
            return QueryType.AGGREGATION

        if _RELATIONSHIP_RE.search(q):
            return QueryType.RELATIONSHIP

        if _TEMPORAL_RE.search(q):
            return QueryType.TEMPORAL

        if _ANALYTICAL_RE.search(q):
            return QueryType.ANALYTICAL

        # Default
        return QueryType.LOOKUP
