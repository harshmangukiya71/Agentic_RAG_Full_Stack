"""
reasoning_agent.py — Post-retrieval Reasoning Agent for the Agentic RAG pipeline.

Purpose:
  Perform structured reasoning over raw retrieval output BEFORE final LLM
  generation.  The agent transforms unstructured chunk text into structured
  evidence (entities, relationships, calculations, rankings) that the
  generation layer can answer from directly — reducing hallucination and
  improving accuracy on complex query types.

Capabilities:
  • Entity extraction from retrieved chunks
  • Relationship resolution between extracted entities
  • Numerical extraction and aggregation (sum, average, count, min, max)
  • Ranking and sorting by extracted criteria
  • Multi-hop synthesis (chaining facts across chunks)
  • Evidence sufficiency evaluation

Document-agnostic: works with ANY uploaded corpus.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from app.models import QueryClassification, ReasoningOutput, RetrievedChunk

logger = logging.getLogger(__name__)

# ── Number extraction ─────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"""
    (?:(?:[$€£₹¥])\s*)?        # optional currency symbol
    (\d{1,3}(?:,\d{3})*         # integer with commas  e.g. 1,234,567
     (?:\.\d+)?                 # optional decimal
    |\d+(?:\.\d+)?)             # or plain number
    \s*
    (%|percent|million|billion|trillion|thousand|k|m|b)?  # optional unit
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MULTIPLIER = {
    "k": 1_000,
    "thousand": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_NUMERIC_FILTER_RE = re.compile(
    r"\b(?P<metric>revenue|income|earnings|amount|transactions?|sales|profit)?\s*"
    r"(?P<op>greater\s+than|less\s+than|more\s+than|fewer\s+than|"
    r"at\s+least|at\s+most|above|below|under|over|>=|<=|>|<|=)\s*"
    r"(?P<value>[$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
    re.IGNORECASE,
)

_BETWEEN_FILTER_RE = re.compile(
    r"\b(?P<metric>revenue|income|earnings|amount|transactions?|sales|profit)?\s*"
    r"between\s+"
    r"(?P<low>[$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)"
    r"\s+and\s+"
    r"(?P<high>[$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
    re.IGNORECASE,
)


def _parse_number(text: str) -> float | None:
    """Try to extract the first meaningful number from text."""
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    unit = (m.group(2) or "").lower().strip()
    if unit in _MULTIPLIER:
        value *= _MULTIPLIER[unit]
    return value


def _extract_all_numbers(text: str) -> list[dict]:
    """Extract all numbers with surrounding context from text."""
    results = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower().strip()
        if unit in _MULTIPLIER:
            value *= _MULTIPLIER[unit]
        # Capture context: 30 chars before and after the match
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        context = text[start:end].strip()
        results.append({
            "value": value,
            "raw": m.group(0).strip(),
            "unit": unit or None,
            "context": context,
        })
    return results


# ── Entity extraction (lightweight) ──────────────────────────────────────────

_ENTITY_LABEL_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

_PROJECT_REVENUE_RE = re.compile(
    r"\b(Project\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4})"
    r"\s+generated\s+revenue\s+of\s+"
    r"([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?)"
    r"(?:\s+dollars?)?\b",
    re.IGNORECASE,
)

_INVOICE_AMOUNT_RE = re.compile(
    r"\bInvoice\s+([A-Z]{2,}-\d+)\s+amount\s+was\s+"
    r"([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?)\b",
    re.IGNORECASE,
)

_TRANSACTION_COUNT_RE = re.compile(
    r"\b(Project\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4})"
    r"\s+processed\s+exactly\s+"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s+transactions?\b",
    re.IGNORECASE,
)

_REVENUE_FLEXIBLE_PATTERNS = [
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+earned\s+([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+recorded\s+revenue\s+of\s+([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+brought\s+in\s+([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brevenue\s+of\s+([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s*:\s*([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+revenue\s*:\s*([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+had\s+revenue\s+([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})"
        r"\s+revenue\s+was\s+([$â‚¬Â£â‚¹Â¥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|k|m|b))?)",
        re.IGNORECASE,
    ),
]

_GENERIC_ENTITY_RE = re.compile(
    r"\b([A-Z]{2,}-\d+|[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5})\b"
)

_GENERIC_METRIC_WORD_RE = re.compile(
    r"\b(revenue|sales|amount|cost|profit|transactions?|count)\b",
    re.IGNORECASE,
)

_EMPLOYEE_RE = re.compile(
    r"\bEmployee\s+([A-Z]{2,}-\d+|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b"
)

_CONTRACT_RE = re.compile(r"\bContract\s+([A-Z]{1,4}-\d+)\b", re.IGNORECASE)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _extract_named_entities(text: str) -> list[dict]:
    """Extract capitalised multi-word entities from text."""
    seen: set[str] = set()
    entities: list[dict] = []
    for m in _ENTITY_LABEL_RE.finditer(text):
        name = m.group(1).strip()
        if len(name) < 2 or name.lower() in {"the", "this", "that", "from", "with"}:
            continue
        if name not in seen:
            seen.add(name)
            entities.append({"name": name, "source": "chunk_extraction"})
    return entities


# ── Core reasoning class ─────────────────────────────────────────────────────


class ReasoningAgent:
    """
    Post-retrieval reasoning agent.

    Takes a query, its classification, and retrieved chunks, then produces
    structured evidence for the generation layer.
    """

    def run(
        self,
        query: str,
        classification: QueryClassification,
        chunks: list[RetrievedChunk],
    ) -> ReasoningOutput:
        """
        Main entry point.  Dispatches to specialised reasoning methods
        based on query_type, then evaluates evidence sufficiency.
        """
        if not chunks:
            return ReasoningOutput(
                summary="No chunks retrieved — cannot reason.",
                evidence_sufficient=False,
                reasoning_steps=["No retrieval results available"],
            )

        query_type = classification.query_type
        hints = set(classification.reasoning_hints)

        logger.info(
            "Reasoning agent: type=%s chunks=%d hints=%s",
            query_type, len(chunks), hints,
        )

        # Collect structured evidence
        entities: list[dict] = []
        relationships: list[dict] = []
        calculations: list[dict] = []
        rankings: list[dict] = []
        steps: list[str] = []

        # ── Always extract entities from top chunks ───────────────────────────
        all_text = "\n".join(c.chunk for c in chunks[:10])
        entities = _extract_named_entities(all_text)
        steps.append(f"Extracted {len(entities)} entities from top {min(len(chunks), 10)} chunks")

        # ── Dispatch by query type ────────────────────────────────────────────
        if query_type == "NUMERICAL_FILTER" or "apply_numeric_filter" in hints:
            calculations, calc_steps = self._numeric_filter(query, chunks)
            steps.extend(calc_steps)

        elif query_type == "AGGREGATION" or "aggregate" in hints:
            calculations, calc_steps = self._aggregate(query, chunks)
            steps.extend(calc_steps)

        elif query_type == "COUNTING" or "count" in hints:
            calculations, calc_steps = self._count(query, chunks)
            steps.extend(calc_steps)

        elif query_type == "RANKING" or "sort_by_criterion" in hints:
            rankings, rank_steps = self._rank(query, chunks, hints)
            steps.extend(rank_steps)

        elif query_type == "COMPARISON" or "compare_attributes" in hints:
            relationships, rel_steps = self._compare(query, chunks, entities)
            steps.extend(rel_steps)

        elif query_type == "RELATIONSHIP" or "resolve_relationships" in hints:
            relationships, rel_steps = self._resolve_relationships(chunks, entities)
            steps.extend(rel_steps)

        elif query_type == "TEMPORAL" or "order_chronologically" in hints:
            rankings, rank_steps = self._temporal_order(query, chunks)
            steps.extend(rank_steps)

        elif query_type == "MULTI_HOP" or "chain_facts" in hints:
            relationships, rel_steps = self._multi_hop(query, chunks)
            steps.extend(rel_steps)

        # ── Build summary ─────────────────────────────────────────────────────
        summary = self._build_summary(
            query, query_type, entities, relationships, calculations, rankings, chunks
        )

        # ── Evidence sufficiency ──────────────────────────────────────────────
        sufficient = self._evaluate_sufficiency(
            query_type, chunks, entities, calculations, rankings
        )

        return ReasoningOutput(
            entities=entities,
            relationships=relationships,
            calculations=calculations,
            rankings=rankings,
            summary=summary,
            evidence_sufficient=sufficient,
            reasoning_steps=steps,
        )

    # ── Specialised reasoning methods ─────────────────────────────────────────

    def _number_from_match(self, raw: str) -> float | None:
        cleaned = re.sub(r"[$€£₹¥,\s]", "", raw)
        try:
            value = float(cleaned)
        except ValueError:
            return None
        return int(value) if value.is_integer() else value

    def _extract_project_revenues(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract 'Project X generated revenue of N dollars' records."""
        pattern = re.compile(
            r"\b(Project\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4})"
            r"\s+generated\s+revenue\s+of\s+"
            r"([$€£₹¥]?\s*\d[\d,]*(?:\.\d+)?)"
            r"(?:\s+dollars?)?\b",
            re.IGNORECASE,
        )
        records: list[dict] = []
        for chunk in chunks:
            for match in pattern.finditer(chunk.chunk):
                value = self._number_from_match(match.group(2))
                if value is None:
                    continue
                records.append({
                    "entity": match.group(1).strip(),
                    "entity_type": "project",
                    "metric": "revenue",
                    "value": value,
                    "chunk_document": chunk.document,
                    "chunk_page": chunk.page,
                    "confidence": 1.0,
                })
        return records

    def _extract_invoice_amounts(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract 'Invoice INV-123 amount was $N' records."""
        pattern = re.compile(
            r"\bInvoice\s+([A-Z]{2,}-\d+)\s+amount\s+was\s+"
            r"([$€£₹¥]?\s*\d[\d,]*(?:\.\d+)?)\b",
            re.IGNORECASE,
        )
        records: list[dict] = []
        for chunk in chunks:
            for match in pattern.finditer(chunk.chunk):
                value = self._number_from_match(match.group(2))
                if value is None:
                    continue
                records.append({
                    "entity": match.group(1).strip(),
                    "entity_type": "invoice",
                    "metric": "amount",
                    "value": value,
                    "chunk_document": chunk.document,
                    "chunk_page": chunk.page,
                    "confidence": 1.0,
                })
        return records

    def _extract_transaction_counts(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract 'Project X processed exactly N transactions' records."""
        pattern = re.compile(
            r"\b(Project\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4})"
            r"\s+processed\s+exactly\s+"
            r"(\d[\d,]*(?:\.\d+)?)"
            r"\s+transactions?\b",
            re.IGNORECASE,
        )
        records: list[dict] = []
        for chunk in chunks:
            for match in pattern.finditer(chunk.chunk):
                value = self._number_from_match(match.group(2))
                if value is None:
                    continue
                records.append({
                    "entity": match.group(1).strip(),
                    "entity_type": "project",
                    "metric": "transactions",
                    "value": value,
                    "chunk_document": chunk.document,
                    "chunk_page": chunk.page,
                    "confidence": 1.0,
                })
        return records

    def _extract_revenue_flexible(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract revenue records from exact and flexible revenue phrasings."""
        records = self._extract_project_revenues(chunks)
        for chunk in chunks:
            for pattern in _REVENUE_FLEXIBLE_PATTERNS:
                for match in pattern.finditer(chunk.chunk):
                    value = _parse_number(match.group(2))
                    if value is None:
                        continue
                    records.append({
                        "entity": match.group(1).strip(),
                        "entity_type": "project",
                        "metric": "revenue",
                        "value": value,
                        "chunk_document": chunk.document,
                        "chunk_page": chunk.page,
                        "confidence": 0.85,
                    })
        return self._dedupe_records(records)

    def _extract_generic_numeric_records(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Fallback numeric metric extraction for novel entity/metric phrasing."""
        records: list[dict] = []
        for chunk in chunks:
            for number in _extract_all_numbers(chunk.chunk):
                raw = number["raw"]
                context = number["context"]
                number_index = context.find(raw)
                if number_index < 0:
                    continue

                before = context[:number_index]
                entity_matches = list(_GENERIC_ENTITY_RE.finditer(before))
                metric_matches = list(_GENERIC_METRIC_WORD_RE.finditer(before))
                if not entity_matches or not metric_matches:
                    continue

                metric = metric_matches[-1].group(1).lower()
                if metric == "transaction":
                    metric = "transactions"

                records.append({
                    "entity": entity_matches[-1].group(1).strip(),
                    "entity_type": "generic",
                    "metric": metric,
                    "value": number["value"],
                    "chunk_document": chunk.document,
                    "chunk_page": chunk.page,
                    "confidence": 0.60,
                })
        return self._dedupe_records(records)

    def _dedupe_records(self, records: list[dict]) -> list[dict]:
        """Deduplicate by metric/entity, preferring larger and higher-confidence values."""
        deduped: dict[tuple[str, str], dict] = {}
        for record in records:
            key = (record["metric"], record["entity"].lower())
            current = deduped.get(key)
            if (
                current is None
                or record["value"] > current["value"]
                or (
                    record["value"] == current["value"]
                    and record.get("confidence", 0.0) > current.get("confidence", 0.0)
                )
            ):
                deduped[key] = record
        return list(deduped.values())

    def _extract_metric_records(
        self, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Merge structured metric extractors, adding generic fallback when sparse."""
        records = (
            self._extract_revenue_flexible(chunks)
            + self._extract_invoice_amounts(chunks)
            + self._extract_transaction_counts(chunks)
        )
        if len(records) < 2:
            records += self._extract_generic_numeric_records(chunks)
        return self._dedupe_records(records)

    def _requested_metric(self, query: str) -> str | None:
        q = query.lower()
        if "revenue" in q:
            return "revenue"
        if "amount" in q or "invoice" in q:
            return "amount"
        if "transaction" in q:
            return "transactions"
        return None

    def _aggregate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[dict], list[str]]:
        """Aggregate requested structured metric records."""
        metric = self._requested_metric(query)
        records = self._extract_metric_records(chunks)
        if metric:
            records = [r for r in records if r["metric"] == metric]

        steps = [f"Extracted {len(records)} metric records from {len(chunks)} chunks"]
        if not metric:
            return [], steps + ["No supported aggregation metric requested"]
        if not records:
            return [], steps + [f"No {metric} records found for aggregation"]

        total = sum(r["value"] for r in records)
        calculations = [{
            "operation": "summary_stats",
            "metric": metric,
            "sum": total,
            "average": round(total / len(records), 2),
            "min": min(r["value"] for r in records),
            "max": max(r["value"] for r in records),
            "count": len(records),
            "records": records,
        }]
        steps.append(f"Aggregation: metric={metric} sum={total} count={len(records)}")
        return calculations, steps

    def _numeric_filter(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[dict], list[str]]:
        """Apply explicit numeric comparisons over extracted metric records."""
        constraint = self._parse_numeric_constraint(query)
        records = self._extract_metric_records(chunks)
        metric = self._requested_metric(query) or (constraint or {}).get("metric")
        if metric:
            metric_aliases = {
                "income": "revenue",
                "earning": "revenue",
                "earnings": "revenue",
                "sales": "revenue",
                "transaction": "transactions",
            }
            metric = metric_aliases.get(metric, metric)
            records = [r for r in records if r["metric"] == metric]

        steps = [f"Extracted {len(records)} numeric records from {len(chunks)} chunks"]
        if not constraint:
            return [], steps + ["No numeric filter constraint found"]
        if not records:
            return [], steps + ["No matching numeric records found"]

        matched = [
            record for record in records
            if self._record_matches_constraint(record["value"], constraint)
        ]
        calculations = [{
            "operation": "numeric_filter",
            "metric": metric or "numeric_value",
            "operator": constraint["operator"],
            "threshold": constraint.get("threshold"),
            "low": constraint.get("low"),
            "high": constraint.get("high"),
            "records": matched,
            "excluded_records": [
                record for record in records
                if record not in matched
            ],
            "result": len(matched),
        }]
        steps.append(
            "Numeric filter: "
            f"{len(matched)}/{len(records)} records matched "
            f"{constraint['operator']}"
        )
        return calculations, steps

    def _parse_numeric_constraint(self, query: str) -> dict | None:
        between = _BETWEEN_FILTER_RE.search(query)
        if between:
            low = _parse_number(between.group("low"))
            high = _parse_number(between.group("high"))
            if low is None or high is None:
                return None
            return {
                "operator": "between",
                "metric": (between.group("metric") or "").lower() or None,
                "low": min(low, high),
                "high": max(low, high),
            }

        match = _NUMERIC_FILTER_RE.search(query)
        if not match:
            return None
        value = _parse_number(match.group("value"))
        if value is None:
            return None
        op = match.group("op").lower()
        operator_map = {
            "greater than": ">",
            "more than": ">",
            "above": ">",
            "over": ">",
            "less than": "<",
            "fewer than": "<",
            "below": "<",
            "under": "<",
            "at least": ">=",
            "no less than": ">=",
            "at most": "<=",
            "no more than": "<=",
        }
        return {
            "operator": operator_map.get(op, op),
            "metric": (match.group("metric") or "").lower() or None,
            "threshold": value,
        }

    def _record_matches_constraint(self, value: float, constraint: dict) -> bool:
        operator = constraint["operator"]
        if operator == "between":
            return constraint["low"] <= value <= constraint["high"]
        threshold = constraint["threshold"]
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        if operator == "=":
            return value == threshold
        return False

    def _count(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[dict], list[str]]:
        """Count requested entity types in a single pass over extracted records."""
        q = query.lower()
        records = self._extract_metric_records(chunks)
        entity_type: str | None = None
        if "project" in q:
            entity_type = "project"
        elif "invoice" in q:
            entity_type = "invoice"
        elif "employee" in q:
            entity_type = "employee"
        elif "contract" in q:
            entity_type = "contract"

        entities: set[str] = set()
        if entity_type in {"project", "invoice"}:
            entities = {
                r["entity"]
                for r in records
                if r.get("entity_type") == entity_type
            }
        elif entity_type == "employee":
            for chunk in chunks:
                entities.update(m.group(1).strip() for m in _EMPLOYEE_RE.finditer(chunk.chunk))
        elif entity_type == "contract":
            for chunk in chunks:
                entities.update(m.group(1).strip() for m in _CONTRACT_RE.finditer(chunk.chunk))

        steps = [f"Counted {len(entities)} unique {entity_type or 'matching'} entities across {len(chunks)} chunks"]
        calculations = [{
            "operation": "entity_count",
            "metric": entity_type or "entity",
            "result": len(entities),
            "entities": sorted(entities)[:100],
        }]
        return calculations, steps

    def _rank(
        self, query: str, chunks: list[RetrievedChunk], hints: set[str] | None = None
    ) -> tuple[list[dict], list[str]]:
        """Rank extracted metric records by the metric requested in the query."""
        hints = hints or set()
        metric = self._requested_metric(query)
        records = self._extract_metric_records(chunks)
        if metric:
            records = [r for r in records if r["metric"] == metric]
        records.sort(key=lambda x: x["value"], reverse=True)
        limit = 1 if "find_top_1" in hints else 20

        steps = [f"Ranked {len(records)} {metric or 'metric'} records"]
        if "find_top_1" in hints:
            steps.append("Returning top result only")
        rankings = [
            {
                "rank": i + 1,
                "label": "Top result" if "find_top_1" in hints and i == 0 else None,
                "entity": item["entity"],
                "value": item["value"],
                "metric": item["metric"],
                "chunk_document": item.get("chunk_document"),
                "chunk_page": item.get("chunk_page"),
                "confidence": item.get("confidence"),
            }
            for i, item in enumerate(records[:limit])
        ]
        return rankings, steps

    def _compare(
        self, query: str, chunks: list[RetrievedChunk], entities: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Build comparison relationships between entities mentioned in the query."""
        relationships: list[dict] = []
        entity_names = [e["name"] for e in entities[:20]]

        # Group chunks by which entities they mention
        entity_chunks: dict[str, list[str]] = {}
        for chunk in chunks:
            for name in entity_names:
                if name.lower() in chunk.chunk.lower():
                    entity_chunks.setdefault(name, []).append(chunk.chunk[:200])

        # Build comparison pairs
        matched_entities = list(entity_chunks.keys())
        for i, a in enumerate(matched_entities):
            for b in matched_entities[i + 1:]:
                relationships.append({
                    "type": "COMPARISON",
                    "entity_a": a,
                    "entity_b": b,
                    "a_evidence": entity_chunks[a][:2],
                    "b_evidence": entity_chunks[b][:2],
                })

        steps = [f"Built {len(relationships)} comparison pairs from {len(matched_entities)} entities"]
        return relationships, steps

    def _resolve_relationships(
        self, chunks: list[RetrievedChunk], entities: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Relationship extraction is handled by the financial KG LLM pipeline."""
        return [], ["Skipped co-occurrence relationship inference"]

    def _temporal_order(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[dict], list[str]]:
        """Extract date/year mentions and order chunks chronologically."""
        items: list[dict] = []

        for chunk in chunks:
            years = _YEAR_RE.findall(chunk.chunk)
            if years:
                items.append({
                    "year": int(years[0]),
                    "text": chunk.chunk[:200],
                    "document": chunk.document,
                    "page": chunk.page,
                })

        items.sort(key=lambda x: x["year"])
        rankings = [{"rank": i + 1, **item} for i, item in enumerate(items)]
        steps = [f"Found {len(items)} temporal references, ordered chronologically"]
        return rankings, steps

    def _multi_hop(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[dict], list[str]]:
        """Chain facts across multiple chunks for multi-hop reasoning."""
        chain: list[dict] = []
        all_entities: dict[str, list[dict]] = {}

        for chunk in chunks:
            entities = _extract_named_entities(chunk.chunk)
            for e in entities:
                all_entities.setdefault(e["name"], []).append({
                    "chunk": chunk.chunk[:200],
                    "document": chunk.document,
                    "page": chunk.page,
                })

        # Find entities appearing in multiple chunks (bridging entities)
        bridging = {
            name: appearances
            for name, appearances in all_entities.items()
            if len(appearances) >= 2
        }

        for name, appearances in bridging.items():
            chain.append({
                "type": "MULTI_HOP_BRIDGE",
                "entity": name,
                "appearances": len(appearances),
                "sources": appearances[:3],
            })

        steps = [
            f"Found {len(bridging)} bridging entities across chunks",
            f"Built {len(chain)} multi-hop chains",
        ]
        return chain, steps

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        query: str,
        query_type: str,
        entities: list[dict],
        relationships: list[dict],
        calculations: list[dict],
        rankings: list[dict],
        chunks: list[RetrievedChunk],
    ) -> str:
        """Build a reasoning summary to prepend to the generation context."""
        parts: list[str] = []

        parts.append(f"Query type: {query_type}")
        parts.append(f"Retrieved {len(chunks)} chunks.")

        if entities:
            names = [e["name"] for e in entities[:10]]
            parts.append(f"Key entities: {', '.join(names)}")

        if calculations:
            for calc in calculations:
                if calc.get("operation") == "summary_stats":
                    parts.append(
                        f"  {calc['metric']} summary: "
                        f"sum={calc['sum']}, average={calc['average']}, "
                        f"min={calc['min']}, max={calc['max']}, count={calc['count']}"
                    )
                    for record in calc.get("records", [])[:10]:
                        suffix = (
                            " (low confidence)"
                            if record.get("confidence", 1.0) < 0.75
                            else ""
                        )
                        parts.append(
                            f"    {record['entity']}: {record['value']}{suffix}"
                        )
                elif calc.get("operation") in ("sum", "average", "min", "max", "count", "entity_count"):
                    parts.append(f"  {calc['operation']}: {calc['result']}")

        if rankings:
            parts.append(f"Rankings ({len(rankings)} items):")
            for r in rankings[:5]:
                entity = r.get("entity", r.get("text", "")[:40])
                value = r.get("value", r.get("year", ""))
                label = f"{r['label']}: " if r.get("label") else f"#{r['rank']}: "
                suffix = " (low confidence)" if r.get("confidence", 1.0) < 0.75 else ""
                parts.append(f"  {label}{entity} = {value}{suffix}")

        if relationships:
            parts.append(f"Relationships: {len(relationships)}")

        return "\n".join(parts)

    def _evaluate_sufficiency(
        self,
        query_type: str,
        chunks: list[RetrievedChunk],
        entities: list[dict],
        calculations: list[dict],
        rankings: list[dict],
    ) -> bool:
        """Determine if the retrieved evidence is sufficient to answer the query."""
        if not chunks:
            return False

        top_score = chunks[0].relevance_score if chunks else 0.0

        # Simple heuristics by query type
        if query_type in ("LOOKUP", "ENTITY_LOOKUP", "SEMANTIC_LOOKUP"):
            return top_score >= 0.3

        if query_type in ("AGGREGATION", "COUNTING", "NUMERICAL_FILTER"):
            # Need at least some numbers/entities
            return bool(calculations) and any(
                c.get("result", c.get("count", 0)) > 0
                for c in calculations
                if c.get("operation") not in ("raw_values",)
            )

        if query_type == "RANKING":
            return len(rankings) >= 1

        if query_type in ("COMPARISON", "RELATIONSHIP"):
            return len(entities) >= 2

        if query_type == "MULTI_HOP":
            return len(chunks) >= 2 and top_score >= 0.2

        # Default: sufficient if we have reasonable retrieval
        return top_score >= 0.2 and len(chunks) >= 1
