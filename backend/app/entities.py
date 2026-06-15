"""
entities.py - LLM-based financial knowledge graph extraction.

Entity normalization, deduplication, and deterministic IDs stay local. Entity
and relation extraction come from the configured NVIDIA OpenAI-compatible LLM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.models import Chunk, EntityMention

logger = logging.getLogger(__name__)


_RAW_LOG_LIMIT = 12000


ALLOWED_ENTITY_TYPES: set[str] = {
    "COMPANY",
    "CORPORATION",
    "SUBSIDIARY",
    "INVESTOR",
    "SHAREHOLDER",
    "FOUNDER",
    "CEO",
    "EXECUTIVE",
    "BOARD_MEMBER",
    "PERSON",
    "BILLIONAIRE",
    "ENTREPRENEUR",
    "PROJECT",
    "PRODUCT",
    "BUSINESS_UNIT",
    "CONTRACT",
    "AGREEMENT",
    "PARTNERSHIP",
    "INVOICE",
    "PAYMENT",
    "TRANSACTION",
    "REVENUE",
    "PROFIT",
    "LOSS",
    "EXPENSE",
    "CASHFLOW",
    "ASSET",
    "LIABILITY",
    "EQUITY",
    "STOCK",
    "BOND",
    "FUND",
    "ETF",
    "BANK",
    "FINANCIAL_INSTITUTION",
    "DEPARTMENT",
    "EMPLOYEE",
    "COUNTRY",
    "CITY",
    "REGION",
    "QUARTER",
    "FISCAL_YEAR",
    "DATE",
    "CUSTOM_ENTITY",
}

ALLOWED_RELATION_TYPES: set[str] = {
    "OWNS",
    "INVESTED_IN",
    "FOUNDED",
    "MANAGES",
    "WORKS_FOR",
    "LEADS",
    "MEMBER_OF",
    "SIGNED",
    "SIGNED_BY",
    "PARTNERED_WITH",
    "GENERATED_REVENUE",
    "GENERATED_PROFIT",
    "INCURRED_LOSS",
    "PROCESSED_TRANSACTION",
    "ISSUED",
    "PAID",
    "RECEIVED",
    "BELONGS_TO",
    "PART_OF",
    "ACQUIRED",
    "MERGED_WITH",
    "HOLDS_STOCK_IN",
    "LOCATED_IN",
    "REPORTED_IN",
    "OCCURRED_IN",
    "RELATED_TO",
}

_SPACE_RE = re.compile(r"\s+")
_EXACT_DEDUPE_LABELS = {
    "AMOUNT",
    "CURRENCY",
    "DATE",
    "DIVIDEND",
    "FISCAL_YEAR",
    "MARKET_CAP",
    "PERCENTAGE",
    "PROFIT",
    "QUARTER",
    "REVENUE",
    "LOSS",
}

_ENTITY_PREFIX_RE = re.compile(
    r"^(?:company|corporation|corp\.?|subsidiary|investor|shareholder|"
    r"founder|ceo|executive|board\s+member|person|billionaire|entrepreneur|"
    r"project|product|business\s+unit|contract|agreement|partnership|"
    r"invoice|payment|transaction|revenue|profit|loss|expense|cashflow|"
    r"asset|liability|equity|stock|bond|fund|etf|bank|financial\s+institution|"
    r"department|employee|country|city|region|quarter|fiscal\s+year|date)\s+",
    re.IGNORECASE,
)

_KG_PROMPT = """You are a Financial Knowledge Graph Extraction System.

Stage 1: Extract entities and relations only from the predefined financial ontology.

Use ONLY the allowed entity types and relation types.

Allowed entity types:
{entity_types}

Allowed relation types:
{relation_types}

Never invent entity types.
Never invent relation types.

If a relation does not exactly match, map it to the closest allowed relation.

Normalize entity names by removing leading type words:
- "Contract CX-1001" -> "CX-1001"
- "Project Orion" -> "Orion"
- "Invoice INV-987654" -> "INV-987654"
- "Company Gamma Corp" -> "Gamma Corp"

Stage 2: If important financial entities are not covered by the ontology, include
them as CUSTOM_ENTITY. Do not create custom relation types.

Return ONLY valid JSON:

{{
  "entities":[
    {{"name":"...","type":"..."}}
  ],
  "relations":[
    {{
      "source":"...",
      "relation":"...",
      "target":"..."
    }}
  ]
}}

TEXT:
{text}
"""


class FinancialLLMExtractor:
    """Extract financial KG entities and relations with the configured NVIDIA LLM."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: OpenAI | None = None

    def extract(self, text: str) -> dict[str, list[dict[str, str]]]:
        if not text.strip() or not self._settings.nvidia_api_key:
            return {"entities": [], "relations": []}

        prompt = _KG_PROMPT.format(
            entity_types=", ".join(sorted(ALLOWED_ENTITY_TYPES)),
            relation_types=", ".join(sorted(ALLOWED_RELATION_TYPES)),
            text=text,
        )
        try:
            if self._client is None:
                self._client = OpenAI(
                    base_url=self._settings.nvidia_base_url,
                    api_key=self._settings.nvidia_api_key,
                )
            response = self._client.chat.completions.create(
                model=self._settings.nvidia_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._settings.nvidia_max_tokens,
                temperature=0,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            logger.info("Raw KG LLM response before parsing: %s", _log_text(content))
            payload = self._parse_json(content)
            sanitized = self._sanitize_payload(payload)
            logger.info(
                "KG parsed: %d entities, %d relationships",
                len(sanitized["entities"]),
                len(sanitized["relations"]),
            )
            return sanitized
        except Exception as exc:
            logger.warning("Financial KG extraction failed: %s", exc, exc_info=True)
            return {"entities": [], "relations": []}

    def _parse_json(self, content: str) -> dict[str, Any]:
        candidates = _json_candidates(content)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "KG JSON parse attempt failed at line %s column %s: %s",
                    exc.lineno,
                    exc.colno,
                    exc.msg,
                )

        fallback = _fallback_parse_payload(content)
        if fallback.get("entities") or fallback.get("relations"):
            logger.warning(
                "KG JSON repaired with fallback parser: %d raw entities, %d raw relationships",
                len(fallback.get("entities") or []),
                len(fallback.get("relations") or []),
            )
            return fallback

        if last_error:
            raise last_error
        return {}

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()
        valid_names: set[str] = set()

        for item in payload.get("entities") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            entity_type = str(item.get("type") or "").strip().upper()
            if not name or entity_type not in ALLOWED_ENTITY_TYPES:
                continue
            key = (name.lower(), entity_type)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            valid_names.add(name.lower())
            entities.append({"name": name, "type": entity_type})

        relations: list[dict[str, str]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        raw_relations = payload.get("relations") or payload.get("relationships") or []
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            relation = str(item.get("relation") or "").strip().upper()
            target = str(item.get("target") or "").strip()
            if (
                not source
                or not target
                or relation not in ALLOWED_RELATION_TYPES
                or source.lower() not in valid_names
                or target.lower() not in valid_names
            ):
                continue
            key = (source.lower(), relation, target.lower())
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append({"source": source, "relation": relation, "target": target})

        return {"entities": entities, "relations": relations}


class EntityExtractor:
    def __init__(self, llm_extractor: FinancialLLMExtractor | None = None) -> None:
        self._llm_extractor = llm_extractor or FinancialLLMExtractor()

    def enrich_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[EntityMention]]:
        all_mentions: list[EntityMention] = []
        enriched: list[Chunk] = []
        for chunk in chunks:
            try:
                payload = self._llm_extractor.extract(chunk.text)
            except Exception as exc:
                logger.warning(
                    "KG extraction failed for %s page=%s chunk=%s: %s",
                    chunk.document,
                    chunk.page,
                    chunk.chunk_index,
                    exc,
                    exc_info=True,
                )
                payload = {"entities": [], "relations": []}
            logger.info(
                "KG extracted: %d entities, %d relationships",
                len(payload.get("entities", [])),
                len(payload.get("relations", [])),
            )
            mentions = self._mentions_from_entities(
                payload.get("entities", []),
                document=chunk.document,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
            )
            all_mentions.extend(mentions)
            enriched.append(
                chunk.model_copy(
                    update={
                        "entities": sorted({m.entity_id for m in mentions}),
                        "kg_relations": payload.get("relations", []),
                    }
                )
            )
        return enriched, self.deduplicate_mentions(all_mentions)

    def extract(
        self,
        text: str,
        document: str | None = None,
        page: int | None = None,
        chunk_index: int | None = None,
    ) -> list[EntityMention]:
        payload = self._llm_extractor.extract(text)
        return self._mentions_from_entities(
            payload.get("entities", []),
            document=document,
            page=page,
            chunk_index=chunk_index,
        )

    def deduplicate_mentions(self, mentions: list[EntityMention]) -> list[EntityMention]:
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for mention in mentions:
            grouped[mention.label].append(mention)

        canonical: dict[str, str] = {}
        for label, label_mentions in grouped.items():
            norms = sorted({m.normalized for m in label_mentions})
            for norm in norms:
                key = f"{label}:{norm}"
                if label in _EXACT_DEDUPE_LABELS:
                    canonical[key] = norm
                    continue
                existing = next(
                    (
                        n
                        for n in canonical
                        if n.startswith(f"{label}:")
                        and self._similar(norm, canonical[n]) >= 0.88
                    ),
                    None,
                )
                canonical[key] = canonical[existing] if existing else norm

        deduped: list[EntityMention] = []
        for mention in mentions:
            canonical_norm = canonical.get(
                f"{mention.label}:{mention.normalized}",
                mention.normalized,
            )
            entity_id = self.entity_id(mention.label, canonical_norm)
            deduped.append(
                mention.model_copy(
                    update={"entity_id": entity_id, "normalized": canonical_norm}
                )
            )
        return self._unique_mentions(deduped)

    def entity_id(self, label: str, normalized: str) -> str:
        digest = hashlib.sha1(f"{label}:{normalized}".encode("utf-8")).hexdigest()[:12]
        return f"{label.lower()}:{digest}"

    def normalize(self, text: str, label: str) -> str:
        value = _SPACE_RE.sub(" ", text.strip())
        value = _ENTITY_PREFIX_RE.sub("", value).strip()
        if label in _EXACT_DEDUPE_LABELS:
            return value.replace(",", "").lower()
        return re.sub(r"[^\w\s&.-]", "", value.lower()).strip()

    def _mentions_from_entities(
        self,
        entities: list[dict[str, str]],
        document: str | None,
        page: int | None,
        chunk_index: int | None,
    ) -> list[EntityMention]:
        mentions = [
            self._mention(
                text=str(entity.get("name") or ""),
                label=str(entity.get("type") or ""),
                confidence=0.92,
                document=document,
                page=page,
                chunk_index=chunk_index,
            )
            for entity in entities
            if str(entity.get("name") or "").strip()
            and str(entity.get("type") or "").strip().upper() in ALLOWED_ENTITY_TYPES
        ]
        return self._unique_mentions(mentions)

    def _mention(
        self,
        text: str,
        label: str,
        confidence: float,
        document: str | None,
        page: int | None,
        chunk_index: int | None,
    ) -> EntityMention:
        label = label.upper()
        normalized = self.normalize(text, label)
        return EntityMention(
            entity_id=self.entity_id(label, normalized),
            text=text.strip(),
            label=label,
            normalized=normalized,
            confidence=confidence,
            document=document,
            page=page,
            chunk_index=chunk_index,
            start_char=None,
            end_char=None,
        )

    def _unique_mentions(self, mentions: list[EntityMention]) -> list[EntityMention]:
        seen: set[tuple[str, str, int | None, int | None]] = set()
        unique: list[EntityMention] = []
        for mention in mentions:
            key = (mention.entity_id, mention.text.lower(), mention.page, mention.chunk_index)
            if key not in seen:
                seen.add(key)
                unique.append(mention)
        return unique

    def _similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


def _log_text(text: str) -> str:
    if len(text) <= _RAW_LOG_LIMIT:
        return text
    return text[:_RAW_LOG_LIMIT] + f"... [truncated {len(text) - _RAW_LOG_LIMIT} chars]"


def _json_candidates(content: str) -> list[str]:
    text = _strip_code_fences(content.strip())
    extracted = _extract_json_object(text) or text
    candidates: list[str] = []
    for value in (text, extracted):
        for candidate in (
            value,
            _repair_json_text(value),
            _close_partial_json(_repair_json_text(value)),
        ):
            candidate = candidate.strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return text[start:]


def _repair_json_text(text: str) -> str:
    repaired = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    repaired = re.sub(r",\s*,+", ",", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([}\]])\s*([{[])", r"\1,\2", repaired)
    repaired = re.sub(
        r'("[^"]*")\s+("(?:name|type|source|relation|target|entities|relations)"\s*:)',
        r"\1, \2",
        repaired,
    )
    repaired = re.sub(
        r"(?<![\"\\])\b(entities|relations|name|type|source|relation|target)\b\s*:",
        r'"\1":',
        repaired,
    )
    return repaired


def _close_partial_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        text += '"'
    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[item] for item in reversed(stack))


def _fallback_parse_payload(content: str) -> dict[str, list[dict[str, str]]]:
    text = _strip_code_fences(content)
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    for object_text in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        values = {
            key.lower(): value.strip()
            for key, value in re.findall(
                r"""["']?(name|type|source|relation|target)["']?\s*:\s*["']([^"']+)["']""",
                object_text,
                flags=re.IGNORECASE,
            )
        }
        if values.get("name") and values.get("type"):
            entities.append({"name": values["name"], "type": values["type"]})
        if values.get("source") and values.get("relation") and values.get("target"):
            relations.append(
                {
                    "source": values["source"],
                    "relation": values["relation"],
                    "target": values["target"],
                }
            )
    return {"entities": entities, "relations": relations}
