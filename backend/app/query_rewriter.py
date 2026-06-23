"""
query_rewriter.py - lightweight query optimization before classification.

The default path is deterministic and fast. An optional LLM rewrite can be
enabled from config, but the heuristic rewriter is intentionally conservative:
it clarifies retrieval intent without adding facts.
"""
from __future__ import annotations

import logging
import re

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_DOC_HINT_RE = re.compile(
    r"\b(document|pdf|file|page|section|clause|contract|report|agreement|policy)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^\s*(what|who|when|where|why|how|which|list|show|find|identify|compare|summarize)\b",
    re.IGNORECASE,
)


class QueryRewriter:
    """Rewrite short or ambiguous user text into retrieval-friendly queries."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def rewrite(self, query: str) -> str:
        original = re.sub(r"\s+", " ", query.strip())
        if not original:
            return original

        if self._settings.query_rewriter_llm_enabled:
            rewritten = self._rewrite_with_llm(original)
            if rewritten:
                return rewritten

        return self._rewrite_with_rules(original)

    def _rewrite_with_rules(self, query: str) -> str:
        cleaned = query.strip()
        words = cleaned.split()

        if len(words) <= 3 and not cleaned.endswith("?"):
            return f"Find document evidence about {cleaned}."

        if not _QUESTION_START_RE.search(cleaned) and not cleaned.endswith("?"):
            return f"Find document evidence related to: {cleaned}"

        if not _DOC_HINT_RE.search(cleaned):
            return f"{cleaned} Answer using the uploaded document evidence."

        return cleaned

    def _rewrite_with_llm(self, query: str) -> str | None:
        try:
            client = OpenAI(
                base_url=self._settings.nvidia_base_url,
                api_key=self._settings.nvidia_api_key,
            )
            resp = client.chat.completions.create(
                model=self._settings.nvidia_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user query for document retrieval. "
                            "Preserve meaning exactly. Do not answer. "
                            "Do not add facts. Return only the rewritten query."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            rewritten = (resp.choices[0].message.content or "").strip().strip('"')
            if rewritten and len(rewritten) <= 500:
                return re.sub(r"\s+", " ", rewritten)
        except Exception as exc:
            logger.warning("LLM query rewrite failed; using heuristic rewrite: %s", exc)
        return None
