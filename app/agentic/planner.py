#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

# provides "popular articles" based on a cached popularity threshold,
# and (optionally) asks an LLM to propose experiments.
#
# The popularity threshold is computed as:
#     max(min_read, quantile(read_count, top_quantile))
# It is stored in MongoDB (collection `metrics_cache`) and recomputed at most
# once per TTL (24h by default) via Store.get_popular_threshold(...).


from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Iterable

# Try modern OpenAI client first; fallback to legacy. Only used in propose().
try:
    from openai import OpenAI  # type: ignore
    _use_client = True
except Exception:  # pragma: no cover
    _use_client = False
    try:
        import openai  # type: ignore
    except Exception:
        openai = None  # type: ignore

# ---------------------------------------------------------------------
# Robust JSON parsing for LLM outputs
# ---------------------------------------------------------------------
def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse the LLM output as JSON.
    If it isn't clean JSON, attempt to extract the largest {...} block or an array.
    Return {} on failure.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Largest JSON object
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Array fallback → normalize to expected dict
    m2 = re.search(r"\[.*\]", text, re.S)
    if m2:
        try:
            arr = json.loads(m2.group(0))
            return {"register_wall": {"experiments": arr}, "homepage_ordering": {"experiments": []}}
        except Exception:
            pass

    return {}

# ---------------------------------------------------------------------
# Local default prompts; may be overridden via environment variables:
#   PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
# We intentionally do NOT import prompt strings from config to avoid hard import
# dependencies. This fixes ImportError in environments without those symbols.
# ---------------------------------------------------------------------
DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant that proposes controlled experiments for a news site. "
    "Respond in concise JSON with keys 'register_wall' and 'homepage_ordering', each "
    "containing an 'experiments' array."
)
DEFAULT_PLANNER_USER_PROMPT = (
    "Propose a small set of candidate experiments for the register wall and homepage ordering. "
    "Keep it simple and safe; do not require migrations."
)

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
from .config import POPULAR_MIN_READ, POPULAR_TOP_QUANTILE
from .store import Store


class Planner:
    """
    Planner for proposing experiments and providing popularity-based selections.

    - Provides "popular articles" using a cached popularity threshold in MongoDB.
    - Can (optionally) ask an LLM to propose experiments. If no API key is set,
      propose() returns an empty normalized plan without raising.
    """

    def __init__(self, store: Store, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.store = store
        self.model = model
        self.temperature = temperature

        self._api_ready = bool(os.getenv("OPENAI_API_KEY"))
        self._client = None
        if self._api_ready:
            try:
                if _use_client:
                    self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore
                else:
                    if openai is not None:  # type: ignore
                        openai.api_key = os.getenv("OPENAI_API_KEY")  # type: ignore
            except Exception:
                # Don't hard-fail; just mark not ready
                self._api_ready = False
                self._client = None

    # -------------------------------------------------------------------------
    # Threshold access (cached in MongoDB via Store)
    # -------------------------------------------------------------------------
    def popular_threshold(
        self,
        *,
        min_read: int = POPULAR_MIN_READ,
        top_quantile: float = POPULAR_TOP_QUANTILE,
        ttl_hours: int = 24,
        cache_key: str = "popular_threshold_v1",
    ) -> int:
        """
        Return the integer threshold defined as max(min_read, quantile(read_count, top_quantile)).
        The value is cached in `metrics_cache` and recomputed only if it is older than `ttl_hours`.
        """
        return self.store.get_popular_threshold(
            min_read=min_read,
            top_quantile=top_quantile,
            ttl_hours=ttl_hours,
            cache_key=cache_key,
        )

    # -------------------------------------------------------------------------
    # Public API: popular articles
    # -------------------------------------------------------------------------
    def popular_articles(
        self,
        *,
        min_read: int = POPULAR_MIN_READ,
        top_quantile: float = POPULAR_TOP_QUANTILE,
        ttl_hours: int = 24,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `limit` popular articles, sorted by read_count (desc).

        The threshold is read from the Mongo cache (recomputed at most once per
        `ttl_hours`). Only the necessary subset is fetched; no full collection
        scan is performed in the hot path.
        """
        thr = self.popular_threshold(min_read=min_read, top_quantile=top_quantile, ttl_hours=ttl_hours)

        cursor = self.store.news.find(
            {"read_count": {"$gte": int(thr)}},
            {"uuid": 1, "title": 1, "sections": 1, "read_count": 1, "published": 1},
        ).sort("read_count", -1).limit(int(limit))

        return list(cursor)

    # -------------------------------------------------------------------------
    # Helper for batch preview endpoints
    # -------------------------------------------------------------------------
    def get_read_counts(self, uuids: Iterable[str]) -> Dict[str, int]:
        """
        Return a mapping {uuid: read_count} only for the provided UUIDs.

        This is used by the batch preview endpoint to avoid N roundtrips
        and to keep the homepage preview fast.
        """
        return self.store.read_counts_for(uuids)

    # -------------------------------------------------------------------------
    # Optional: use an LLM to propose experiments
    # -------------------------------------------------------------------------
    def _normalize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the returned structure always has the expected shape:
        {
          "register_wall": {"experiments": [...]},
          "homepage_ordering": {"experiments": [...]}
        }
        """
        norm = {"register_wall": {"experiments": []}, "homepage_ordering": {"experiments": []}}
        if isinstance(plan, dict):
            for k in ("register_wall", "homepage_ordering"):
                v = plan.get(k)
                if isinstance(v, dict) and isinstance(v.get("experiments"), list):
                    norm[k]["experiments"] = v["experiments"]
                elif isinstance(v, list):
                    norm[k]["experiments"] = v
        return norm

    def propose(self) -> Dict[str, Any]:
        """
        Ask an LLM for experiments. Robust to non-JSON output.
        If no API key is configured, returns an empty normalized plan.
        """
        if not self._api_ready:
            return {"register_wall": {"experiments": []}, "homepage_ordering": {"experiments": []}}

        system_prompt = os.getenv("PLANNER_SYSTEM_PROMPT", DEFAULT_PLANNER_SYSTEM_PROMPT).strip()
        user_prompt = os.getenv("PLANNER_USER_PROMPT", DEFAULT_PLANNER_USER_PROMPT).strip()

        try:
            if _use_client and self._client is not None:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                text = (resp.choices[0].message.content or "").strip()
            elif openai is not None:  # type: ignore
                resp = openai.ChatCompletion.create(  # type: ignore
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                text = (resp["choices"][0]["message"]["content"] or "").strip()
            else:
                return {"register_wall": {"experiments": []}, "homepage_ordering": {"experiments": []}}
        except Exception:
            # Fail soft; never crash the app due to LLM issues
            return {"register_wall": {"experiments": []}, "homepage_ordering": {"experiments": []}}

        plan = _safe_json_loads(text)
        return self._normalize_plan(plan)
