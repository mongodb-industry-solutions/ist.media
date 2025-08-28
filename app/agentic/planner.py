#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional

# Try modern OpenAI client first; fallback to legacy
try:
    from openai import OpenAI  # type: ignore
    _use_client = True
except Exception:  # pragma: no cover
    import openai  # type: ignore
    _use_client = False

from .config import POPULAR_MIN_READ, POPULAR_TOP_QUANTILE
from .util import utcnow

# System prompt the LLM sees
PLANNER_SYSTEM = (
    "You are a product optimization planner for a media site. "
    "Propose experiments & arms to improve register CTA clicks and momentum. "
    "Output strictly JSON with top-level keys 'register_wall' and 'homepage_ordering'."
)

def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse the LLM output as JSON.
    If it isn't clean JSON, attempt to extract the largest {...} block.
    Return {} on failure.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # First try: direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: extract a JSON object substring
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Third try: sometimes models wrap arrays
    m2 = re.search(r"\[.*\]", text, re.S)
    if m2:
        try:
            arr = json.loads(m2.group(0))
            # If it's an array of experiments, normalize to our schema
            return {
                "register_wall": {"experiments": arr},
                "homepage_ordering": {"experiments": []},
            }
        except Exception:
            pass

    # Give up
    print("[planner] Failed to parse LLM output:", text[:400])
    return {}


class Planner:
    """
    LLM-driven planner.
    - Gathers a popularity sample from MongoDB.
    - Asks the LLM to propose experiments and arms.
    - Returns a dict consumable by Experiments.create_from_plan().
    """

    def __init__(self, store, model: str = "gpt-4o-mini", temperature: float = 0.3):
        # LLM REQUIRED (per your requirement)
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required — planner is LLM-only by design.")

        self.store = store
        self.model = model
        self.temperature = temperature

        if _use_client:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            import openai  # type: ignore
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = None

    def popular_articles(
        self,
        min_read: int = POPULAR_MIN_READ,
        top_quantile: float = POPULAR_TOP_QUANTILE,
    ) -> List[Dict[str, Any]]:
        """
        Return a trimmed list of 'popular' articles for the LLM to consider.
        Uses a dynamic quantile AND a static floor.
        """
        docs = list(
            self.store.news.find(
                {},
                {"uuid": 1, "title": 1, "sections": 1, "read_count": 1, "published": 1},
            ).limit(5000)
        )
        reads = sorted(int(d.get("read_count", 0)) for d in docs) if docs else []
        qthr = reads[max(0, int(len(reads) * top_quantile) - 1)] if reads else 0
        thr = max(min_read, qthr)
        popular = [d for d in docs if int(d.get("read_count", 0)) >= thr]
        # Return up to 200, most-read first
        return sorted(popular, key=lambda d: int(d.get("read_count", 0)), reverse=True)[:200]

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
                    # Allow direct list fallback
                    norm[k]["experiments"] = v
        return norm

    def propose(self) -> Dict[str, Any]:
        """
        Ask the LLM for experiments. Robust to non-JSON output.
        Returns normalized plan (never raises JSON decode).
        """
        popular = self.popular_articles()
        payload = {
            "objective": {
                "conversion": "register_cta_click within 5m of assignment",
                "momentum": "ema7_over_ema28 increases within 48h",
            },
            "popular_articles_sample": [
                {
                    "uuid": d.get("uuid"),
                    "sections": d.get("sections", []),
                    "read_count": d.get("read_count", 0),
                }
                for d in popular[:50]
            ],
            "ask": (
                "Propose new experiment instances for classes 'register_wall' and 'homepage_ordering'. "
                "For 'register_wall', suggest arms keyed by article category with concise copy. "
                "For 'homepage_ordering', suggest arms specifying strategy ('time','popular','interest_boosted') "
                "and parameters (decay_days, exclude_read). Provide 1-3 audience_filters where useful. "
                "Return STRICT JSON with keys 'register_wall' and 'homepage_ordering', each containing "
                "{\"experiments\": [ ... ]}. Do NOT include any commentary."
            ),
        }

        # ---- Call LLM ----
        if _use_client:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            import openai  # type: ignore
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=self.temperature,
            )
            text = (resp["choices"][0]["message"]["content"] or "").strip()

        # ---- Parse robustly ----
        plan = _safe_json_loads(text)
        norm = self._normalize_plan(plan)

        # Optional: quick sanity guard; if both are empty, log the raw snippet
        if not norm["register_wall"]["experiments"] and not norm["homepage_ordering"]["experiments"]:
            print("[planner] Empty plan after parsing; snippet:", text[:200])

        return norm
