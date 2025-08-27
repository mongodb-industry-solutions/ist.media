#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import json, os
from typing import Any, Dict, List
try:
    from openai import OpenAI  # type: ignore
    _use_client = True
except Exception:  # pragma: no cover
    import openai  # type: ignore
    _use_client = False
from .config import POPULAR_MIN_READ, POPULAR_TOP_QUANTILE
from .util import utcnow
PLANNER_SYSTEM = (
    "You are a product optimization planner for a media site. "
    "Propose experiments & arms to improve register CTA clicks and momentum. "
    "Output strictly JSON."
)
class Planner:
    def __init__(self, store, model: str = "gpt-4o-mini", temperature: float = 0.3):
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
    def popular_articles(self, min_read: int = POPULAR_MIN_READ, top_quantile: float = POPULAR_TOP_QUANTILE) -> List[Dict[str, Any]]:
        docs = list(self.store.news.find({}, {"uuid": 1, "title": 1, "sections": 1, "read_count": 1, "published": 1}).limit(5000))
        reads = sorted(int(d.get("read_count", 0)) for d in docs) if docs else []
        qthr = reads[max(0, int(len(reads) * top_quantile) - 1)] if reads else 0
        thr = max(min_read, qthr)
        popular = [d for d in docs if int(d.get("read_count", 0)) >= thr]
        return sorted(popular, key=lambda d: int(d.get("read_count", 0)), reverse=True)[:200]
    def propose(self) -> Dict[str, Any]:
        popular = self.popular_articles()
        payload = {
            "objective": {
                "conversion": "register_cta_click within 5m of assignment",
                "momentum": "ema7_over_ema28 increases within 48h"
            },
            "popular_articles_sample": [
                {"uuid": d.get("uuid"), "sections": d.get("sections", []), "read_count": d.get("read_count", 0)}
                for d in popular[:50]
            ],
            "ask": (
                "Propose new experiment instances for classes 'register_wall' and 'homepage_ordering'. "
                "For 'register_wall', suggest arms keyed by article category with concise copy. "
                "For 'homepage_ordering', suggest arms specifying strategy ('time','popular','interest_boosted') "
                "and parameters (decay_days, exclude_read). Provide 1-3 audience_filters where useful."
            )
        }
        if _use_client:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": json.dumps(payload)}],
                temperature=self.temperature,
            )
            text = resp.choices[0].message.content.strip()
        else:
            import openai  # type: ignore
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": json.dumps(payload)}],
                temperature=self.temperature,
            )
            text = resp["choices"][0]["message"]["content"].strip()
        return json.loads(text)
