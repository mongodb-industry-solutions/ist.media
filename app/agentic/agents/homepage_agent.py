#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .base import Agent
from ..bandit import ThompsonScalarized
from ..strategies import score_time, score_popular, score_interest_boosted
from ..util import utcnow
from datetime import timedelta
from ..config import ARTICLE_AGE_MAX_DAYS, DEFAULT_DECAY_DAYS, DEFAULT_WEIGHTS, MAX_HOMEPAGE_RESULTS
class HomepageAgent(Agent):
    name = "homepage"
    def __init__(self, store, blackboard, experiments_api):
        super().__init__(store, blackboard)
        self.experiments_api = experiments_api
    def decide(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ex = self.experiments_api.pick_experiment("homepage_ordering")
        if not ex:
            return {"articles": []}
        weights = tuple(ex.get("parameters", {}).get("weights", list(DEFAULT_WEIGHTS)))
        policy = ThompsonScalarized(self.store.arms, ex, weights)
        arm_id, _ = policy.choose_arm(context)
        arm = self.store.arms.find_one({"experiment_id": ex["_id"], "arm_id": arm_id}) or {}
        payload = arm.get("payload", {})
        read_uuids = set(ev.get("article_id") for ev in self.store.events.find({"user_id": user_id, "event": "read_to_end"}, {"article_id": 1}))
        min_published = (utcnow() - timedelta(days=ARTICLE_AGE_MAX_DAYS)).isoformat()
        candidates = list(self.store.news.find({"published": {"$gte": min_published}}, {"uuid": 1, "title": 1, "sections": 1, "read_count": 1, "published": 1}).limit(2000))
        candidates = [c for c in candidates if c.get("uuid") not in read_uuids]
        sec_counts: Dict[str, int] = {}
        for ev in self.store.events.find({"user_id": user_id, "event": "read_to_end"}, {"article_id": 1}):
            art = self.store.news.find_one({"uuid": ev.get("article_id")}, {"sections": 1})
            for s in (art or {}).get("sections", []) or []:
                sec_counts[s] = sec_counts.get(s, 0) + 1
        user_top_sections = [s for s, _ in sorted(sec_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        strat = payload.get("strategy", "popular")
        decay_days = int(payload.get("decay_days", DEFAULT_DECAY_DAYS))
        results: List[Tuple[float, Dict[str, Any]]] = []
        for a in candidates:
            if strat == "time": sc = score_time(a)
            elif strat == "interest_boosted": sc = score_interest_boosted(a, user_top_sections=user_top_sections, decay_days=decay_days)
            else: sc = score_popular(a, decay_days=decay_days)
            results.append((sc, a))
        results.sort(key=lambda x: x[0], reverse=True)
        top = [{"uuid": a["uuid"], "title": a.get("title") or ""} for _, a in results[:MAX_HOMEPAGE_RESULTS]]
        assign = {
            "experiment_id": ex["_id"],
            "experiment_class": "homepage_ordering",
            "user_id": user_id,
            "arm_id": arm_id,
            "context": {"user_top_sections": user_top_sections, **context},
            "ts_assigned": utcnow(),
            "resolved": False,
        }
        self.store.assignments.insert_one(assign)
        self.bb.log(self.name, "assigned", {"user_id": user_id, "experiment_id": ex.get("experiment_id"), "arm_id": arm_id, "articles_count": len(top)})
        return {"experiment_id": ex.get("experiment_id"), "experiment_class": "homepage_ordering", "ordering_strategy": arm_id, "parameters": {"decay_days": decay_days, "exclude_read": True}, "articles": top}
