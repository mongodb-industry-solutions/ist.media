#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from .util import utcnow, day_str
from .config import CONV_WINDOW_MINUTES, MOMENTUM_WINDOW_HOURS
class Rewards:
    def __init__(self, store):
        self.store = store
    def conversion_success(self, user_id: str, t_assign: datetime) -> bool:
        t_end = t_assign + timedelta(minutes=CONV_WINDOW_MINUTES)
        ev = self.store.events.find_one({
            "user_id": user_id,
            "event": "register_cta_click",
            "ts": {"$gte": t_assign, "$lte": t_end},
        })
        return bool(ev)
    def momentum_success(self, user_id: str, t_assign: datetime) -> Optional[bool]:
        d0 = day_str(t_assign)
        d2 = day_str(t_assign + timedelta(hours=MOMENTUM_WINDOW_HOURS))
        m0 = self.store.metrics_daily.find_one({"user_id": user_id, "day": d0})
        m2 = self.store.metrics_daily.find_one({"user_id": user_id, "day": d2})
        if not m0 or not m2:
            return None
        try:
            r0 = float(((m0.get("momentum") or {}).get("ema7_over_ema28")))
            r2 = float(((m2.get("momentum") or {}).get("ema7_over_ema28")))
        except Exception:
            return None
        return (r2 - r0) > 0.0
    def apply_to_arm(self, arm_doc: Dict[str, Any], conv: Optional[bool], mom: Optional[bool]) -> None:
        st = arm_doc.get("stats", {})
        conv_alpha = float(st.get("conv_alpha", 1.0)) + (1 if conv else 0)
        conv_beta  = float(st.get("conv_beta", 1.0)) + (0 if conv else 1)
        if mom is None:
            mom_alpha = float(st.get("mom_alpha", 1.0))
            mom_beta  = float(st.get("mom_beta", 1.0))
        else:
            mom_alpha = float(st.get("mom_alpha", 1.0)) + (1 if mom else 0)
            mom_beta  = float(st.get("mom_beta", 1.0)) + (0 if mom else 1)
        assignments = int(st.get("assignments", 0)) + 1
        conversions = int(st.get("conversions", 0)) + (1 if conv else 0)
        momentum_successes = int(st.get("momentum_successes", 0)) + (1 if mom else 0)
        self.store.arms.update_one({"_id": arm_doc["_id"]}, {"$set": {
            "stats": {
                "conv_alpha": conv_alpha, "conv_beta": conv_beta,
                "mom_alpha": mom_alpha,   "mom_beta": mom_beta,
                "assignments": assignments,
                "conversions": conversions,
                "momentum_successes": momentum_successes,
            }
        }})
    def resolve_pending(self) -> int:
        count = 0
        cur = self.store.assignments.find({"resolved": {"$ne": True}}).limit(500)
        for a in cur:
            conv = None; mom = None
            if a.get("experiment_class") == "register_wall":
                conv = self.conversion_success(a["user_id"], a["ts_assigned"])
            elif a.get("experiment_class") == "homepage_ordering":
                mom = self.momentum_success(a["user_id"], a["ts_assigned"])
            self.store.assignments.update_one({"_id": a["_id"]}, {"$set": {
                "conversion_success": conv,
                "momentum_success": mom,
                "resolved": True,
                "ts_resolved": utcnow(),
            }})
            arm = self.store.arms.find_one({"experiment_id": a["experiment_id"], "arm_id": a["arm_id"]})
            if arm:
                self.apply_to_arm(arm, conv, mom)
            count += 1
        if count:
            self.recompute_rollups(since_days=30)
        return count
    def recompute_rollups(self, since_days: int = 14) -> None:
        start = utcnow() - timedelta(days=since_days)
        start_day = start.date().isoformat()
        self.store.campaign_ts.delete_many({"day": {"$gte": start_day}})
        pipeline = [
            {"$match": {"ts_assigned": {"$gte": start}}},
            {"$project": {
                "experiment_id": 1,
                "arm_id": 1,
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts_assigned"}},
                "conversion_success": 1,
                "momentum_success": 1,
            }},
            {"$group": {
                "_id": {"experiment_id": "$experiment_id", "arm_id": "$arm_id", "day": "$day"},
                "assignments": {"$sum": 1},
                "conversions": {"$sum": {"$cond": ["$conversion_success", 1, 0]}},
                "mom_success": {"$sum": {"$cond": ["$momentum_success", 1, 0]}},
            }},
        ]
        rows = list(self.store.assignments.aggregate(pipeline))
        for r in rows:
            eid = r["_id"]["experiment_id"]; arm = r["_id"]["arm_id"]; day = r["_id"]["day"]
            a = int(r["assignments"]) or 1
            self.store.campaign_ts.update_one(
                {"experiment_id": eid, "arm_id": arm, "day": day},
                {"$set": {
                    "experiment_id": eid,
                    "arm_id": arm,
                    "day": day,
                    "assignments": a,
                    "conversion_rate": r["conversions"] / a,
                    "momentum_success_rate": r["mom_success"] / a,
                    "ts_updated": utcnow(),
                }},
                upsert=True,
            )
