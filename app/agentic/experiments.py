#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Any, Dict, List, Optional
from bson import ObjectId
from .util import utcnow
from .bandit import ArmStats
from .config import MAX_ACTIVE_EXPERIMENTS_PER_CLASS

class Experiments:
    def __init__(self, store):
        self.store = store
    def _ensure_id(self) -> str:
        return f"exp_{ObjectId()}"

    def count_active(self, experiment_class: str) -> int:
        return self.store.experiments.count_documents({
            "experiment_class": experiment_class, "status": "active"
        })

    def prune_active(self, experiment_class: str, max_active: int, make_room_for: int = 1) -> int:
        if max_active <= 0:
            return 0
        active_total = self.count_active(experiment_class)
        need_free = max(0, (active_total + make_room_for) - max_active)
        if need_free == 0:
            return 0
        to_archive = list(self.store.experiments.find(
            {"experiment_class": experiment_class, "status": "active"},
            {"_id": 1}
        ).sort("ts_updated", 1).limit(need_free))
        if not to_archive:
            return 0
        ids = [d["_id"] for d in to_archive]
        self.store.experiments.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"status": "archived", "ts_updated": utcnow(), "archived_reason": "capacity"}}
        )
        return len(ids)

    def create_from_plan(self, experiment_class: str, plan: Dict[str, Any]) -> List[ObjectId]:
        created: List[ObjectId] = []
        ex_list = plan.get("experiments", []) or []
        if not ex_list:
            return created

        self.prune_active(
            experiment_class=experiment_class,
            max_active=MAX_ACTIVE_EXPERIMENTS_PER_CLASS,
            make_room_for=len(ex_list)
        )

        for ex in ex_list:
            doc = {
                "experiment_id": self._ensure_id(),
                "experiment_class": experiment_class,
                "parameters": ex.get("parameters", {"weights": [0.7, 0.3]}),
                "targeting": ex.get("targeting", {}),
                "arm_ids": [],
                "status": "active",
                "origin": "llm_autospawn",
                "ts_created": utcnow(),
                "ts_updated": utcnow(),
            }
            self.store.experiments.insert_one(doc)
            exp_doc = self.store.experiments.find_one({"experiment_id": doc["experiment_id"]})
            if not exp_doc:
                continue

            arm_ids: List[str] = []
            for a in ex.get("arms", []) or []:
                arm_ids.append(a.get("id"))
                self.store.arms.update_one(
                    {"experiment_id": exp_doc["_id"], "arm_id": a.get("id")},
                    {"$set": {
                        "experiment_id": exp_doc["_id"],
                        "arm_id": a.get("id"),
                        "label": a.get("label", a.get("id")),
                        "kind": a.get("kind", "register_wall_copy"),
                        "payload": a.get("payload", {}),
                        "category": a.get("category"),
                        "status": a.get("status", "active"),
                        "stats": ArmStats().__dict__,
                        "ts_updated": utcnow(),
                    }},
                    upsert=True,
                )
            self.store.experiments.update_one(
                {"_id": exp_doc["_id"]},
                {"$set": {"arm_ids": arm_ids, "ts_updated": utcnow()}}
            )
            created.append(exp_doc["_id"])

        self.prune_active(
            experiment_class=experiment_class,
            max_active=MAX_ACTIVE_EXPERIMENTS_PER_CLASS,
            make_room_for=0
        )

        return created

    def pick_experiment(self, experiment_class: str):
        return self.store.experiments.find_one(
            {"experiment_class": experiment_class, "status": "active"},
            sort=[("ts_updated", -1)]
        )
