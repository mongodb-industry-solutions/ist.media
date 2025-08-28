#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Dict, Any
from .base import Agent
from ..bandit import ThompsonScalarized
from ..util import utcnow
from ..config import DEFAULT_WEIGHTS

class RegisterWallAgent(Agent):
    name = "register_wall"

    def __init__(self, store, blackboard, experiments_api):
        super().__init__(store, blackboard)
        self.experiments_api = experiments_api

    def decide(self, user_id: str, article_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ex = self.experiments_api.pick_experiment("register_wall")
        if not ex:
            return {"display_register_wall": False}
        weights = tuple(ex.get("parameters", {}).get("weights", list(DEFAULT_WEIGHTS)))
        policy = ThompsonScalarized(self.store.arms, ex, weights)
        arm_id, _ = policy.choose_arm(context)
        assign = {
            "experiment_id": ex["_id"],
            "experiment_class": "register_wall",
            "user_id": user_id,
            "arm_id": arm_id,
            "context": context,
            "ts_assigned": utcnow(),
            "resolved": False,
        }
        self.store.assignments.insert_one(assign)

        arm_doc = self.store.arms.find_one({"experiment_id": ex["_id"], "arm_id": arm_id}) or {}
        self.bb.log(self.name, "assigned", {
            "user_id": user_id,
            "experiment_id": ex.get("experiment_id"),
            "arm_id": arm_id,
            "arm_label": arm_doc.get("label", arm_id),
            "article_id": article_id,
            "copy_hint": (arm_doc.get("payload") or {}).get("copy") or (arm_doc.get("payload") or {}).get("headline"),
        })

        return {"display_register_wall": True, "arm_id": arm_id, "experiment_id": ex.get("experiment_id")}
