#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from .base import Agent
class PlannerAgent(Agent):
    name = "planner"
    def __init__(self, store, blackboard, planner, experiments_api):
        super().__init__(store, blackboard)
        self.planner = planner
        self.experiments_api = experiments_api
    def tick(self):
        plan = self.planner.propose()
        self.bb.log(self.name, "proposed", {"keys": list(plan.keys())})
        created_total = 0
        for ex_class in ("register_wall", "homepage_ordering"):
            if plan.get(ex_class):
                ids = self.experiments_api.create_from_plan(ex_class, plan.get(ex_class))
                created_total += len(ids)
                self.bb.log(self.name, "applied", {"experiment_class": ex_class, "created_count": len(ids)})
        return created_total
