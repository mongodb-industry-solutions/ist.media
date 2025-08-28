#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from .base import Agent

class AnalyticsAgent(Agent):
    name = "analytics"

    def __init__(self, store, blackboard, rewards):
        super().__init__(store, blackboard)
        self.rewards = rewards
        self._last_rollup = 0

    def tick(self):
        self._last_rollup = (self._last_rollup + 1) % 5
        if self._last_rollup == 0:
            rows = self.rewards.recompute_rollups(since_days=30)
            self.bb.log(self.name, "rollup", {"since_days": 30, "rows": rows})
