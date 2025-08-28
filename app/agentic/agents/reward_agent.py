#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from .base import Agent

class RewardAgent(Agent):
    name = "rewards"

    def __init__(self, store, blackboard, rewards):
        super().__init__(store, blackboard)
        self.rewards = rewards

    def tick(self):
        summary = self.rewards.resolve_pending()
        if summary.get("total_resolved", 0):
            self.bb.log(self.name, "resolved", summary)
        return summary.get("total_resolved", 0)
