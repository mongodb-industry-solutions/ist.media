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
        resolved = self.rewards.resolve_pending()
        if resolved:
            self.bb.log(self.name, "resolved", {"count": resolved})
        return resolved
