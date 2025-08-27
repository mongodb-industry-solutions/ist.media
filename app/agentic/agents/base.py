#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Dict, List
class Agent:
    name = "agent"
    def __init__(self, store, blackboard):
        self.store = store
        self.bb = blackboard
    def tick(self): pass
    def handle_job(self, job): pass
class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
    def register(self, agent: Agent):
        self.agents[agent.name] = agent
    def get(self, name: str) -> Agent:
        return self.agents[name]
    def all(self) -> List[Agent]:
        return list(self.agents.values())
