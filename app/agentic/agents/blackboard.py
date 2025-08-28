#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Dict, Any
from ..util import utcnow
from ..config import COLL
import json, logging
logger = logging.getLogger("bandit")

class Blackboard:
    def __init__(self, store):
        self.store = store
        self.logs = store.db[COLL["agent_logs"]]
        self.jobs = store.db[COLL["agent_jobs"]]

    def log(self, agent: str, action: str, payload: Dict[str, Any]):
        doc = {"agent": agent, "action": action, "payload": payload, "details": payload, "ts": utcnow()}
        self.logs.insert_one(doc)
        logger.info(json.dumps({"agent": agent, "action": action, **payload}))

    def enqueue(self, agent: str, task: str, payload: Dict[str, Any]):
        job = {"agent": agent, "task": task, "payload": payload, "ts": utcnow(), "run_at": utcnow(), "status": "queued"}
        self.jobs.insert_one(job)
        self.log(agent, "job.enqueued", {"task": task, "payload": payload})
