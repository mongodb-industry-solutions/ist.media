#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from .config import DB_NAME, COLL
class Store:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client[DB_NAME]
        self.experiments: Collection = self.db[COLL["experiments"]]
        self.arms: Collection = self.db[COLL["arms"]]
        self.assignments: Collection = self.db[COLL["assignments"]]
        self.events: Collection = self.db[COLL["events"]]
        self.metrics_daily: Collection = self.db[COLL["metrics_daily"]]
        self.campaign_ts: Collection = self.db[COLL["campaign_ts"]]
        self.news: Collection = self.db[COLL["news"]]
        self.agent_logs: Collection = self.db[COLL["agent_logs"]]
        self.agent_jobs: Collection = self.db[COLL["agent_jobs"]]
    def init_indexes(self) -> None:
        self.experiments.create_index([("experiment_id", ASCENDING)], unique=True)
        self.experiments.create_index([("experiment_class", ASCENDING), ("status", ASCENDING)])
        self.arms.create_index([("experiment_id", ASCENDING), ("arm_id", ASCENDING)], unique=True)
        self.assignments.create_index([("experiment_id", ASCENDING), ("user_id", ASCENDING), ("ts_assigned", DESCENDING)])
        self.assignments.create_index([("resolved", ASCENDING), ("ts_assigned", ASCENDING)])
        self.events.create_index([("user_id", ASCENDING), ("event", ASCENDING), ("ts", DESCENDING)])
        self.metrics_daily.create_index([("user_id", ASCENDING), ("day", ASCENDING)], unique=True)
        self.campaign_ts.create_index([("experiment_id", ASCENDING), ("day", ASCENDING), ("arm_id", ASCENDING)], unique=True)
        self.news.create_index([("uuid", ASCENDING)], unique=True)
        self.news.create_index([("read_count", DESCENDING)])
        self.agent_logs.create_index([("ts", DESCENDING)])
        self.agent_jobs.create_index([("run_at", ASCENDING)])
