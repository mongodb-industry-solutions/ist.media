#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Iterable, Dict, Any
from datetime import datetime, timedelta

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection, ReturnDocument

from .config import DB_NAME, NEWS_DB_NAME, COLL

class Store:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client[DB_NAME]
        self.news_db = self.client[NEWS_DB_NAME]

        self.experiments: Collection   = self.db[COLL["experiments"]]
        self.arms: Collection          = self.db[COLL["arms"]]
        self.assignments: Collection   = self.db[COLL["assignments"]]
        self.events: Collection        = self.db[COLL["events"]]
        self.metrics_daily: Collection = self.db[COLL["metrics_daily"]]
        self.campaign_ts: Collection   = self.db[COLL["campaign_ts"]]
        self.agent_logs: Collection    = self.db[COLL["agent_logs"]]
        self.agent_jobs: Collection    = self.db[COLL["agent_jobs"]]
        self.metrics_cache: Collection = self.db[COLL.get("metrics_cache", "metrics_cache")]

        self.news: Collection          = self.news_db[COLL["news"]]

    def init_indexes(self) -> None:
        self.experiments.create_index([("experiment_id", ASCENDING)], unique=True)
        self.experiments.create_index([("experiment_class", ASCENDING), ("status", ASCENDING)])
        self.arms.create_index([("experiment_id", ASCENDING), ("arm_id", ASCENDING)], unique=True)
        self.assignments.create_index([("experiment_id", ASCENDING), ("user_id", ASCENDING), ("ts_assigned", DESCENDING)])
        self.assignments.create_index([("resolved", ASCENDING), ("ts_assigned", ASCENDING)])
        self.events.create_index([("user_id", ASCENDING), ("event", ASCENDING), ("ts", DESCENDING)])
        self.metrics_daily.create_index([("user_id", ASCENDING), ("day", ASCENDING)], unique=True)
        self.campaign_ts.create_index([("experiment_id", ASCENDING), ("day", ASCENDING), ("arm_id", ASCENDING)], unique=True)
        self.agent_logs.create_index([("ts", DESCENDING)])
        self.agent_jobs.create_index([("run_at", ASCENDING)])
        self.metrics_cache.create_index([("updated_at", DESCENDING)])


    def get_popular_threshold(
        self, *, min_read: int, top_quantile: float, ttl_hours: int = 24, cache_key: str = "popular_threshold_v1"
    ) -> int:
        """
        Return the integer threshold defined as max(min_read, quantile(read_count, top_quantile)).
        The value is cached in `metrics_cache` and recomputed only if it is older than `ttl_hours`.
        """
        now = datetime.utcnow()
        stale_before = now - timedelta(hours=ttl_hours)

        doc = self.metrics_cache.find_one({"_id": cache_key})
        if doc and doc.get("updated_at") and doc["updated_at"] > stale_before:
            return int(doc.get("thr", 0))

        # recompute (prefer Mongo $percentile; fallback Python)
        qthr, n_docs = 0.0, 0
        try:
            pipeline = [
                # Treat missing read_count as 0 and force numeric type
                {"$project": {
                    "rc": {"$toInt": {"$ifNull": ["$read_count", 0]}},
                }},
                {"$group": {
                    "_id": None,
                    "n": {"$sum": 1},
                    "p": {"$percentile": {
                        "input": "$rc",
                        "p": [float(top_quantile)],   # or the constant you use
                        "method": "approximate",
                    }},
                }},
            ]
            agg = list(self.news.aggregate(pipeline, allowDiskUse=True))
            if agg:
                n_docs = int(agg[0].get("n", 0))
                ps = agg[0].get("p") or [0.0]
                qthr = float(ps[0] if isinstance(ps, (list, tuple)) else ps)
        except Exception:
            counts = [int(d.get("read_count", 0)) for d in self.news.find({}, {"read_count": 1})]
            n_docs = len(counts)
            if n_docs:
                counts.sort()
                k = max(0, min(n_docs - 1, int(round(top_quantile * (n_docs - 1)))))
                qthr = float(counts[k])

        thr = max(int(min_read), int(qthr))

        self.metrics_cache.find_one_and_update(
            {"_id": cache_key},
            {"$set": {
                "thr": int(thr),
                "qthr": float(qthr),
                "n_docs": int(n_docs),
                "min_read": int(min_read),
                "top_quantile": float(top_quantile),
                "updated_at": now,
            }},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        print(f"[popular-threshold] recomputed n={n_docs} q={top_quantile} qthr={qthr:.2f} thr={thr} at {now.isoformat()}Z")
        return int(thr)

    # Hilfsfunktion für Batch-Preview etc.
    def read_counts_for(self, uuids: Iterable[str]) -> Dict[str, int]:
        m: Dict[str, int] = {}
        if not uuids:
            return m
        for d in self.news.find({"uuid": {"$in": list(uuids)}}, {"uuid": 1, "read_count": 1}):
            m[str(d.get("uuid"))] = int(d.get("read_count", 0))
        return m
