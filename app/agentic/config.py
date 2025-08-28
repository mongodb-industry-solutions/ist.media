#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import os

DB_NAME = "banditdb"  # fixed
CONV_WINDOW_MINUTES = 5
MOMENTUM_WINDOW_HOURS = 48
DEFAULT_WEIGHTS = (0.7, 0.3)
MAX_HOMEPAGE_RESULTS = 16
ARTICLE_AGE_MAX_DAYS = 30
DEFAULT_DECAY_DAYS = 7
POPULAR_MIN_READ = 3
POPULAR_TOP_QUANTILE = 0.8
MAX_ACTIVE_EXPERIMENTS_PER_CLASS = 5

PLANNER_INTERVAL_SEC = int(os.getenv("PLANNER_INTERVAL_SEC", "600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

COLL = {
    "experiments": "experiments",
    "arms": "arms",
    "assignments": "assignments",
    "events": "events",
    "metrics_daily": "metrics_daily",
    "campaign_ts": "campaign_ts",
    "news": "news",
    "agent_logs": "agent_logs",
    "agent_jobs": "agent_jobs",
}
