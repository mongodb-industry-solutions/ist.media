#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import os
from .store import Store
def run(user_id: str = "user123"):
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI first")
    store = Store(os.getenv("MONGODB_URI"))
    now = datetime.now(timezone.utc)
    arts = []
    sections = [["Politics"],["Health"],["Sports"],["Tech"],["Culture"],["World"],["Business"]]
    for i in range(30):
        arts.append({
            "uuid": str(uuid4()),
            "title": f"Demo Article {i}",
            "sections": sections[i % len(sections)],
            "read_count": 1 + (i * 3) % 50,
            "published": (now - timedelta(days=i % 20)).isoformat(),
        })
    store.news.delete_many({})
    store.news.insert_many(arts)
    store.metrics_daily.delete_many({"user_id": user_id})
    for d in range(0, 5):
        day = (now - timedelta(days=4 - d)).date().isoformat()
        store.metrics_daily.insert_one({
            "user_id": user_id,
            "day": day,
            "ema": {"ema3": 5 + d, "ema7": 7 + d, "ema28": 18},
            "momentum": {"ema7_over_ema28": (7 + d)/18.0},
        })
    print("Seeded news + metrics_daily. Example user:", user_id)
