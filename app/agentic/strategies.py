#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import math
from datetime import datetime
from typing import Any, Dict, List
from .util import utcnow
from .config import DEFAULT_DECAY_DAYS
def _age_days(published_iso: str) -> float:
    try:
        d = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
    except Exception:
        return 9999.0
    return max(0.0, (utcnow() - d).total_seconds() / 86400.0)
def score_time(article: Dict[str, Any], **kw) -> float:
    return -_age_days(article.get("published", utcnow().isoformat()))
def score_popular(article: Dict[str, Any], decay_days: int = DEFAULT_DECAY_DAYS, **kw) -> float:
    reads = float(article.get("read_count", 0))
    age = _age_days(article.get("published", utcnow().isoformat()))
    lam = 1.0 / max(decay_days, 1)
    return reads * math.exp(-lam * age)
def score_interest_boosted(article: Dict[str, Any], user_top_sections: List[str], decay_days: int = DEFAULT_DECAY_DAYS, **kw) -> float:
    base = score_popular(article, decay_days=decay_days)
    secs = article.get("sections", []) or []
    boost = 0.3 if any(s in (user_top_sections or []) for s in secs) else 0.0
    return base * (1.0 + boost)
