#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from datetime import datetime, timezone
def utcnow() -> datetime:
    return datetime.now(timezone.utc)
def day_str(dt_: datetime) -> str:
    return dt_.date().isoformat()
