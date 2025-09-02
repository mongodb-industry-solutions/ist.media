#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from datetime import datetime, timedelta, timezone


def _daterange_utc_days(start_incl: datetime, end_excl: datetime):
    """
    Yield UTC midnights from start (inclusive) to end (exclusive).
    """
    cur = datetime(start_incl.year, start_incl.month, start_incl.day,
                   tzinfo=timezone.utc)
    end = datetime(end_excl.year, end_excl.month, end_excl.day,
                   tzinfo=timezone.utc)
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def _compute_daily_indices(user_id: str, days_back: int = 28,
                           fill_gaps: bool = True,
                           include_today: bool = True):
    """
    Aggregate daily engagement indices (UTC) for the past 'days_back'
    *calendar* days.

    Daily index:
        index = 5 * read_to_end_count + 0.3 * avg_scroll_depth
                + 2 * distinct_articles

    Returns:
      - daily list (newest first). If fill_gaps=True, missing days are
        zero-filled, but ONLY starting from the first observed activity
        day (no 28-day zero burden for brand-new users).
      - effective_start (UTC midnight of earliest included day)
      - effective_end   (UTC midnight of today, exclusive; if
        include_today=True then tomorrow 00:00 UTC)
      - active_days_28
      - gaps_count_28
      - days_since_last_visit
      - today_is_partial (bool)
    """
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    today_mid = datetime.utcnow().replace(hour=0, minute=0, second=0,
                                          microsecond=0, tzinfo=timezone.utc)
    end_for_days = today_mid + timedelta(days=1) if include_today else today_mid

    since = now - timedelta(days=days_back)

    pipeline = [
        {"$match": {
            "user_id": user_id,
            "event": {"$in": ["leave", "scroll"]},
            "ts": {"$gte": since, "$lt": end_for_days}
        }},
        {"$addFields": {
            "day": {"$dateTrunc": {"date": "$ts", "unit": "day",
                                   "timezone": "UTC"}}
        }},
        {"$group": {
            "_id": "$day",
            "read_to_end_count": {"$sum": {"$cond": ["$read_to_end", 1, 0]}},
            "avg_scroll_depth": {"$avg": "$scroll_depth"},
            "distinct_articles": {"$addToSet": "$article_uuid"}
        }},
        {"$project": {
            "_id": 0,
            "day": "$_id",
            "read_to_end_count": {"$ifNull": ["$read_to_end_count", 0]},
            "avg_scroll_depth": {"$ifNull": ["$avg_scroll_depth", 0]},
            "distinct_articles": {"$size": {"$ifNull": ["$distinct_articles",
                                                        []]}},
        }},
        {"$sort": {"day": 1}}  # ascending for EMA calculation
    ]

    from .views import engagement_events_collection
    docs = list(engagement_events_collection.aggregate(pipeline))

    # Map existing days -> computed metrics
    by_day = {}
    for d in docs:
        day_dt = d["day"]
        if day_dt.tzinfo is None:
            day_dt = day_dt.replace(tzinfo=timezone.utc)

        R = float(d["read_to_end_count"] or 0)
        A = float(d["avg_scroll_depth"] or 0.0)
        D = float(d["distinct_articles"] or 0)
        idx = 5.0 * R + 0.3 * A + 2.0 * D

        by_day[day_dt] = {
            "read_to_end_count": int(R),
            "avg_scroll_depth": round(A, 2),
            "distinct_articles": int(D),
            "index": round(idx, 2),
        }

    # Earliest observed day (first activity in window), if any
    earliest_observed_dt = min(by_day.keys()) if by_day else None

    # Build daily list
    daily = []
    if fill_gaps:
        range_start = earliest_observed_dt if earliest_observed_dt else since
        for day_dt in _daterange_utc_days(range_start, end_for_days):
            metrics = by_day.get(day_dt, {
                "read_to_end_count": 0,
                "avg_scroll_depth": round(0.0, 2),
                "distinct_articles": 0,
                "index": round(0.0, 2),
            })
            daily.append({"day": day_dt.date().isoformat(), **metrics})
    else:
        for day_dt, metrics in by_day.items():
            daily.append({"day": day_dt.date().isoformat(), **metrics})

    # Newest first for output
    daily.sort(key=lambda x: x["day"], reverse=True)

    # Effective window aligned to the daily array
    if daily:
        effective_start = datetime.fromisoformat(daily[-1]["day"])\
                                  .replace(tzinfo=timezone.utc)
    else:
        effective_start = datetime(since.year, since.month, since.day,
                                   tzinfo=timezone.utc)
    effective_end = end_for_days  # today 24:00 UTC if include_today=True

    # Helper stats
    active_days_28 = sum(1 for x in daily if x["index"] > 0)
    gaps_count_28 = sum(1 for x in daily if x["index"] == 0)

    # days_since_last_visit
    last_active_day_iso = next((x["day"] for x in daily if x["index"] > 0),
                               None)
    if last_active_day_iso:
        last_active_dt = datetime.fromisoformat(last_active_day_iso)\
                                 .replace(tzinfo=timezone.utc)
        ref_anchor = today_mid if include_today else effective_end
        days_since_last_visit = max(0, (ref_anchor - last_active_dt).days)
    else:
        days_since_last_visit = None

    # Flag: is the top entry "today"? (partial day when include_today=True)
    today_is_partial = (include_today and len(daily) > 0 and
                        daily[0]["day"] == today_mid.date().isoformat())

    return (daily, effective_start, effective_end, active_days_28,
            gaps_count_28, days_since_last_visit, today_is_partial)


def _ema_final(values_ascending, N: int):
    """
    Compute final EMA for window N on chronological series (oldest->newest).
    Returns unrounded float (rounding at output).
    """
    if not values_ascending:
        return 0.0
    alpha = 2.0 / (float(N) + 1.0)
    ema = None
    for v in values_ascending:
        ema = v if ema is None else alpha * v + (1.0 - alpha) * ema
    return ema if ema is not None else 0.0


def compute_user_engagement(user_id: str, windows=(3, 7, 28),
                            fill_gaps: bool = True,
                            include_today: bool = True):
    """
    Returns:
      - raw daily_indices (newest first), with missing days zero-filled from
        FIRST activity day onward,
      - EMA snapshot for provided windows (e.g., 3/7/28),
      - helper fields: days_since_last_visit, active_days_28, gaps_count_28,
        momentum (ema7_over_ema28),
      - effective window (start/end) matching the daily_indices array,
      - today_is_partial flag.
    All floats rounded to 2 decimals in the JSON response.
    """
    (daily, effective_start, effective_end, active_days_28, gaps_count_28,
     days_since_last_visit, today_is_partial) = _compute_daily_indices(
         user_id, days_back=max(windows) if windows else 28,
         fill_gaps=fill_gaps, include_today=include_today)

    # EMA on chronological order (oldest -> newest)
    values_asc = [d["index"] for d in sorted(daily, key=lambda x: x["day"])]

    ema_snapshot = {}
    for N in sorted(set(windows)):
        ema_val = _ema_final(values_asc, N)
        ema_snapshot[f"ema{N}"] = round(ema_val, 2) if values_asc else 0.00

    # Momentum: ema7 / ema28 (guard division by zero)
    ema7 = ema_snapshot.get("ema7", 0.00)
    ema28 = ema_snapshot.get("ema28", 0.00)
    if ema28:
        ema7_over_ema28 = round(ema7 / ema28, 2)
    else:
        ema7_over_ema28 = None

    return {
        "window_days_observed": len(daily),
        "window_start_utc": effective_start.isoformat(),
        "window_end_utc": effective_end.isoformat(),
        "today_is_partial": bool(today_is_partial),
        "ema": ema_snapshot,
        "momentum": {"ema7_over_ema28": ema7_over_ema28},
        "active_days_28": active_days_28,
        "gaps_count_28": gaps_count_28,
        "days_since_last_visit": days_since_last_visit,
        "daily_indices": daily
    }


def compute_article_engagement():
    match_stage = {"event": {"$in": ["scroll_progress", "leave"]}}

    try:
        days = request.args.get("days", 30, type=int)
        if days and days > 0:
            start_date = datetime.utcnow() - timedelta(days=days)
            match_stage["ts"] = {"$gte": start_date}
    except ValueError:
        start_date = datetime.utcnow() - timedelta(days=30)
        match_stage["ts"] = {"$gte": start_date}

    sort_param = request.args.get("sort", "read_visit").lower()
    try:
        limit = request.args.get("limit", 15, type=int)
        if limit <= 0:
            limit = 15
    except ValueError:
        limit = 15

    if sort_param == "scroll":
        sort_stage = {"$sort": {"avg_scroll_depth": -1, "read_to_end_rate_visit": -1}}
    elif sort_param == "read_user":
        sort_stage = {"$sort": {"read_to_end_rate_user": -1, "avg_scroll_depth": -1}}
    elif sort_param == "read_first":
        sort_stage = {"$sort": {"read_to_end_rate_first_visit": -1, "avg_scroll_depth": -1}}
    else:
        sort_stage = {"$sort": {"read_to_end_rate_visit": -1, "avg_scroll_depth": -1}}

    pipeline = [
        {"$match": match_stage},

        {"$facet": {
            # visit-based metrics
            "visits": [
                {"$match": {"event": "leave"}},
                {"$group": {
                    "_id": "$article_uuid",
                    "total_leaves": {"$sum": 1},
                    "read_to_end_count": {"$sum": {"$cond": ["$read_to_end", 1, 0]}},
                    "avg_leave_scroll": {"$avg": "$scroll_depth"}
                }}
            ],
            # user-based best-ever
            "users": [
                {"$addFields": {"user_key": {"$ifNull": ["$user_id", "$anon_id"]}}},
                {"$group": {
                    "_id": {"article_uuid": "$article_uuid", "user_key": "$user_key"},
                    "best_read_to_end": {"$max": {"$cond": ["$read_to_end", 1, 0]}},
                    "any_leave": {"$max": {"$cond": [{"$eq": ["$event", "leave"]}, 1, 0]}}
                }},
                {"$group": {
                    "_id": "$_id.article_uuid",
                    "read_to_end_users": {"$sum": {"$cond": ["$best_read_to_end", 1, 0]}},
                    "total_users_with_leave": {"$sum": {"$cond": ["$any_leave", 1, 0]}}
                }}
            ],
            # first-visit-only metric
            "first_visits": [
                {"$addFields": {"user_key": {"$ifNull": ["$user_id", "$anon_id"]}}},
                {"$match": {"event": "leave"}},
                {"$sort": {"ts": 1}},
                {"$group": {
                    "_id": {"article_uuid": "$article_uuid", "user_key": "$user_key"},
                    "first_ts": {"$first": "$ts"},
                    "first_read_to_end": {"$first": "$read_to_end"}
                }},
                {"$group": {
                    "_id": "$_id.article_uuid",
                    "read_to_end_first_visit_count": {"$sum": {"$cond": ["$first_read_to_end", 1, 0]}},
                    "total_first_visits": {"$sum": 1}
                }}
            ]
        }},

        # merge facet outputs
        {"$project": {
            "merged": {
                "$map": {
                    "input": "$visits",
                    "as": "v",
                    "in": {
                        "$mergeObjects": [
                            "$$v",
                            {
                                "$ifNull": [
                                    {
                                        "$first": {
                                            "$filter": {
                                                "input": "$users",
                                                "as": "u",
                                                "cond": {"$eq": ["$$u._id", "$$v._id"]}
                                            }
                                        }
                                    },
                                    {"_id": "$$v._id", "read_to_end_users": 0, "total_users_with_leave": 0}
                                ]
                            },
                            {
                                "$ifNull": [
                                    {
                                        "$first": {
                                            "$filter": {
                                                "input": "$first_visits",
                                                "as": "f",
                                                "cond": {"$eq": ["$$f._id", "$$v._id"]}
                                            }
                                        }
                                    },
                                    {"_id": "$$v._id", "read_to_end_first_visit_count": 0, "total_first_visits": 0}
                                ]
                            }
                        ]
                    }
                }
            }
        }},
        {"$unwind": "$merged"},
        {"$replaceRoot": {"newRoot": "$merged"}},

        # computed fields
        {"$addFields": {
            "article_uuid_str": {"$toString": "$_id"},

            "avg_scroll_depth": {"$round": [{"$multiply": ["$avg_leave_scroll", 100]}, 2]},
            "avg_scroll_depth_int": {"$toInt": {"$round": [{"$multiply": ["$avg_leave_scroll", 100]}, 0]}},

            "read_to_end_rate_visit": {
                "$round": [
                    {"$multiply": [
                        {"$cond": [
                            {"$gt": ["$total_leaves", 0]},
                            {"$divide": ["$read_to_end_count", "$total_leaves"]},
                            0
                        ]},
                        100
                    ]},
                    2
                ]
            },
            "read_to_end_rate_visit_int": {
                "$toInt": {
                    "$round": [
                        {"$multiply": [
                            {"$cond": [
                                {"$gt": ["$total_leaves", 0]},
                                {"$divide": ["$read_to_end_count", "$total_leaves"]},
                                0
                            ]},
                            100
                        ]},
                        0
                    ]
                }
            },

            "read_to_end_rate_user": {
                "$round": [
                    {"$multiply": [
                        {"$cond": [
                            {"$gt": ["$total_users_with_leave", 0]},
                            {"$divide": ["$read_to_end_users", "$total_users_with_leave"]},
                            0
                        ]},
                        100
                    ]},
                    2
                ]
            },
            "read_to_end_rate_user_int": {
                "$toInt": {
                    "$round": [
                        {"$multiply": [
                            {"$cond": [
                                {"$gt": ["$total_users_with_leave", 0]},
                                {"$divide": ["$read_to_end_users", "$total_users_with_leave"]},
                                0
                            ]},
                            100
                        ]},
                        0
                    ]
                }
            },

            "read_to_end_rate_first_visit": {
                "$round": [
                    {"$multiply": [
                        {"$cond": [
                            {"$gt": ["$total_first_visits", 0]},
                            {"$divide": ["$read_to_end_first_visit_count", "$total_first_visits"]},
                            0
                        ]},
                        100
                    ]},
                    2
                ]
            },
            "read_to_end_rate_first_visit_int": {
                "$toInt": {
                    "$round": [
                        {"$multiply": [
                            {"$cond": [
                                {"$gt": ["$total_first_visits", 0]},
                                {"$divide": ["$read_to_end_first_visit_count", "$total_first_visits"]},
                                0
                            ]},
                            100
                        ]},
                        0
                    ]
                }
            }
        }},

        {"$lookup": {
            "from": "news",
            "localField": "article_uuid_str",
            "foreignField": "uuid",
            "as": "article_info"
        }},
        {"$unwind": {"path": "$article_info", "preserveNullAndEmptyArrays": True}},

        {"$project": {
            "_id": 0,
            "uuid": "$article_uuid_str",
            "title": "$article_info.title",

            "total_leaves": 1,
            "read_to_end_count": 1,
            "read_to_end_rate_visit": 1,
            "read_to_end_rate_visit_int": 1,

            "read_to_end_users": 1,
            "total_users_with_leave": 1,
            "read_to_end_rate_user": 1,
            "read_to_end_rate_user_int": 1,

            "read_to_end_first_visit_count": 1,
            "total_first_visits": 1,
            "read_to_end_rate_first_visit": 1,
            "read_to_end_rate_first_visit_int": 1,

            "avg_scroll_depth": 1,
            "avg_scroll_depth_int": 1
        }},

        sort_stage,
        {"$limit": limit}
    ]

    from .views import engagement_events_collection
    results = list(engagement_events_collection.aggregate(pipeline))
    return jsonify(results)
