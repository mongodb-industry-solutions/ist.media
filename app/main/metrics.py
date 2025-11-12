#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import request, jsonify
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

    since = now - timedelta(days=days_back-1)

    pipeline = [
        { "$match" : {
            "user_id" : user_id,
            "event" : { "$in" : [ "leave", "scroll" ]},
            "ts" : { "$gte" : since, "$lt" : end_for_days }
        }},
        { "$addFields" : {
            "day" : { "$dateTrunc" : { "date" : "$ts", "unit" : "day", "timezone": "UTC" }}
        }},
        { "$group" : {
            "_id" : "$day",
            "read_to_end_count" : { "$sum" : { "$cond" : [ "$read_to_end", 1, 0 ]}},
            "avg_scroll_depth" : { "$avg" : "$scroll_depth" },
            "distinct_articles" : { "$addToSet" : "$article_uuid" }
        }},
        { "$project" : {
            "_id" : 0,
            "day" : "$_id",
            "read_to_end_count" : { "$ifNull" : [ "$read_to_end_count", 0 ]},
            "avg_scroll_depth" : { "$ifNull" : [ "$avg_scroll_depth", 0 ]},
            "distinct_articles" : { "$size" : { "$ifNull" : [ "$distinct_articles", []]}},
        }},
        { "$sort" : { "day" : 1 }}
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
        if earliest_observed_dt:
            range_start = earliest_observed_dt
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
        effective_start = today_mid  # set to today for no activity
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
        days_since_last_visit = 0  # default for no activity

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
    # Momentum: ema7 / ema28 (return 0 if ema28 is 0)
    ema7_over_ema28 = round(ema7 / ema28, 2) if ema28 else 0.00

    # More momentum metrics
    window_days_observed = len(daily)
    gap_days_ratio = (
        round(gaps_count_28 / window_days_observed, 2)
        if window_days_observed > 0 else 0.00
    )

    # Calculate the first time a user_id was logged
    from .views import engagement_events_collection
    first_seen_doc = next(engagement_events_collection.aggregate([
        { "$match" : { "user_id" : user_id }},
        { "$group" : { "_id" : None, "first_seen" : { "$min" : "$ts" }}}
    ]), None)
    # Fallback: if no events exist, use effective_start (today if no activity)
    first_seen = (first_seen_doc or {}).get("first_seen") or effective_start

    return {
        "first_seen" : first_seen.isoformat(),
        "window_days_observed" : window_days_observed,
        "window_start_utc" : effective_start.isoformat(),
        "window_end_utc" : effective_end.isoformat(),
        "smoothed_indexes" : ema_snapshot,
        "momentum" : {
            "ema7_over_ema28" : ema7_over_ema28,
            "gap_days_ratio" : gap_days_ratio
        },
        "active_days_28" : active_days_28,
        "gaps_count_28" : gaps_count_28,
        "days_since_last_visit" : days_since_last_visit,
        "daily_indices" : daily
    }


request_stats_pipelines = {
        'countries' : [
            {
                "$match": {
                    "country": { "$exists": True },
                    "city": { "$exists": True }
                }
            },
            {
                "$group": {
                    "_id": {
                        "country": "$country",
                        "city": "$city"
                    },
                    "city_access_count": { "$sum": 1 }
                }
            },
            {
                "$sort": { "city_access_count": -1 }
            },
            {
                "$group": {
                    "_id": "$_id.country",
                    "access_count": { "$sum": "$city_access_count" },
                    "top_cities": {
                        "$push": {
                            "city": "$_id.city",
                            "access_count": "$city_access_count"
                        }
                    }
                }
            },
            {
                "$sort": { "access_count": -1 }
            },
            {
                "$limit": 12
            },
            {
                "$project": {
                    "_id": 1,
                    "access_count": 1,
                    "top_cities": {
                        "$slice": ["$top_cities", 5]
                    }
                }
            }
        ],
        'paths' : [
            {
                "$addFields": {
                    "month_year_sort": {
                        "$dateToString": {
                            "format": "%Y-%m",
                            "date": { "$toDate": "$timestamp" }
                        }
                    },
                    "month_year": {
                        "$dateToString": {
                            "format": "%b %Y",
                            "date": { "$toDate": "$timestamp" }
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "path": "$path",
                        "month_year_sort": "$month_year_sort",
                        "month_year": "$month_year"
                    },
                    "access_count": { "$sum": 1 }
                }
            },
            {
                "$group": {
                    "_id": {
                        "month_year_sort": "$_id.month_year_sort",
                        "month_year": "$_id.month_year"
                    },
                    "paths": {
                        "$push": {
                            "path": "$_id.path",
                            "access_count": "$access_count"
                        }
                    }
                }
            },
            {
                "$project": {
                    "month_year_sort": "$_id.month_year_sort",
                    "month_year": "$_id.month_year",
                    "top_paths": {
                        "$slice": [
                            { "$sortArray": {
                                "input": "$paths",
                                "sortBy": { "access_count": -1 }
                            }},
                            9
                        ]
                    }
                }
            },
            {
                "$unwind": "$top_paths"
            },
            {
                "$project": {
                    "_id": 0,
                    "month_year_sort": 1,
                    "month_year": 1,
                    "path": "$top_paths.path",
                    "access_count": "$top_paths.access_count"
                }
            },
            {
                "$sort": {
                    "month_year_sort": 1,
                    "access_count": -1
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "month_year": 1,
                    "path": 1,
                    "access_count": 1
                }
            }
        ]
}


def user_consumption_pipeline(user_id):
    pipeline = [
        {
            '$match': {
                'user_id': user_id,
                'event': 'view'
            }
        },
        {
            '$addFields': {
                'article_uuid_str': {
                    '$toString': '$article_uuid'
                },
                'day_of_week': {
                    '$arrayElemAt': [
                        [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                        { '$subtract': [ { '$isoDayOfWeek': '$ts' }, 1 ] }
                    ]
                },
                'day_of_week_index': { '$isoDayOfWeek': '$ts' }
            }
        },
        {
            '$lookup': {
                'from': 'news',
                'localField': 'article_uuid_str',
                'foreignField': 'uuid',
                'as': 'article'
            }
        },
        {
            '$unwind': '$article'
        },
        {
            '$facet': {
                'top_sections': [
                    {
                        '$unwind': '$article.sections'
                    },
                    {
                        '$group': {
                            '_id': '$article.sections',
                            'count': {'$sum': 1}
                        }
                    },
                    {
                        '$sort': {'count': -1}
                    },
                    {
                        '$limit': 7
                    },
                    {
                        '$project': {
                            'section': '$_id',
                            'count': 1,
                            '_id': 0
                        }
                    }
                ],
                'articles_by_day_of_week': [
                    {
                        '$group': {
                            '_id': '$day_of_week_index',
                            'day_of_week': { '$first': '$day_of_week' },
                            'count': { '$sum': 1 }
                        }
                    },
                    {
                        '$sort': { '_id' : 1 }
                    },
                    {
                        '$project': {
                            'day_of_week': '$day_of_week',
                            'count': 1,
                            '_id': 0
                        }
                    }
                ],
                'recent_articles': [
                    {
                        '$sort': {'ts': -1}
                    },
                    {
                        '$limit': 10
                    },
                    {
                        '$project': {
                            'title': '$article.title',
                            'sections': '$article.sections',
                            'clicked_at': '$ts',
                            'weekday_of_click': '$day_of_week',
                            '_id': 0
                        }
                    }
                ]
            }
        }
    ]
    return pipeline
