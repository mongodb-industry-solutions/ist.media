#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from flask import Blueprint, request, jsonify, render_template
from .planner import Planner
from .experiments import Experiments
from .util import utcnow
from datetime import datetime, timedelta

bp = Blueprint("agentic", __name__)
registry = None
planner_singleton: Planner = None
experiments_api: Experiments = None
store = None


def _summarize_log(r):
    agent = r.get("agent", "")
    act = r.get("action", "")
    d = (r.get("details") or r.get("payload") or {})
    if agent == "planner" and act == "applied":
        ec = d.get("experiment_class"); cc = d.get("created_count"); aft = d.get("active_after")
        return f"{ec} — created {cc} (active {aft})" if (ec is not None and cc is not None and aft is not None) else "applied new experiments"
    if agent == "planner" and act == "proposed":
        keys = ", ".join((d.get("keys") or [])); cnts = d.get("counts") or {}
        return f"proposed: {keys} (rw={cnts.get('register_wall',0)}, hp={cnts.get('homepage_ordering',0)})" if keys else "proposed plan"
    if agent == "rewards" and act == "resolved":
        return f"resolved={d.get('total_resolved',0)}; conv_true={d.get('conv_true',0)}; mom_true={d.get('mom_true',0)}; rollup_rows={d.get('rollup_rows',0)}"
    if agent == "analytics" and act == "rollup":
        return f"rollup last {d.get('since_days',0)}d → {d.get('rows',0)} points"
    if agent == "register_wall" and act == "assigned":
        arm = d.get("arm_label") or d.get("arm_id"); usr = d.get("user_id"); art = d.get("article_id")
        msg = f"assigned arm={arm} to user={usr}" if arm and usr else "assigned register wall arm"
        return f"{msg} (article={art})" if art else msg
    if agent == "homepage" and act == "assigned":
        strat = d.get("strategy"); k = d.get("decay_days"); n = d.get("articles_count"); secs = d.get("user_top_sections") or []
        if strat and n is not None:
            sec_txt = ", ".join(secs) if secs else "—"
            return f"{strat} (k={k}) → {n} articles; user sections: {sec_txt}" if k is not None else f"{strat} → {n} articles; user sections: {sec_txt}"
    return ""


def user_status_json(user_id: str) -> dict:
    """Return assignments, events, and momentum_series for a user."""
    asg = list(store.assignments.find(
        {"user_id": user_id},
        {
            "experiment_id": 1, "experiment_class": 1, "arm_id": 1,
            "ts_assigned": 1, "ts_resolved": 1, "resolved": 1,
            "conversion_success": 1, "momentum_success": 1, "context": 1
        }
    ).sort("ts_assigned", -1).limit(50))

    # hydrate experiment + arm labels
    exp_ids = list({a["experiment_id"] for a in asg})
    arms = {}
    if exp_ids:
        for a in store.arms.find(
            {"experiment_id": {"$in": exp_ids}},
            {"experiment_id": 1, "arm_id": 1, "label": 1, "stats": 1}
        ):
            arms[(a["experiment_id"], a["arm_id"])] = a

    hydrated = []
    for a in asg:
        arm = arms.get((a["experiment_id"], a["arm_id"]))
        hydrated.append({
            "experiment_class": a["experiment_class"],
            "experiment_id": str(a["experiment_id"]),
            "arm_id": a["arm_id"],
            "arm_label": (arm or {}).get("label", a["arm_id"]),
            "assigned_at": a.get("ts_assigned").isoformat() if a.get("ts_assigned") else None,
            "resolved_at": a.get("ts_resolved").isoformat() if a.get("ts_resolved") else None,
            "resolved": bool(a.get("resolved")),
            "conversion_success": a.get("conversion_success"),
            "momentum_success": a.get("momentum_success"),
            "context": a.get("context", {}),
            "arm_stats": (arm or {}).get("stats", {}),
        })

    # events
    evs = list(store.events.find(
        {"user_id": user_id},
        {"event": 1, "article_id": 1, "ts": 1}
    ).sort("ts", -1).limit(15))
    for e in evs:
        e["_id"] = str(e["_id"])
        e["ts"] = e["ts"].isoformat()

    # momentum (last 14 days)
    from datetime import datetime, timedelta, timezone
    since = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
    series = list(store.metrics_daily.find(
        {"user_id": user_id, "day": {"$gte": since}},
        {"day": 1, "momentum.ema7_over_ema28": 1}
    ).sort("day", 1))
    momentum_series = [
        {"day": p["day"], "ema7_over_ema28": (p.get("momentum") or {}).get("ema7_over_ema28")}
        for p in series
    ]

    return {
        "user_id": user_id,
        "assignments": hydrated,
        "events": evs,
        "momentum_series": momentum_series,
    }


def ensure_minimum_bootstrap():
    if not store.experiments.find_one({"status": "active"}):
        plan = planner_singleton.propose()
        for ex_class in ("register_wall", "homepage_ordering"):
            if plan.get(ex_class):
                experiments_api.create_from_plan(ex_class, plan.get(ex_class))


@bp.post("/decide_register_wall")
def decide_register_wall():
    ensure_minimum_bootstrap()
    data = request.get_json(force=True)
    user_id = data.get("user_id"); article_id = data.get("article_id")
    if not user_id or not article_id:
        return jsonify({"error": "user_id and article_id required"}), 400
    popular_docs = planner_singleton.popular_articles()
    if article_id not in {d.get("uuid") for d in popular_docs}:
        return jsonify({"display_register_wall": False})
    ctx = {"days_since_last_visit": data.get("days_since_last_visit"), "momentum": data.get("momentum")}
    result = registry.get("register_wall").decide(user_id=user_id, article_id=article_id, context=ctx)
    return jsonify(result)


@bp.post("/decide_homepage")
def decide_homepage():
    ensure_minimum_bootstrap()
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    ctx = {"days_since_last_visit": data.get("days_since_last_visit"), "momentum": data.get("momentum")}
    result = registry.get("homepage").decide(user_id=user_id, context=ctx)
    return jsonify(result)


@bp.post("/event")
def ingest_event():
    data = request.get_json(force=True)
    if not data.get("user_id") or not data.get("event"):
        return jsonify({"error": "user_id and event required"}), 400
    ts = data.get("ts")
    when = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else utcnow()
    doc = {"user_id": data["user_id"], "event": data["event"], "ts": when}
    if data.get("article_id"): doc["article_id"] = data["article_id"]
    if data.get("scroll_depth") is not None: doc["scroll_depth"] = float(data.get("scroll_depth"))
    store.events.insert_one(doc)
    return jsonify({"ok": True})


@bp.get("/admin/arms/overview")
def admin_overview():
    ex_class = request.args.get("experiment_class")
    days = int(request.args.get("days", "14"))
    since = (utcnow() - timedelta(days=days)).date().isoformat()
    ex_ids = [e["_id"] for e in store.experiments.find({"experiment_class": ex_class,
                                                        "status": "active"}, {"_id": 1})]
    out = []
    for ex in ex_ids:
        rows = list(store.campaign_ts.find({"experiment_id": ex, "day": {"$gte": since}}))
        acc = {}
        for r in rows:
            arm = r["arm_id"]
            a = acc.setdefault(arm, {"assignments": 0, "conv": 0.0, "mom": 0.0, "days": 0})
            a["assignments"] += r.get("assignments", 0)
            a["conv"] += r.get("conversion_rate", 0.0)
            a["mom"] += r.get("momentum_success_rate", 0.0)
            a["days"] += 1
        arms = [ {"arm_id": k, "assignments": v["assignments"],
                  "conversion_rate": (v["conv"]/max(v["days"],1)),
                  "momentum_success_rate": (v["mom"]/max(v["days"],1))} for k,v in acc.items() ]
        out.append({"experiment_id": str(ex), "arms": arms})
    return jsonify({"experiment_class": ex_class, "days": days, "experiments": out})


@bp.get("/admin/arms/series")
def admin_series():
    from bson import ObjectId
    experiment_id = request.args.get("experiment_id")
    days = int(request.args.get("days", "28"))
    if not experiment_id:
        return jsonify({"error": "experiment_id required"}), 400
    ex_oid = ObjectId(experiment_id) if len(experiment_id) == 24 else None
    if not ex_oid:
        return jsonify({"error": "experiment_id must be a Mongo ObjectId (24 hex)"}), 400
    since = (utcnow() - timedelta(days=days)).date().isoformat()
    rows = list(store.campaign_ts.find({"experiment_id": ex_oid, "day": {"$gte": since}}).sort("day", 1))
    series = {}
    for r in rows:
        series.setdefault(r["arm_id"], []).append({"day": r["day"], "assignments": r.get("assignments", 0),
                                                   "conversion_rate": r.get("conversion_rate", 0.0),
                                                   "momentum_success_rate": r.get("momentum_success_rate", 0.0)})
    out = [{"arm_id": k, "points": v} for k, v in series.items()]
    return jsonify({"experiment_id": experiment_id, "days": days, "series": out})


@bp.get("/admin/arms/weights")
def admin_weights():
    from bson import ObjectId
    experiment_id = request.args.get("experiment_id")
    if not experiment_id:
        return jsonify({"error": "experiment_id required"}), 400
    ex_oid = ObjectId(experiment_id) if len(experiment_id) == 24 else None
    if not ex_oid:
        return jsonify({"error": "experiment_id must be a Mongo ObjectId (24 hex)"}), 400
    arms = list(store.arms.find({"experiment_id": ex_oid}))
    weights = {}
    for a in arms:
        st = a.get("stats", {})
        weights[a.get("arm_id")] = {"conv_alpha": st.get("conv_alpha", 1.0),
                                    "conv_beta": st.get("conv_beta", 1.0), "mom_alpha": st.get("mom_alpha", 1.0),
                                    "mom_beta": st.get("mom_beta", 1.0)}
    return jsonify({"experiment_id": experiment_id, "weights": weights})


@bp.get("/admin/agents/logs")
def admin_agent_logs():
    rows = list(store.agent_logs.find({}).sort("ts", -1).limit(100))
    for r in rows:
        r["_id"] = str(r["_id"])
        if r.get("ts"): r["ts"] = r["ts"].isoformat()
        r["summary"] = _summarize_log(r)
        r["has_details"] = bool(r.get("details") or r.get("payload"))
    return jsonify({"logs": rows})


@bp.get("/status/user")
def status_user():
    # ADAPT: pull from your auth (e.g., flask-login). For demo we use query param or fallback.
    user_id = request.args.get("user_id") or "bjjl"
    data = user_status_json(user_id)
    return render_template("status_user.html", **data)


@bp.get("/status/system")
def status_system():
    from datetime import timedelta
    from .util import utcnow

    days = int(request.args.get("days", "14"))
    since = (utcnow() - timedelta(days=days)).date().isoformat()

    exps = list(store.experiments.find({"status": "active"}))
    out = []
    for ex in exps:
        rows = list(store.campaign_ts.find({"experiment_id": ex["_id"], "day": {"$gte": since}}))
        acc = {}
        for r in rows:
            arm = r["arm_id"]
            a = acc.setdefault(arm, {"assignments": 0, "conv": 0.0, "mom": 0.0, "days": 0})
            a["assignments"] += r.get("assignments", 0)
            a["conv"] += r.get("conversion_rate", 0.0)
            a["mom"] += r.get("momentum_success_rate", 0.0)
            a["days"] += 1
        arms = [{
            "arm_id": k,
            "assignments": v["assignments"],
            "conversion_rate": (v["conv"]/max(v["days"],1)),
            "momentum_success_rate": (v["mom"]/max(v["days"],1)),
        } for k, v in acc.items()]
        out.append({"experiment_class": ex.get("experiment_class"),
                    "experiment_id": ex.get("experiment_id"),
                    "arms": arms})

    logs = list(store.agent_logs.find({}).sort("ts", -1).limit(100))
    for l in logs:
        l["_id"] = str(l["_id"])
        if l.get("ts"): l["ts"] = l["ts"].isoformat()
        l["summary"] = _summarize_log(l)          # <-- neu
        l["has_details"] = bool(l.get("details") or l.get("payload"))

    return render_template("status_system.html",
                           experiments=out,
                           agent_logs=logs,
                           days=days)
