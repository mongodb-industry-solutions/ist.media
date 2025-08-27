#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from flask import Blueprint, request, jsonify
from .planner import Planner
from .experiments import Experiments
from .util import utcnow
from datetime import datetime, timedelta
bp = Blueprint("agentic", __name__)
registry = None
planner_singleton: Planner = None
experiments_api: Experiments = None
store = None
def ensure_minimum_bootstrap():
    if not store.experiments.find_one({"status": "active"}):
        plan = planner_singleton.propose()
        for ex_class in ("register_wall", "homepage_ordering"):
            if plan.get(ex_class):
                experiments_api.create_from_plan(ex_class, plan.get(ex_class))
@bp.post("/decide")
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
    ex_ids = [e["_id"] for e in store.experiments.find({"experiment_class": ex_class, "status": "active"}, {"_id": 1})]
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
        arms = [ {"arm_id": k, "assignments": v["assignments"], "conversion_rate": (v["conv"]/max(v["days"],1)), "momentum_success_rate": (v["mom"]/max(v["days"],1))} for k,v in acc.items() ]
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
        series.setdefault(r["arm_id"], []).append({"day": r["day"], "assignments": r.get("assignments", 0), "conversion_rate": r.get("conversion_rate", 0.0), "momentum_success_rate": r.get("momentum_success_rate", 0.0)})
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
        weights[a.get("arm_id")] = {"conv_alpha": st.get("conv_alpha", 1.0), "conv_beta": st.get("conv_beta", 1.0), "mom_alpha": st.get("mom_alpha", 1.0), "mom_beta": st.get("mom_beta", 1.0)}
    return jsonify({"experiment_id": experiment_id, "weights": weights})
@bp.get("/admin/agents/logs")
def admin_agent_logs():
    rows = list(store.agent_logs.find({}).sort("ts", -1).limit(100))
    for r in rows:
        r["_id"] = str(r["_id"]); r["ts"] = r["ts"].isoformat()
    return jsonify({"logs": rows})
