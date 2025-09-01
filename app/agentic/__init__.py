#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import os, threading, time, logging
from flask import Flask

from .config import LOG_LEVEL, PLANNER_INTERVAL_SEC
from .store import Store
from .planner import Planner
from .experiments import Experiments
from .rewards import Rewards
from .agents.blackboard import Blackboard
from .agents.base import AgentRegistry
from .agents.planner_agent import PlannerAgent
from .agents.reward_agent import RewardAgent
from .agents.analytics_agent import AnalyticsAgent
from .agents.register_wall_agent import RegisterWallAgent
from .agents.homepage_agent import HomepageAgent
from .routes import bp as api_bp
import app.agentic.routes as routes  # to inject singletons

logger = logging.getLogger("bandit")

def register_agentic(app: Flask) -> None:
    """Mount the agentic blueprint at /agentic and start coordinator."""
    if not os.getenv("MONGODB_IST_MEDIA"):
        raise SystemExit("MONGODB_IST_MEDIA is required.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required — planner is LLM-only.")

    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(message)s')

    store = Store(os.getenv("MONGODB_IST_MEDIA"))
    store.init_indexes()
    planner = Planner(store)
    experiments_api = Experiments(store)
    rewards = Rewards(store)

    bb = Blackboard(store)
    registry = AgentRegistry()
    registry.register(PlannerAgent(store, bb, planner, experiments_api))
    registry.register(RewardAgent(store, bb, rewards))
    registry.register(AnalyticsAgent(store, bb, rewards))
    registry.register(RegisterWallAgent(store, bb, experiments_api))
    registry.register(HomepageAgent(store, bb, experiments_api))

    routes.registry = registry
    routes.planner_singleton = planner
    routes.experiments_api = experiments_api
    routes.store = store

    try:
        # LLM-driven bootstrap at startup so preview/assignments run in a live experiment context
        routes.ensure_minimum_bootstrap()
    except Exception as e:
        logging.getLogger(__name__).warning("Startup bootstrap failed: %s", e)

    def coordinator_loop():
        while True:
            registry.get("planner").tick()
            registry.get("rewards").tick()
            registry.get("analytics").tick()
            time.sleep(PLANNER_INTERVAL_SEC)
    threading.Thread(target=coordinator_loop, daemon=True).start()

    app.register_blueprint(api_bp, url_prefix="/agentic")
