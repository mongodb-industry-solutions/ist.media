#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
from typing import Dict, Any
from .base import Agent
from ..bandit import ThompsonScalarized
from ..util import utcnow
from ..config import DEFAULT_WEIGHTS

class RegisterWallAgent(Agent):
    name = "register_wall"

    def __init__(self, store, blackboard, experiments_api):
        super().__init__(store, blackboard)
        self.experiments_api = experiments_api

    def _matches_targeting(self, context: dict, targeting: dict) -> bool:
        # very simple matcher: only enforce anonymous if specified
        if targeting.get("user_is_anonymous") is True and context.get("is_authenticated") is True:
            return False
        return True

    def decide(self, *, user_id: str, article_id: str, context: dict) -> dict:
        """
        Preview-safe and assignment-safe decision:
        - Preview: no bandit, no assignment; just eligibility via cached popularity threshold.
        - Assignment: only choose an arm if the experiment is active and has ≥1 active arm;
          otherwise return control (no register wall) without raising.
        """
        preview = bool(context.get("preview", False))
        is_authenticated = bool(context.get("is_authenticated", False))
        exp_id = "register_wall"

        # Gate consistent with routes: authenticated users never see the wall
        if is_authenticated or not user_id:
            return {
                "arm_id": "control",
                "display_register_wall": False,
                "preview": preview,
                "reason": "auth_or_missing_user",
            }

        if preview:
            # PREVIEW PATH (no bandit):
            # Use the cached popularity threshold and the article's current read_count.
            thr = self.store.get_popular_threshold(
                min_read=POPULAR_MIN_READ,
                top_quantile=POPULAR_TOP_QUANTILE,
                ttl_hours=24,
                cache_key="popular_threshold_v1",
            )
            rc = self.store.read_counts_for([article_id]).get(article_id, 0)
            return {
                "arm_id": "preview",  # informative label; no assignment was made
                "display_register_wall": bool(rc >= thr),
                "preview": True,
                "thr": int(thr),          # optional debug
                "read_count": int(rc),    # optional debug
            }

        # ASSIGNMENT PATH (bandit):
        # Ensure the experiment is active
        exp = self.store.experiments.find_one({"experiment_id": exp_id})
        if not exp or exp.get("status") != "active":
            return {
                "arm_id": "control",
                "display_register_wall": False,
                "preview": False,
                "reason": "experiment_inactive",
            }

        # Ensure there is at least one active arm before choosing
        has_active_arm = self.store.arms.count_documents(
            {"experiment_id": exp_id, "active": True}, limit=1
        ) > 0
        if not has_active_arm:
            return {
                "arm_id": "control",
                "display_register_wall": False,
                "preview": False,
                "reason": "no_active_arms",
            }

        # Normal bandit choice (use your existing policy construction)
        policy = self._policy_for(exp_id)  # if you previously had local creation, keep that line
        arm_id, meta = policy.choose_arm(context)

        # Map arm -> rendering decision (adjust if you use different arm names)
        display = (arm_id == "register")

        return {
            "arm_id": arm_id,
            "display_register_wall": bool(display),
            "preview": False,
            "meta": meta,
        }

    # If you previously created the policy inline, keep that and remove this helper.
    # This helper simply centralizes however you build your policy.
    def _policy_for(self, experiment_id: str):
        """Return the bandit policy for the given experiment_id using your existing wiring."""
        # Example: if you previously did something else, replicate it here:
        # return self.policy_factory(experiment_id=experiment_id, store=self.store)
        # or return Policy(self.store, experiment_id)
        raise NotImplementedError  # replace with your existing policy construction

