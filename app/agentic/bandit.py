#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from pymongo.collection import Collection
@dataclass
class ArmStats:
    conv_alpha: float = 1.0
    conv_beta: float = 1.0
    mom_alpha: float = 1.0
    mom_beta: float = 1.0
    assignments: int = 0
    conversions: int = 0
    momentum_successes: int = 0
class ThompsonScalarized:
    def __init__(self, arms_coll: Collection, experiment: Dict[str, Any], weights: Tuple[float, float]):
        self.arms = arms_coll
        self.experiment = experiment
        self.weights = weights
    @staticmethod
    def _beta_sample(a: float, b: float) -> float:
        x = random.gammavariate(max(a, 1e-6), 1.0)
        y = random.gammavariate(max(b, 1e-6), 1.0)
        return x / (x + y) if (x + y) > 0 else 0.5
    def choose_arm(self, context: Dict[str, Any]):
        arms = list(self.arms.find({"experiment_id": self.experiment["_id"], "status": {"$ne": "disabled"}}))
        if not arms:
            raise RuntimeError("No active arms for experiment")
        w_conv, w_mom = self.weights
        scores, samples = {}, {}
        for arm in arms:
            s = arm.get("stats", {})
            s_conv = self._beta_sample(float(s.get("conv_alpha", 1.0)), float(s.get("conv_beta", 1.0)))
            s_mom = self._beta_sample(float(s.get("mom_alpha", 1.0)), float(s.get("mom_beta", 1.0)))
            score = w_conv * s_conv + w_mom * s_mom
            scores[arm["arm_id"]] = score
            samples[arm["arm_id"]] = {"sample_conv": s_conv, "sample_mom": s_mom, "score": score}
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        return chosen, {"weights": [w_conv, w_mom], "samples": samples, "context": context}
