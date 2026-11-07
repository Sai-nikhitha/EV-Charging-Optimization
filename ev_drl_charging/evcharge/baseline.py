"""Stage 4 - deterministic baseline policy: urgency-weighted proportional allocation.

Method (per interval):
  1. Chargeable EVs = present and not yet at target (caps_kw > 0).
  2. Weight_i = priority_score_i + 1   (the +1 keeps every chargeable EV eligible).
  3. Share the deliverable power (min(station, solar + grid limit)) in proportion to the weights,
     each EV capped at its own cap (max power / energy needed to target). Power freed by capped EVs
     is redistributed ("water-filling").
Properties: deterministic; independent of the order EVs are listed in; never charges absent or
completed EVs; ignores electricity price (it charges as soon as possible, like a simple
load-sharing charger); handles infeasible EVs by still giving them a (high-priority) share.
Justification: uncontrolled "charge on arrival" is the usual reference, but it needs a sharing
rule once the station is oversubscribed; priority-weighted sharing is a transparent choice.
This is a reference policy for this study, not a model of every real station.
"""
from __future__ import annotations

import numpy as np

from .allocation import weighted_waterfill
from .core import ChargingCore
from .priority import PriorityWeights, evaluate_priority


def baseline_requests(core: ChargingCore, w: PriorityWeights = PriorityWeights()) -> np.ndarray:
    pk = core.peek()
    scores = np.zeros(core.n)
    for i, s in enumerate(core.states):
        if pk.caps_kw[i] > 0:
            info = evaluate_priority(s, pk.t0, pk.limit_kw, core.prev_alloc[i] > 1e-9, w)
            scores[i] = info.score
    weights = np.where(pk.caps_kw > 0, scores + 1.0, 0.0)
    return weighted_waterfill(pk.limit_kw, weights, pk.caps_kw)
