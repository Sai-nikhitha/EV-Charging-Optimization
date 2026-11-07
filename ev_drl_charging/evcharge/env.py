"""Stage 5 - Gymnasium environment for PPO.

Fixed-size spaces with MAX_EVS slots. Slots beyond the scenario's EV count are *inactive*:
all their observation features are 0 and their action component is ignored.

ACTION  Box(0, 1, shape=(MAX_EVS,)):  a_i = requested fraction of EV i's maximum power.
        request_i = a_i * P_max_i. Requests are then projected by `allocate_power` (core.py):
        clipped to the EV cap (max power / energy-to-target / presence), and scaled down
        proportionally if their sum exceeds min(station capacity, solar + grid limit).
        So every action maps to a physically valid allocation; constraints are NOT left to rewards.

OBSERVATION (all roughly in [0, 1], see OBS_* below)
  per slot (12): chargeable, arrived, completed, SOC, target SOC, required-energy fraction,
                 remaining time, max power, required/available power ratio, priority score,
                 previous allocation fraction, cap fraction
  global (9): station capacity, solar now, grid limit, tariff now, mean/min remaining tariff,
              mean solar next hour, elapsed-time fraction, active-EV fraction

REWARD (per step; R = total energy required at start, tref = mean tariff over horizon)
  + 1.0  * delivered_kWh / R                     progress: delivering required energy is rewarded, so
                                                  "save money by not charging" is never attractive
  - 0.5  * grid_cost / (R * tref)                 cost: cost of delivering R at the mean tariff = 0.5
  + 0.1  * solar_kWh / R                          small bonus for using free solar energy
  + 0.1  per EV reaching its target
  - (1.0 + 3.0 * unmet_i / R) per EV leaving without target   deadline-miss penalty (v2; v1 used 0.5 + 2.0*unmet/R)
  - 0.05 * limit_excess_kW / station_limit        discourages over-requesting (already corrected by projection)
Because progress is paid per kWh independent of time, timing decisions are driven by cost and
deadlines only. Weights are design choices, not tuned to favour PPO.
"""
from __future__ import annotations

import math
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import ChargingCore
from .models.scenario import Scenario
from .priority import PriorityWeights, evaluate_priority, required_power_kw
from .scenario_gen import random_scenario

MAX_EVS = 5
EV_FEATURES = 12
GLOBAL_FEATURES = 9
OBS_DIM = MAX_EVS * EV_FEATURES + GLOBAL_FEATURES
P_SCALE_KW = 50.0       # power normaliser
TARIFF_SCALE = 15.0     # Rs/kWh normaliser
T_SCALE_H = 12.0        # time normaliser


def build_observation(core: ChargingCore, max_evs: int = MAX_EVS) -> np.ndarray:
    pk = core.peek() if not core.done else None
    st = core.station
    obs = np.zeros(max_evs * EV_FEATURES + GLOBAL_FEATURES, dtype=np.float32)
    if pk is None:
        return obs
    w = PriorityWeights()
    for i in range(min(core.n, max_evs)):
        s, c = core.states[i], core.cfgs[i]
        chargeable = pk.caps_kw[i] > 0
        arrived = pk.t1 > c.arrival_h + 1e-9
        t_rem = max(0.0, c.departure_h - max(pk.t0, c.arrival_h))
        if chargeable:
            info = evaluate_priority(s, pk.t0, pk.limit_kw, core.prev_alloc[i] > 1e-9, w)
            score = info.score / 100.0
            pl = max(0.0, min(c.max_power_kw, pk.limit_kw))
            p_req = required_power_kw(s.required_energy_kwh, t_rem)
            ratio = 2.0 if (pl <= 1e-9 or not math.isfinite(p_req)) else min(2.0, p_req / pl)
        else:
            score, ratio = 0.0, 0.0
        f = [float(chargeable), float(arrived), float(s.target_reached), s.soc / 100.0, c.target_soc / 100.0,
             min(1.0, s.required_energy_kwh / c.battery_kwh), min(t_rem, T_SCALE_H) / T_SCALE_H,
             min(c.max_power_kw / 25.0, 1.5), ratio / 2.0, score, core.prev_alloc[i] / c.max_power_kw,
             pk.caps_kw[i] / c.max_power_kw]
        obs[i * EV_FEATURES:(i + 1) * EV_FEATURES] = f
    k, n = pk.step, st.n_steps
    rem_t = [st.tariff_at(j) for j in range(k, n)]
    nxt = max(1, int(round(1.0 / st.dt_h)))
    sol_next = np.mean([st.solar_at(j) for j in range(k, min(n, k + nxt))])
    g = [st.station_power_kw / P_SCALE_KW, pk.solar_kw / P_SCALE_KW, st.grid_limit_kw / P_SCALE_KW,
         pk.tariff / TARIFF_SCALE, float(np.mean(rem_t)) / TARIFF_SCALE, float(np.min(rem_t)) / TARIFF_SCALE,
         float(sol_next) / P_SCALE_KW, k / n, core.n / max_evs]
    obs[max_evs * EV_FEATURES:] = g
    return np.clip(obs, 0.0, 3.0)


class ChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario | None = None,
                 sampler: Callable[[np.random.Generator], Scenario] | None = None,
                 max_evs: int = MAX_EVS, record: bool = False):
        super().__init__()
        if scenario is None and sampler is None:
            sampler = lambda rng: random_scenario(rng, max_evs)
        self.fixed, self.sampler, self.max_evs, self.record = scenario, sampler, max_evs, record
        if scenario is not None and len(scenario.evs) > max_evs:
            raise ValueError(f"Scenario has {len(scenario.evs)} EVs; this environment supports at most {max_evs}.")
        self.observation_space = spaces.Box(0.0, 3.0, shape=(max_evs * EV_FEATURES + GLOBAL_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(max_evs,), dtype=np.float32)
        self.core: ChargingCore | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sc = (options or {}).get("scenario") or self.fixed or self.sampler(self.np_random)
        self.core = ChargingCore(sc, record=self.record)
        self.R = max(float(self.core.initial_required.sum()), 1e-6)
        self.tref = max(float(np.mean(sc.station.tariff_per_kwh)), 1e-6)
        return build_observation(self.core, self.max_evs), {}

    def step(self, action):
        core = self.core
        a = np.clip(np.asarray(action, dtype=float).ravel(), 0.0, 1.0)[: core.n]
        pmax = np.array([c.max_power_kw for c in core.cfgs])
        req = np.zeros(core.n)
        req[: a.size] = a * pmax[: a.size]
        info = core.step(req)
        R = self.R
        r = info.delivered_kwh / R - 0.5 * info.cost_rs / (R * self.tref) + 0.1 * info.solar_kwh / R
        r += 0.1 * len(info.newly_completed)
        for i in info.newly_missed:
            r -= 1.0 + 3.0 * core.states[i].required_energy_kwh / R
        r -= 0.05 * info.limit_excess_kw / max(core.station.station_power_kw, 1e-6)
        obs = build_observation(core, self.max_evs)
        return obs, float(r), bool(info.done), False, {"cost_rs": info.cost_rs, "delivered_kwh": info.delivered_kwh}
