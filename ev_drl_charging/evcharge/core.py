"""Stage 7 - interval-based charging simulation core (used by baseline, Gym env and the UI).

One call to `ChargingCore.step(requests_kw)` performs, for the interval [t, t+dt):
  arrivals/active set -> solar, grid limit, tariff -> priority inputs (via `peek`) -> requests
  -> allocation (EV, station, grid limits) -> solar/grid split -> cost -> SOC update
  -> target/deadline checks -> schedule rows.

Reallocation policy: allocations are recomputed at every interval boundary. An arrival or
departure therefore changes the *next* allocation only; EVs already charging keep being served
(the baseline's continuity term and the allocator never interrupt an EV for no reason).
Arrival/departure inside an interval: the EV is charged only for the overlapping hours.
No charging losses are modeled (see energy.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import energy
from .allocation import allocate_power
from .models.ev import EVState
from .models.scenario import Scenario

EPS = 1e-9


def scenario_signature(sc: Scenario) -> str:
    """Stable hash of every scenario input; used to prove baseline and DRL saw identical inputs."""
    st = sc.station
    payload = {
        "evs": [[e.ev_id, e.vehicle_type.value, e.battery_kwh, e.initial_soc, e.target_soc,
                 e.max_power_kw, e.arrival_h, e.departure_h] for e in sc.evs],
        "station": [st.station_power_kw, st.grid_limit_kw, st.interval_min, st.duration_h,
                    list(st.solar_kw), list(st.tariff_per_kwh)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Peek:
    """Everything a policy may look at before deciding (no future leakage except profiles)."""
    step: int
    t0: float
    t1: float
    overlap_h: np.ndarray     # hours each EV is present within this interval
    caps_kw: np.ndarray       # max power each EV can take now (0 if absent/complete)
    limit_kw: float           # max total deliverable power now
    solar_kw: float
    tariff: float


@dataclass
class StepInfo:
    cost_rs: float
    delivered_kwh: float
    solar_kwh: float
    grid_kwh: float
    limit_excess_kw: float
    alloc_kw: np.ndarray
    ev_energy_kwh: np.ndarray
    newly_completed: list[int]
    newly_missed: list[int]
    done: bool


class ChargingCore:
    def __init__(self, scenario: Scenario, record: bool = True):
        errs = scenario.validate()
        if errs:
            raise ValueError("Invalid scenario: " + "; ".join(errs))
        self.scenario = scenario
        self.station = scenario.station
        self.cfgs = list(scenario.evs)
        self.n = len(self.cfgs)
        self.record = record
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.k = 0
        self.states = [EVState(c) for c in self.cfgs]
        self.initial_required = np.array([s.required_energy_kwh for s in self.states])
        self.prev_alloc = np.zeros(self.n)
        self.rows: list[dict] = []
        self.step_rows: list[dict] = []
        self.missed = [False] * self.n
        self.completed_flag = [s.target_reached for s in self.states]
        self.tot_solar = self.tot_grid = self.tot_cost = self.tot_delivered = 0.0
        self.tot_excess = 0.0

    @property
    def done(self) -> bool:
        return self.k >= self.station.n_steps

    # ------------------------------------------------------------------ policy view
    def peek(self) -> Peek:
        st, k = self.station, self.k
        dt = st.dt_h
        t0, t1 = k * dt, (k + 1) * dt
        overlap = np.array([max(0.0, min(t1, c.departure_h) - max(t0, c.arrival_h)) for c in self.cfgs])
        overlap[overlap < EPS] = 0.0
        required = np.array([s.required_energy_kwh for s in self.states])
        pmax = np.array([c.max_power_kw for c in self.cfgs])
        with np.errstate(divide="ignore", invalid="ignore"):
            energy_cap = np.where(overlap > 0, required / np.where(overlap > 0, overlap, 1.0), 0.0)
        caps = np.where(overlap > 0, np.minimum(pmax, energy_cap), 0.0)
        return Peek(k, t0, t1, overlap, caps, st.max_deliverable_kw(k), st.solar_at(k), st.tariff_at(k))

    # ------------------------------------------------------------------ one interval
    def step(self, requests_kw) -> StepInfo:
        if self.done:
            raise RuntimeError("Simulation already finished.")
        pk = self.peek()
        st, dt = self.station, self.station.dt_h
        req = np.zeros(self.n)
        r = np.asarray(requests_kw, dtype=float).ravel()
        req[:min(self.n, r.size)] = r[:self.n]

        alloc_res = allocate_power(req, pk.caps_kw, pk.limit_kw)
        alloc = alloc_res.allocated_kw
        ev_energy_raw = alloc * pk.overlap_h
        soc_start = [s.soc for s in self.states]
        status_start = [self._status(i, pk) for i in range(self.n)]
        rem_start = [max(0.0, c.departure_h - max(pk.t0, c.arrival_h)) for c in self.cfgs]

        stored = np.array([s.add_energy(float(e)) for s, e in zip(self.states, ev_energy_raw)])
        total_e = float(stored.sum())
        split = energy.split_solar_grid(total_e / dt, pk.solar_kw, dt)
        cost = energy.interval_cost_rs(split.grid_kwh, pk.tariff)
        share = split.solar_kwh / total_e if total_e > EPS else 0.0
        per_ev = energy.split_ev_energy({str(i): float(stored[i]) for i in range(self.n)}, share)

        newly_completed, newly_missed = [], []
        for i, s in enumerate(self.states):
            if s.target_reached and not self.completed_flag[i]:
                self.completed_flag[i] = True
                newly_completed.append(i)
            if (not s.target_reached) and (not self.missed[i]) and self.cfgs[i].departure_h <= pk.t1 + EPS:
                self.missed[i] = True
                newly_missed.append(i)

        self.tot_solar += split.solar_kwh
        self.tot_grid += split.grid_kwh
        self.tot_cost += cost
        self.tot_delivered += total_e
        self.tot_excess += alloc_res.limit_excess_kw
        self.prev_alloc = alloc.copy()

        if self.record:
            for i, s in enumerate(self.states):
                c = self.cfgs[i]
                tgt = "Reached" if s.target_reached else "Not reached"
                dl = "Met" if s.target_reached else ("Missed" if pk.t1 >= c.departure_h - EPS else "Pending")
                self.rows.append({
                    "Step": pk.step, "Time (min)": round(pk.t0 * 60, 3), "EV ID": c.ev_id,
                    "Vehicle type": c.vehicle_type.value, "Status": status_start[i],
                    "SOC start (%)": round(soc_start[i], 4), "Requested power (kW)": round(float(req[i]), 4),
                    "Allocated power (kW)": round(float(alloc[i]), 4),
                    "Solar energy (kWh)": round(per_ev[str(i)].solar_kwh, 5),
                    "Grid energy (kWh)": round(per_ev[str(i)].grid_kwh, 5),
                    "Tariff (Rs/kWh)": pk.tariff,
                    "Interval cost (Rs)": round(per_ev[str(i)].grid_kwh * pk.tariff, 5),
                    "SOC updated (%)": round(s.soc, 4), "Remaining time (min)": round(rem_start[i] * 60, 3),
                    "Target status": tgt, "Deadline status": dl,
                })
            self.step_rows.append({
                "Step": pk.step, "Time (min)": round(pk.t0 * 60, 3), "Total power (kW)": float(alloc.sum()),
                "Station capacity (kW)": st.station_power_kw, "Deliverable limit (kW)": pk.limit_kw,
                "Solar available (kW)": pk.solar_kw, "Solar energy (kWh)": split.solar_kwh,
                "Grid energy (kWh)": split.grid_kwh, "Tariff (Rs/kWh)": pk.tariff, "Interval cost (Rs)": cost,
            })
        self.k += 1
        return StepInfo(cost, total_e, split.solar_kwh, split.grid_kwh, alloc_res.limit_excess_kw,
                        alloc, stored, newly_completed, newly_missed, self.done)

    def _status(self, i: int, pk: Peek) -> str:
        c, s = self.cfgs[i], self.states[i]
        if c.arrival_h >= pk.t1 - EPS:
            return "Not arrived"
        if c.departure_h <= pk.t0 + EPS:
            return "Departed"
        return "Completed" if s.target_reached else "Active"

    # ------------------------------------------------------------------ outputs
    def schedule_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def station_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.step_rows)

    def summary(self) -> dict:
        st = self.station
        reached = [s.target_reached for s in self.states]
        n = self.n
        util = (pd.DataFrame(self.step_rows)["Total power (kW)"].mean() / st.station_power_kw * 100
                if self.step_rows else float("nan"))
        solar_avail = sum(st.solar_at(k) for k in range(st.n_steps)) * st.dt_h
        required_total = float(self.initial_required.sum())
        return {
            "grid_kwh": self.tot_grid, "solar_kwh": self.tot_solar, "delivered_kwh": self.tot_delivered,
            "cost_rs": self.tot_cost, "required_kwh": required_total,
            "unmet_kwh": float(sum(s.required_energy_kwh for s in self.states)),
            "n_evs": n, "targets_reached": int(sum(reached)), "targets_pct": 100.0 * sum(reached) / n,
            "deadlines_met": int(sum(reached)), "deadlines_pct": 100.0 * sum(reached) / n,
            "deadline_misses": int(n - sum(reached)),
            "final_soc": {s.config.ev_id: s.soc for s in self.states},
            "target_reached": {s.config.ev_id: s.target_reached for s in self.states},
            "station_utilization_pct": util,
            "solar_available_kwh": solar_avail,
            "solar_utilization_pct": (100.0 * self.tot_solar / solar_avail) if solar_avail > EPS else float("nan"),
            "limit_excess_kw_total": self.tot_excess,
            "completion": f"Completed {self.k}/{st.n_steps} intervals" if self.done
                          else f"Incomplete ({self.k}/{st.n_steps} intervals)",
        }
