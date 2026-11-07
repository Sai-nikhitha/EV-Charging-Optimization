"""Stage 3 - Dynamic priority engine.

Definitions (t = current time, all in hours / kW / kWh):
  E_req   = max(0, (target_SOC - SOC) / 100 * capacity)
  T_rem   = max(0, departure - max(t, arrival))      (0 => deadline reached)
  P_req   = E_req / T_rem                            (inf if E_req > 0 and T_rem = 0)
  P_limit = min(P_max, station power available now)  (what the EV could receive at best)

Feasibility (physical question: *can* the target be met?)
  Target reached | Missed deadline | Infeasible (EV power limit: P_req > P_max)
  | At risk (station limit: P_req > P_limit) | Tight (P_req > 0.85 P_limit) | Feasible
  NOTE: P_limit assumes this EV could use all available station power; competition with other
  EVs can make things worse, which only the allocation step reveals.

Priority score (urgency question: *how much should this EV be served now?*), 0..100:
  score = 100 * ( w_urg * U + w_need * N + w_time * T + w_cont * C )
    U = min(1, P_req / P_limit)       deadline pressure (1 if infeasible)      w_urg  = 0.50
    N = min(1, E_req / battery_kWh)   share of battery still to fill           w_need = 0.20
    T = 1 - min(1, T_rem / t_ref)     closeness of departure (t_ref = 4 h)     w_time = 0.20
    C = 1 if the EV was charging in the previous interval else 0   continuity  w_cont = 0.10
  Classes: Critical >= 70, High >= 50, Medium >= 30, Low otherwise.
Weights are not fitted to data; they are documented defaults, tunable via PriorityWeights.
Urgency and feasibility are deliberately separate: an Infeasible EV still gets a high score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .models.ev import EVState


@dataclass(frozen=True)
class PriorityWeights:
    urgency: float = 0.50
    need: float = 0.20
    time: float = 0.20
    continuity: float = 0.10
    t_ref_h: float = 4.0
    critical: float = 70.0
    high: float = 50.0
    medium: float = 30.0
    tight_ratio: float = 0.85


@dataclass
class PriorityInfo:
    ev_id: str
    vehicle_type: str
    arrival_h: float
    soc: float
    target_soc: float
    status: str            # Not arrived | Active | Completed | Departed
    e_req_kwh: float
    t_rem_h: float
    p_req_kw: float
    p_max_kw: float
    urgency: float
    score: float
    priority_class: str
    feasibility: str
    explanation: str


def required_power_kw(e_req_kwh: float, t_rem_h: float) -> float:
    if e_req_kwh <= 1e-9:
        return 0.0
    if t_rem_h <= 1e-9:
        return math.inf
    return e_req_kwh / t_rem_h


def classify(score: float, w: PriorityWeights) -> str:
    return "Critical" if score >= w.critical else "High" if score >= w.high else "Medium" if score >= w.medium else "Low"


def evaluate_priority(state: EVState, t_h: float, available_station_kw: float,
                      was_charging: bool = False, w: PriorityWeights = PriorityWeights()) -> PriorityInfo:
    cfg = state.config
    e_req = state.required_energy_kwh
    t_rem = max(0.0, cfg.departure_h - max(t_h, cfg.arrival_h))
    p_req = required_power_kw(e_req, t_rem)
    p_limit = max(0.0, min(cfg.max_power_kw, available_station_kw))

    if t_h >= cfg.departure_h - 1e-9:
        status = "Departed"
    elif t_h < cfg.arrival_h - 1e-9:
        status = "Not arrived"
    elif state.target_reached:
        status = "Completed"
    else:
        status = "Active"

    # ---- feasibility ----
    if state.target_reached:
        feas = "Target reached"
    elif status == "Departed":
        feas = "Missed deadline"
    elif p_req > cfg.max_power_kw + 1e-9:
        feas = "Infeasible (EV power limit)"
    elif p_req > p_limit + 1e-9:
        feas = "At risk (station limit)"
    elif p_req > w.tight_ratio * p_limit:
        feas = "Tight"
    else:
        feas = "Feasible"

    # ---- urgency / score ----
    if status != "Active":
        u, score, cls = 0.0, 0.0, ("Pending" if status == "Not arrived" else "—")
    else:
        u = 1.0 if (p_limit <= 1e-9 or not math.isfinite(p_req)) else min(1.0, p_req / p_limit)
        need = min(1.0, e_req / cfg.battery_kwh)
        tm = 1.0 - min(1.0, t_rem / w.t_ref_h)
        cont = 1.0 if was_charging else 0.0
        score = 100.0 * (w.urgency * u + w.need * need + w.time * tm + w.continuity * cont)
        cls = classify(score, w)

    # ---- short explanation ----
    if status == "Active":
        pr = "inf" if not math.isfinite(p_req) else f"{p_req:.2f}"
        expl = (f"Needs {e_req:.2f} kWh in {t_rem * 60:.0f} min -> {pr} kW avg vs limit {p_limit:.2f} kW "
                f"(urgency {u:.2f}); {feas.lower()}.")
    elif status == "Not arrived":
        expl = f"Arrives at {cfg.arrival_h * 60:.0f} min; not eligible for charging yet."
    elif status == "Completed":
        expl = "Target SOC reached; no further charging."
    else:
        expl = "Departed" + ("; target not reached." if not state.target_reached else ".")
    return PriorityInfo(cfg.ev_id, cfg.vehicle_type.value, cfg.arrival_h, state.soc, cfg.target_soc, status,
                        e_req, t_rem, p_req, cfg.max_power_kw, u, score, cls, feas, expl)


def priority_table(states: list[EVState], t_h: float, available_station_kw: float,
                   prev_alloc_kw: dict[str, float] | None = None,
                   w: PriorityWeights = PriorityWeights()) -> pd.DataFrame:
    prev_alloc_kw = prev_alloc_kw or {}
    rows = []
    for s in states:
        i = evaluate_priority(s, t_h, available_station_kw, prev_alloc_kw.get(s.config.ev_id, 0.0) > 1e-9, w)
        rows.append({
            "EV ID": i.ev_id, "Type": i.vehicle_type, "Arrival (min)": round(i.arrival_h * 60, 1),
            "Status": i.status, "Current SOC (%)": round(i.soc, 1), "Target SOC (%)": i.target_soc,
            "Required energy (kWh)": round(i.e_req_kwh, 3), "Remaining time (min)": round(i.t_rem_h * 60, 1),
            "Required avg power (kW)": (round(i.p_req_kw, 3) if math.isfinite(i.p_req_kw) else float("inf")),
            "Max power (kW)": i.p_max_kw, "Priority score": round(i.score, 1), "Priority class": i.priority_class,
            "Feasibility": i.feasibility, "Explanation": i.explanation,
        })
    return pd.DataFrame(rows)
