"""Stage 8 - baseline vs DRL comparison (all numbers come from SimResult summaries)."""
from __future__ import annotations

import math

import pandas as pd

from .simulator import SimResult

TOL_KWH = 1e-3


def cost_savings(baseline_cost: float, drl_cost: float) -> tuple[float, float | None]:
    """Cost_Savings = Baseline - DRL (negative => DRL costs more).
    Percent = savings / baseline * 100, or None when the baseline cost is zero (undefined)."""
    savings = baseline_cost - drl_cost
    pct = None if abs(baseline_cost) < 1e-12 else savings / baseline_cost * 100.0
    return savings, pct


def compare_results(base: SimResult, drl: SimResult) -> dict:
    if base.signature != drl.signature:
        raise ValueError("Baseline and DRL results come from different scenario inputs; refusing to compare.")
    b, d = base.summary, drl.summary
    savings, pct = cost_savings(b["cost_rs"], d["cost_rs"])
    d_deliv = d["delivered_kwh"] - b["delivered_kwh"]
    d_targets = d["targets_reached"] - b["targets_reached"]
    d_miss = d["deadline_misses"] - b["deadline_misses"]
    worse_service = (d_deliv < -TOL_KWH) or d_targets < 0 or d_miss > 0
    return {
        "baseline_cost": b["cost_rs"], "drl_cost": d["cost_rs"], "savings_rs": savings, "savings_pct": pct,
        "delivered_diff_kwh": d_deliv, "targets_diff": d_targets, "deadline_miss_diff": d_miss,
        "grid_diff_kwh": d["grid_kwh"] - b["grid_kwh"], "solar_diff_kwh": d["solar_kwh"] - b["solar_kwh"],
        "baseline_cost_per_kwh": b["cost_rs"] / b["delivered_kwh"] if b["delivered_kwh"] > TOL_KWH else None,
        "drl_cost_per_kwh": d["cost_rs"] / d["delivered_kwh"] if d["delivered_kwh"] > TOL_KWH else None,
        "cheaper_but_worse_service": savings > 0 and worse_service,
        "verdict": verdict(savings, worse_service),
    }


def verdict(savings: float, worse_service: bool) -> str:
    if savings > 1e-9 and worse_service:
        return "DRL is cheaper BUT delivers less energy / meets fewer targets or deadlines - not a like-for-like win."
    if savings > 1e-9:
        return "DRL is cheaper with equal or better service in this scenario."
    if savings < -1e-9:
        return "DRL costs MORE than the baseline in this scenario" + (" and service is also worse." if worse_service else ".")
    return "Costs are equal in this scenario."


def comparison_table(base: SimResult, drl: SimResult) -> pd.DataFrame:
    b, d = base.summary, drl.summary
    n = b["n_evs"]
    rows = [
        ("Grid energy (kWh)", b["grid_kwh"], d["grid_kwh"]),
        ("Solar energy used (kWh)", b["solar_kwh"], d["solar_kwh"]),
        ("Energy delivered to batteries (kWh)", b["delivered_kwh"], d["delivered_kwh"]),
        ("Unmet required energy (kWh)", b["unmet_kwh"], d["unmet_kwh"]),
        ("Grid electricity cost (Rs)", b["cost_rs"], d["cost_rs"]),
        (f"EVs reaching target SOC (of {n})", b["targets_reached"], d["targets_reached"]),
        (f"EVs meeting departure deadline (of {n})", b["deadlines_met"], d["deadlines_met"]),
        ("Deadline misses", b["deadline_misses"], d["deadline_misses"]),
        ("Station power utilization (%)", b["station_utilization_pct"], d["station_utilization_pct"]),
    ]
    for ev in b["final_soc"]:
        rows.append((f"Final SOC {ev} (%)", b["final_soc"][ev], d["final_soc"][ev]))
    df = pd.DataFrame(rows, columns=["Metric", "Baseline", "DRL (PPO)"])
    df["Difference (DRL - Baseline)"] = df["DRL (PPO)"] - df["Baseline"]
    return df
