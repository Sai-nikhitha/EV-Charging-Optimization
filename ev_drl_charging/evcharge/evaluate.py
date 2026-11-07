"""Stage 8 - multi-scenario, multi-seed evaluation on HELD-OUT random scenarios.

    python -m evcharge.evaluate --scenarios 100

Writes results/eval_scenarios.csv (one row per scenario x model) and results/eval_summary.csv.
Every number comes from running the baseline and PPO on the same scenario objects.
Scenarios are drawn with EVAL_SEED, which training never uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import compare_results
from .scenario_gen import random_scenario
from .simulator import MODEL_DIR, load_model, run_baseline, run_drl

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EVAL_SEED = 424_242   # fresh test seed (v1 was evaluated on 987654 before the reward was revised)


def evaluate(model_paths: list[Path], n_scenarios: int = 100, seed: int = EVAL_SEED) -> pd.DataFrame:
    models = {p.stem: load_model(p) for p in model_paths}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_scenarios):
        sc = random_scenario(rng)
        base = run_baseline(sc)
        for name, m in models.items():
            drl = run_drl(sc, m)
            c = compare_results(base, drl)
            rows.append({
                "scenario": i, "model": name, "n_evs": len(sc.evs),
                "baseline_cost": c["baseline_cost"], "drl_cost": c["drl_cost"],
                "savings_rs": c["savings_rs"], "savings_pct": c["savings_pct"],
                "baseline_delivered": base.summary["delivered_kwh"], "drl_delivered": drl.summary["delivered_kwh"],
                "delivered_diff_kwh": c["delivered_diff_kwh"],
                "baseline_targets": base.summary["targets_reached"], "drl_targets": drl.summary["targets_reached"],
                "targets_diff": c["targets_diff"],
                "baseline_misses": base.summary["deadline_misses"], "drl_misses": drl.summary["deadline_misses"],
                "miss_diff": c["deadline_miss_diff"],
                "cheaper_but_worse_service": c["cheaper_but_worse_service"],
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        pct = g["savings_pct"].dropna()
        # "service-matched": DRL delivered >= baseline energy, reached >= targets, missed <= deadlines
        ok = g[(g["delivered_diff_kwh"] >= -1e-3) & (g["targets_diff"] >= 0) & (g["miss_diff"] <= 0)]
        ok_pct = ok["savings_pct"].dropna()
        return pd.Series({
            "scenarios": len(g),
            "mean_baseline_cost": g["baseline_cost"].mean(), "mean_drl_cost": g["drl_cost"].mean(),
            "mean_savings_rs": g["savings_rs"].mean(), "std_savings_rs": g["savings_rs"].std(),
            "mean_savings_pct": pct.mean(), "median_savings_pct": pct.median(),
            "drl_cheaper_pct_of_scenarios": 100 * (g["savings_rs"] > 1e-9).mean(),
            "drl_costlier_pct_of_scenarios": 100 * (g["savings_rs"] < -1e-9).mean(),
            "mean_delivered_diff_kwh": g["delivered_diff_kwh"].mean(),
            "mean_targets_diff": g["targets_diff"].mean(),
            "mean_miss_diff": g["miss_diff"].mean(),
            "cheaper_but_worse_service_pct": 100 * g["cheaper_but_worse_service"].mean(),
            "service_matched_scenarios": len(ok),
            "service_matched_mean_savings_rs": ok["savings_rs"].mean() if len(ok) else float("nan"),
            "service_matched_mean_savings_pct": ok_pct.mean() if len(ok_pct) else float("nan"),
        })
    return df.groupby("model").apply(agg, include_groups=False).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    paths = [Path(p) for p in a.models] if a.models else sorted(MODEL_DIR.glob("ppo_seed*.zip"))
    RESULTS_DIR.mkdir(exist_ok=True)
    df = evaluate(paths, a.scenarios)
    df.to_csv(RESULTS_DIR / "eval_scenarios.csv", index=False)
    summ = summarize(df)
    summ.to_csv(RESULTS_DIR / "eval_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
