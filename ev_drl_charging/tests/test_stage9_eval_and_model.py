import numpy as np
import pandas as pd
import pytest

from evcharge.compare import compare_results
from evcharge.evaluate import evaluate, summarize
from evcharge.models import default_scenario
from evcharge.simulator import MODEL_DIR, load_model, run_baseline, run_drl

MODELS = sorted(MODEL_DIR.glob("ppo_seed*.zip"))
needs_model = pytest.mark.skipif(not MODELS, reason="no trained PPO model in models/")


def test_summarize_reports_variability_and_service_columns():
    df = pd.DataFrame({
        "model": ["m"] * 3, "baseline_cost": [100, 50, 0.0], "drl_cost": [90, 60, 1.0],
        "savings_rs": [10, -10, -1.0], "savings_pct": [10.0, -20.0, np.nan],
        "delivered_diff_kwh": [0, -1, 0], "targets_diff": [0, -1, 0], "miss_diff": [0, 1, 0],
        "cheaper_but_worse_service": [False, False, False],
    })
    s = summarize(df).iloc[0]
    assert s["scenarios"] == 3 and s["drl_cheaper_pct_of_scenarios"] == pytest.approx(100 / 3)
    assert s["drl_costlier_pct_of_scenarios"] == pytest.approx(200 / 3)
    assert s["mean_savings_pct"] == pytest.approx(-5.0)       # zero-baseline scenario excluded, not counted as 0 %
    assert "std_savings_rs" in s.index and "mean_miss_diff" in s.index


@needs_model
def test_trained_model_runs_demo_and_respects_limits():
    model = load_model(MODELS[0])
    sc = default_scenario()
    d = run_drl(sc, model)
    assert d.signature == run_baseline(sc).signature
    stn = d.station
    assert (stn["Total power (kW)"] <= stn["Deliverable limit (kW)"] + 1e-6).all()
    for c in sc.evs:
        g = d.schedule[d.schedule["EV ID"] == c.ev_id]
        assert (g["Allocated power (kW)"] <= c.max_power_kw + 1e-6).all()
        assert d.summary["final_soc"][c.ev_id] <= c.target_soc + 1e-6


@needs_model
def test_trained_model_observation_action_shapes_match():
    model = load_model(MODELS[0])
    from evcharge.env import OBS_DIM, MAX_EVS
    assert model.observation_space.shape == (OBS_DIM,) and model.action_space.shape == (MAX_EVS,)


@needs_model
def test_evaluate_end_to_end_small():
    df = evaluate(MODELS[:1], n_scenarios=4, seed=5)
    assert len(df) == 4 and df["baseline_cost"].ge(0).all()
    assert np.allclose(df["savings_rs"], df["baseline_cost"] - df["drl_cost"])
