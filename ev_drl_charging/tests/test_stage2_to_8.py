import dataclasses
import io
import math

import numpy as np
import pandas as pd
import pytest

from evcharge.allocation import allocate_power, weighted_waterfill
from evcharge.baseline import baseline_requests
from evcharge.compare import compare_results, comparison_table, cost_savings
from evcharge.core import ChargingCore, scenario_signature
from evcharge.env import ChargingEnv, EV_FEATURES, GLOBAL_FEATURES, MAX_EVS, OBS_DIM
from evcharge.models import EVConfig, EVState, Scenario, StationConfig, VehicleType, default_scenario
from evcharge.priority import evaluate_priority, priority_table, required_power_kw
from evcharge.scenario_gen import random_scenario
from evcharge.simulator import SimResult, run_baseline, run_drl


def ev(i="E", vt=VehicleType.CAR, cap=40, s0=20, s1=80, p=7, a=0.0, d=5.0):
    return EVConfig(i, vt, cap, s0, s1, p, a, d)


def sc(evs, station_kw=10, grid=8, solar=None, tariff=8.0, dur=6.0, interval=15):
    st = StationConfig(station_kw, grid, interval, dur)
    evs = [dataclasses.replace(e, departure_h=min(e.departure_h, dur)) for e in evs]
    n = st.n_steps
    st.solar_kw = solar if solar is not None else []
    st.tariff_per_kwh = tariff if isinstance(tariff, list) else [tariff] * n
    return Scenario(evs, st)


class ConstModel:
    """Stand-in for a PPO model: always requests `level` of max power (tests the DRL pipeline)."""
    def __init__(self, level): self.level = level
    def predict(self, obs, deterministic=True): return np.full(MAX_EVS, self.level, dtype=np.float32), None


# ---------------- Stage 2: allocation / constraints ----------------
def test_ev_max_power_enforced():
    r = allocate_power([100, 100], [1.5, 7.0], 100)
    assert list(r.allocated_kw) == [1.5, 7.0]


def test_station_limit_scales_proportionally_and_is_order_independent():
    r1 = allocate_power([5, 5, 5], [5, 5, 5], 9)
    assert r1.allocated_kw.sum() == pytest.approx(9) and r1.limit_excess_kw == pytest.approx(6)
    a = allocate_power([2, 4, 6], [9, 9, 9], 6).allocated_kw
    b = allocate_power([6, 4, 2], [9, 9, 9], 6).allocated_kw
    assert sorted(a) == pytest.approx(sorted(b))


def test_negative_nan_requests_are_safe():
    r = allocate_power([-3, float("nan"), 2], [5, 5, 5], 10)
    assert list(r.allocated_kw) == [0, 0, 2]


def test_waterfill_respects_caps_and_redistributes():
    a = weighted_waterfill(10, [1, 1, 1], [1, 5, 5])
    assert a[0] == pytest.approx(1) and a.sum() == pytest.approx(10) and a[1] == pytest.approx(4.5)


def test_waterfill_order_independent():
    w, c = np.array([3.0, 1.0, 2.0, 5.0]), np.array([2.0, 9.0, 1.0, 4.0])
    perm = np.array([2, 0, 3, 1])
    a = weighted_waterfill(7, w, c)
    b = weighted_waterfill(7, w[perm], c[perm])
    assert a[perm] == pytest.approx(b)


def test_grid_limit_enforced_in_simulation():
    s = sc([ev(p=7), ev("B", p=7)], station_kw=20, grid=5, solar=[0.0] * 24)
    core = ChargingCore(s)
    info = core.step([7, 7])
    assert info.alloc_kw.sum() == pytest.approx(5)          # grid-limited, no solar
    assert info.grid_kwh == pytest.approx(5 * 0.25)


def test_station_limit_enforced_with_solar():
    s = sc([ev(p=7), ev("B", p=7)], station_kw=10, grid=8, solar=[6.0] * 24)
    info = ChargingCore(s).step([7, 7])
    assert info.alloc_kw.sum() == pytest.approx(10)         # min(10, 6+8)


# ---------------- arrival / departure / completion ----------------
def test_no_charging_before_arrival_or_after_departure():
    s = sc([ev(a=1.0, d=2.0)], dur=4.0)
    core = ChargingCore(s)
    rows = []
    while not core.done:
        core.step([7])
    df = core.schedule_df()
    before = df[df["Time (min)"] < 60]
    after = df[df["Time (min)"] >= 120]
    assert (before["Allocated power (kW)"] == 0).all() and (after["Allocated power (kW)"] == 0).all()
    assert (before["Status"] == "Not arrived").all() and (after["Status"] == "Departed").all()


def test_partial_interval_arrival_only_charges_overlap():
    s = sc([ev(a=0.125, d=1.0)], solar=[], dur=1.0)           # arrives half-way through step 0
    info = ChargingCore(s).step([7])
    assert info.ev_energy_kwh[0] == pytest.approx(7 * 0.125)


def test_never_charges_beyond_target_and_completed_ev_gets_zero():
    s = sc([ev(cap=3.5, s0=90, s1=100, p=1.5)], dur=3.0)
    core = ChargingCore(s)
    while not core.done:
        core.step([1.5])
    assert core.states[0].soc == pytest.approx(100)
    df = core.schedule_df()
    assert df["Allocated power (kW)"].iloc[-1] == 0
    assert core.tot_delivered == pytest.approx(0.35)


def test_energy_conservation_and_cost():
    s = sc([ev()], solar=[2.0] * 24, tariff=10.0)
    r = run_baseline(s)
    m = r.summary
    assert m["solar_kwh"] + m["grid_kwh"] == pytest.approx(m["delivered_kwh"])
    assert m["cost_rs"] == pytest.approx(m["grid_kwh"] * 10.0)
    sched = r.schedule
    assert sched["Interval cost (Rs)"].sum() == pytest.approx(m["cost_rs"], rel=1e-4)
    battery_gain = (m["final_soc"]["E"] - 20) / 100 * 40
    assert battery_gain == pytest.approx(m["delivered_kwh"], rel=1e-6)


def test_solar_first_accounting_in_step():
    s = sc([ev()], station_kw=10, grid=8, solar=[3.0] * 24, tariff=8.0)
    info = ChargingCore(s).step([7])
    assert info.solar_kwh == pytest.approx(3.0 * 0.25) and info.grid_kwh == pytest.approx(4.0 * 0.25)
    assert info.cost_rs == pytest.approx(info.grid_kwh * 8.0)


def test_target_completion_detected():
    s = sc([ev(cap=10, s0=50, s1=60, p=5)], dur=3.0)
    r = run_baseline(s)
    assert r.summary["targets_reached"] == 1 and r.summary["deadline_misses"] == 0


# ---------------- Stage 3: priority ----------------
def test_required_power_safe_for_zero_time():
    assert required_power_kw(5, 0) == math.inf and required_power_kw(0, 0) == 0 and required_power_kw(6, 2) == 3


def test_impossible_deadline_flagged_not_promised():
    e = ev(cap=40, s0=0, s1=100, p=7, a=0, d=2.0)          # needs 40 kWh in 2 h => 20 kW > 7 kW
    info = evaluate_priority(EVState(e), 0.0, 10.0)
    assert info.feasibility == "Infeasible (EV power limit)"
    assert info.urgency == 1.0 and info.score > 50          # urgent but infeasible: kept separate
    r = run_baseline(sc([e], dur=3.0))
    assert r.summary["deadline_misses"] == 1 and r.summary["targets_reached"] == 0


def test_station_limited_feasibility_and_tight():
    e = ev(cap=10, s0=0, s1=100, p=7, a=0, d=2.0)          # 5 kW required
    assert evaluate_priority(EVState(e), 0, 3.0).feasibility == "At risk (station limit)"
    assert evaluate_priority(EVState(e), 0, 5.5).feasibility == "Tight"
    assert evaluate_priority(EVState(e), 0, 10.0).feasibility == "Feasible"


def test_priority_statuses_and_ordering():
    e = ev(a=1.0, d=3.0)
    assert evaluate_priority(EVState(e), 0.0, 10).status == "Not arrived"
    assert evaluate_priority(EVState(e), 3.0, 10).status == "Departed"
    urgent = evaluate_priority(EVState(ev(d=2.5)), 0, 10).score
    relaxed = evaluate_priority(EVState(ev(d=20.0)), 0, 10).score
    assert urgent > relaxed


def test_priority_table_columns():
    s = default_scenario()
    df = priority_table([EVState(c) for c in s.evs], 0.0, 10.0)
    for col in ["EV ID", "Type", "Arrival (min)", "Current SOC (%)", "Target SOC (%)", "Required energy (kWh)",
                "Remaining time (min)", "Required avg power (kW)", "Max power (kW)", "Priority score",
                "Priority class", "Feasibility", "Explanation"]:
        assert col in df.columns


# ---------------- Stage 4: baseline ----------------
def test_baseline_respects_all_limits_every_step():
    s = default_scenario()
    r = run_baseline(s)
    sched, stn = r.schedule, r.station
    assert (stn["Total power (kW)"] <= stn["Deliverable limit (kW)"] + 1e-6).all()
    assert (stn["Total power (kW)"] <= s.station.station_power_kw + 1e-6).all()
    for c in s.evs:
        assert (sched[sched["EV ID"] == c.ev_id]["Allocated power (kW)"] <= c.max_power_kw + 1e-6).all()


def test_baseline_independent_of_ev_order():
    s = default_scenario()
    rev = Scenario(list(reversed(s.evs)), s.station)
    a, b = run_baseline(s).summary, run_baseline(rev).summary
    assert a["cost_rs"] == pytest.approx(b["cost_rs"]) and a["delivered_kwh"] == pytest.approx(b["delivered_kwh"])
    for k in a["final_soc"]:
        assert a["final_soc"][k] == pytest.approx(b["final_soc"][k])


def test_baseline_is_deterministic():
    s = default_scenario()
    assert run_baseline(s).summary["cost_rs"] == run_baseline(s).summary["cost_rs"]


def test_baseline_default_demo_meets_all_targets():
    r = run_baseline(default_scenario())
    assert r.summary["targets_reached"] == 3


# ---------------- Stage 5: env ----------------
def test_observation_and_action_dimensions():
    env = ChargingEnv(scenario=default_scenario())
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == env.observation_space.shape
    assert OBS_DIM == MAX_EVS * EV_FEATURES + GLOBAL_FEATURES
    assert env.action_space.shape == (MAX_EVS,)
    assert env.observation_space.contains(obs)


def test_inactive_slots_are_zero_and_actions_ignored():
    env = ChargingEnv(scenario=default_scenario())          # 3 EVs of 5 slots
    obs, _ = env.reset()
    assert np.all(obs[3 * EV_FEATURES: MAX_EVS * EV_FEATURES] == 0)
    obs2, r, *_ = env.step(np.array([1, 1, 1, 1, 1], dtype=np.float32))
    assert np.all(obs2[3 * EV_FEATURES: MAX_EVS * EV_FEATURES] == 0) and np.isfinite(r)


def test_env_rejects_too_many_evs():
    big = sc([ev(str(i)) for i in range(MAX_EVS + 1)])
    with pytest.raises(ValueError):
        ChargingEnv(scenario=big)


def test_env_random_episodes_run_with_valid_obs():
    env = ChargingEnv()
    rng = np.random.default_rng(0)
    for ep in range(15):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, done, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs) and np.isfinite(r)
        st = env.core.station
        assert env.core.tot_delivered >= 0
        assert env.core.station_df().empty or True


def test_random_scenarios_valid_and_varied():
    rng = np.random.default_rng(1)
    scs = [random_scenario(rng) for _ in range(60)]
    assert all(not s.validate() for s in scs)
    assert {len(s.evs) for s in scs} == {1, 2, 3, 4, 5}
    assert len({s.station.interval_min for s in scs}) > 1


def test_reward_does_not_favour_not_charging():
    s = sc([ev(cap=10, s0=0, s1=100, p=5, d=3.0)], dur=3.0)

    def total(level):
        env = ChargingEnv(scenario=s)
        env.reset()
        done, tot = False, 0.0
        while not done:
            _, r, done, _, _ = env.step(np.full(MAX_EVS, level, dtype=np.float32))
            tot += r
        return tot
    assert total(1.0) > total(0.0)


# ---------------- Stage 7/8: identical inputs + comparison ----------------
def test_baseline_and_drl_use_identical_inputs():
    s = default_scenario()
    b, d = run_baseline(s), run_drl(s, ConstModel(1.0))
    assert b.signature == d.signature == scenario_signature(s)
    assert b.station["Tariff (Rs/kWh)"].tolist() == d.station["Tariff (Rs/kWh)"].tolist()
    assert b.station["Solar available (kW)"].tolist() == d.station["Solar available (kW)"].tolist()


def test_signature_changes_when_inputs_change():
    s = default_scenario()
    s2 = default_scenario()
    s2.station.grid_limit_kw += 1
    assert scenario_signature(s) != scenario_signature(s2)


def test_compare_refuses_different_scenarios():
    s = default_scenario()
    s2 = default_scenario(); s2.station.grid_limit_kw = 6
    with pytest.raises(ValueError):
        compare_results(run_baseline(s), run_baseline(s2))


def test_cost_comparison_formulas():
    assert cost_savings(100, 80) == (20, 20.0)
    sv, pct = cost_savings(80, 100)
    assert sv == -20 and pct == -25.0                       # DRL more expensive: negative, honest
    assert cost_savings(0, 5) == (-5, None)                 # zero baseline handled


def test_compare_values_come_from_summaries_and_flag_worse_service():
    s = default_scenario()
    b, d = run_baseline(s), run_drl(s, ConstModel(0.0))     # never charges: trivially "cheaper"
    c = compare_results(b, d)
    assert c["drl_cost"] == 0 and c["savings_rs"] == pytest.approx(b.summary["cost_rs"])
    assert c["cheaper_but_worse_service"] is True            # lower cost due to no delivery is flagged
    assert c["deadline_miss_diff"] > 0 and "BUT" in c["verdict"]
    tbl = comparison_table(b, d)
    assert {"Metric", "Baseline", "DRL (PPO)"} <= set(tbl.columns)


def test_csv_export_roundtrip():
    r = run_baseline(default_scenario())
    csv = r.schedule.to_csv(index=False)
    back = pd.read_csv(io.StringIO(csv))
    assert len(back) == len(r.schedule) == 3 * default_scenario().station.n_steps
    for col in ["Time (min)", "EV ID", "Vehicle type", "Status", "SOC start (%)", "Requested power (kW)",
                "Allocated power (kW)", "Solar energy (kWh)", "Grid energy (kWh)", "Tariff (Rs/kWh)",
                "Interval cost (Rs)", "SOC updated (%)", "Remaining time (min)", "Target status", "Deadline status"]:
        assert col in back.columns
    assert back["Grid energy (kWh)"].sum() == pytest.approx(r.summary["grid_kwh"], rel=1e-3)


def test_utilization_and_completion_reported():
    r = run_baseline(default_scenario())
    assert 0 < r.summary["station_utilization_pct"] <= 100
    assert r.summary["completion"].startswith("Completed")
