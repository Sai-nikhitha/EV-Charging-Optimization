import dataclasses

import pytest

from evcharge import energy
from evcharge.models import EVConfig, EVState, StationConfig, VehicleType, default_scenario, default_evs
from evcharge.models.station import bell_solar_profile, time_of_use_tariff


def make_ev(**kw):
    base = dict(ev_id="X", vehicle_type=VehicleType.CAR, battery_kwh=40, initial_soc=20,
                target_soc=80, max_power_kw=7, arrival_h=0, departure_h=5)
    base.update(kw)
    return EVConfig(**base)


# ---------- default scenario ----------
def test_default_scenario_is_two_scooters_one_car_and_valid():
    sc = default_scenario()
    assert [e.ev_id for e in sc.evs] == ["SCOOTER-01", "SCOOTER-02", "CAR-01"]
    assert [e.vehicle_type for e in sc.evs].count(VehicleType.SCOOTER) == 2
    assert [e.vehicle_type for e in sc.evs].count(VehicleType.CAR) == 1
    assert sc.validate() == []


def test_vehicles_are_independent_and_arrive_at_different_times():
    evs = default_evs()
    assert len({e.arrival_h for e in evs}) == 3
    changed = dataclasses.replace(evs[0], max_power_kw=2.0)
    assert evs[1].max_power_kw == 1.5 and changed.max_power_kw == 2.0


# ---------- EV validation ----------
@pytest.mark.parametrize("kw", [
    dict(battery_kwh=0), dict(initial_soc=-1), dict(target_soc=101),
    dict(initial_soc=90, target_soc=50), dict(max_power_kw=0),
    dict(arrival_h=-1), dict(departure_h=0), dict(ev_id=" "),
])
def test_invalid_ev_inputs_rejected(kw):
    assert make_ev(**kw).validate()


def test_valid_ev_has_no_errors():
    assert make_ev().validate() == []


# ---------- EV state / required energy / SOC ----------
def test_required_energy_matches_formula():
    s = EVState(make_ev())  # 20 -> 80 % of 40 kWh
    assert s.required_energy_kwh == pytest.approx(24.0)
    assert energy.required_energy_kwh(20, 80, 40) == pytest.approx(24.0)


def test_required_energy_never_negative():
    assert energy.required_energy_kwh(90, 80, 40) == 0.0


def test_soc_update_and_cap_at_target():
    s = EVState(make_ev())
    assert s.add_energy(12.0) == pytest.approx(12.0)
    assert s.soc == pytest.approx(50.0)
    stored = s.add_energy(100.0)             # would overshoot; capped at target
    assert stored == pytest.approx(12.0)
    assert s.soc == pytest.approx(80.0)
    assert s.target_reached and s.required_energy_kwh == 0.0
    assert s.add_energy(5.0) == 0.0          # no charging beyond target


def test_negative_energy_rejected():
    with pytest.raises(ValueError):
        EVState(make_ev()).add_energy(-1)


def test_presence_window_and_remaining_time():
    s = EVState(make_ev(arrival_h=1, departure_h=4))
    assert not s.is_present(0.5)
    assert s.is_present(1.0) and s.is_present(3.75)
    assert not s.is_present(4.0)             # departed
    assert s.remaining_time_h(3.0) == pytest.approx(1.0)
    assert s.remaining_time_h(5.0) == 0.0    # never negative


# ---------- station ----------
def test_station_time_grid_and_validation():
    st = default_scenario().station
    assert st.dt_h == 0.25 and st.n_steps == 48
    assert st.validate() == []


@pytest.mark.parametrize("kw", [
    dict(station_power_kw=0), dict(grid_limit_kw=-1), dict(interval_min=7), dict(duration_h=0),
    dict(tariff_per_kwh=[]), dict(tariff_per_kwh=[-1.0]), dict(solar_kw=[1.0, 2.0]),
])
def test_invalid_station_rejected(kw):
    base = dict(station_power_kw=10, grid_limit_kw=8, interval_min=15, duration_h=1.0,
                solar_kw=[], tariff_per_kwh=[8.0])
    base.update(kw)
    assert StationConfig(**base).validate()


def test_max_deliverable_is_min_of_station_and_solar_plus_grid():
    st = StationConfig(10, 8, solar_kw=[0.0, 5.0], tariff_per_kwh=[8.0], duration_h=0.5)
    assert st.max_deliverable_kw(0) == 8.0       # grid-limited
    assert st.max_deliverable_kw(1) == 10.0      # station-limited (5+8=13 > 10)


def test_profile_builders():
    solar = bell_solar_profile(4.0, 48, 0.25, 6, 18)
    assert len(solar) == 48 and max(solar) <= 4.0 and solar[0] == 0.0
    tariff = time_of_use_tariff(96, 0.25, start_clock_h=0)  # 24 h
    assert tariff[0] == 5.0 and tariff[19 * 4] == 11.0 and tariff[10 * 4] == 8.0


def test_scenario_duplicate_ids_and_out_of_range_times():
    sc = default_scenario()
    sc.evs.append(dataclasses.replace(sc.evs[0]))
    assert any("unique" in e for e in sc.validate())
    sc2 = default_scenario()
    sc2.evs[0] = dataclasses.replace(sc2.evs[0], departure_h=99)
    assert any("after the simulation ends" in e for e in sc2.validate())


# ---------- energy accounting ----------
def test_energy_from_power():
    assert energy.energy_from_power(4.0, 0.25) == pytest.approx(1.0)


def test_solar_first_split():
    sp = energy.split_solar_grid(6.0, 4.0, 0.5)
    assert sp.solar_kwh == pytest.approx(2.0) and sp.grid_kwh == pytest.approx(1.0)
    sp = energy.split_solar_grid(3.0, 4.0, 0.5)          # solar surplus: no grid
    assert sp.solar_kwh == pytest.approx(1.5) and sp.grid_kwh == 0.0


def test_solar_plus_grid_equals_total_energy():
    sp = energy.split_solar_grid(7.3, 2.1, 0.25)
    assert sp.solar_kwh + sp.grid_kwh == pytest.approx(7.3 * 0.25)


def test_cost_counts_only_grid_energy():
    sp = energy.split_solar_grid(6.0, 4.0, 0.5)
    assert energy.interval_cost_rs(sp.grid_kwh, 8.0) == pytest.approx(8.0)
    assert energy.interval_cost_rs(0.0, 11.0) == 0.0


def test_soc_after_charging_clipped():
    assert energy.soc_after_charging(50, 2.0, 4.0) == pytest.approx(100.0)
    assert energy.soc_after_charging(50, 1.0, 4.0) == pytest.approx(75.0)


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        energy.split_solar_grid(-1, 0, 0.25)
    with pytest.raises(ValueError):
        energy.interval_cost_rs(1, -2)


def test_split_ev_energy_conserves_energy():
    out = energy.split_ev_energy({"a": 1.0, "b": 3.0}, 0.25)
    assert sum(v.solar_kwh + v.grid_kwh for v in out.values()) == pytest.approx(4.0)
    assert sum(v.solar_kwh for v in out.values()) == pytest.approx(1.0)
