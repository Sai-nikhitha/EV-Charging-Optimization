"""Random scenario generator used for PPO training and held-out evaluation.

Varies: number of active EVs (1..max_evs), vehicle types/batteries/powers, initial and target SOC,
arrival times and deadlines (some physically infeasible), station capacity, grid limit, solar
profile (none/constant/bell), tariff profile (fixed or time-of-use) and the charging interval.
All values are illustrative simulation ranges, not measurements of any real station.
"""
from __future__ import annotations

import numpy as np

from .models.ev import EVConfig, VehicleType
from .models.scenario import Scenario
from .models.station import StationConfig, bell_solar_profile, constant_profile, time_of_use_tariff

INTERVALS_MIN = (5, 10, 15, 30)
DURATIONS_H = (3.0, 4.0, 5.0, 6.0)
_Q = 1.0 / 12.0   # times are rounded to 5 minutes


def _q(x: float) -> float:
    return round(x / _Q) * _Q


def random_scenario(rng: np.random.Generator, max_evs: int = 5, n_evs: int | None = None) -> Scenario:
    n = int(n_evs if n_evs is not None else rng.integers(1, max_evs + 1))
    duration = float(rng.choice(DURATIONS_H))
    interval = int(rng.choice(INTERVALS_MIN))
    evs: list[EVConfig] = []
    for i in range(n):
        is_car = rng.random() < 0.4
        if is_car:
            cap, pmax, vt, tag = rng.uniform(20, 80), rng.uniform(3.3, 22.0), VehicleType.CAR, "CAR"
        else:
            cap, pmax, vt, tag = rng.uniform(1.5, 5.0), rng.uniform(0.8, 3.0), VehicleType.SCOOTER, "SCOOTER"
        soc0 = rng.uniform(5, 70)
        tgt = min(100.0, soc0 + rng.uniform(10, 90))
        e_req = (tgt - soc0) / 100 * cap
        arrival = _q(rng.uniform(0, 0.5 * duration))
        # window = (minimum time at full power) * factor; factor < 1 => physically infeasible
        factor = rng.choice([rng.uniform(0.7, 1.0), rng.uniform(1.0, 5.0)], p=[0.12, 0.88])
        window = max(_Q * 2, e_req / pmax * factor)
        departure = min(duration, _q(arrival + window))
        if departure <= arrival + _Q / 2:
            arrival = max(0.0, _q(departure - 2 * _Q))
            departure = max(departure, arrival + 2 * _Q)
        evs.append(EVConfig(f"{tag}-{i + 1:02d}", vt, round(cap, 2), round(soc0, 1), round(tgt, 1),
                            round(pmax, 2), arrival, min(duration, departure)))
    psum = sum(e.max_power_kw for e in evs)
    station_kw = max(2.0, round(psum * rng.uniform(0.35, 1.1), 1))
    grid_kw = max(1.0, round(station_kw * rng.uniform(0.5, 1.0), 1))
    st = StationConfig(station_kw, grid_kw, interval, duration)
    nst, dt = st.n_steps, st.dt_h
    u = rng.random()
    if u < 0.25:
        st.solar_kw = []
    elif u < 0.45:
        st.solar_kw = constant_profile(round(station_kw * rng.uniform(0.05, 0.5), 2), nst)
    else:
        sunrise = rng.uniform(-2.0, 0.5 * duration)
        st.solar_kw = bell_solar_profile(round(station_kw * rng.uniform(0.1, 0.9), 2), nst, dt,
                                         sunrise, sunrise + rng.uniform(4, 10))
    if rng.random() < 0.4:
        st.tariff_per_kwh = constant_profile(round(rng.uniform(5, 12), 2), nst)
    else:
        lo = rng.uniform(3, 6)
        mid = lo + rng.uniform(1, 3)
        hi = mid + rng.uniform(2, 5)
        st.tariff_per_kwh = time_of_use_tariff(nst, dt, rng.uniform(0, 24), round(lo, 2), round(mid, 2), round(hi, 2))
    sc = Scenario(evs, st)
    assert not sc.validate(), sc.validate()
    return sc
