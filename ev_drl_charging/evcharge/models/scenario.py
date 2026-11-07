"""Scenario = EV list + station. Kept separate so the same EVs can be reused across stations."""
from __future__ import annotations

from dataclasses import dataclass

from .ev import EVConfig, VehicleType
from .station import StationConfig, bell_solar_profile, time_of_use_tariff


@dataclass
class Scenario:
    evs: list[EVConfig]
    station: StationConfig

    def validate(self) -> list[str]:
        errors = self.station.validate()
        ids = [e.ev_id for e in self.evs]
        if len(set(ids)) != len(ids):
            errors.append("Vehicle IDs must be unique.")
        if not self.evs:
            errors.append("At least one EV is required.")
        for ev in self.evs:
            errors += ev.validate()
            if self.station.duration_h and ev.arrival_h >= self.station.duration_h:
                errors.append(f"{ev.ev_id}: arrival is after the simulation ends.")
            if self.station.duration_h and ev.departure_h > self.station.duration_h + 1e-9:
                errors.append(f"{ev.ev_id}: departure is after the simulation ends.")
        return errors


def default_evs() -> list[EVConfig]:
    """Illustrative demo vehicles (NOT universal specs). Times in hours since start."""
    return [
        EVConfig("SCOOTER-01", VehicleType.SCOOTER, 3.5, 20, 90, 1.5, arrival_h=0.0, departure_h=6.0),
        EVConfig("SCOOTER-02", VehicleType.SCOOTER, 3.5, 30, 90, 1.5, arrival_h=1.0, departure_h=7.0),
        EVConfig("CAR-01", VehicleType.CAR, 40.0, 20, 80, 7.0, arrival_h=0.5, departure_h=8.0),
    ]


def default_station(start_clock_h: float = 8.0) -> StationConfig:
    """Demo station: 10 kW chargers, 8 kW grid limit, 4 kW peak solar, time-of-use tariff."""
    st = StationConfig(station_power_kw=10.0, grid_limit_kw=8.0, interval_min=15, duration_h=12.0)
    n = st.n_steps
    # Simulation t=0 corresponds to start_clock_h on the wall clock; solar curve is shifted accordingly.
    st.solar_kw = bell_solar_profile(4.0, n, st.dt_h, sunrise_h=6.0 - start_clock_h, sunset_h=18.0 - start_clock_h)
    st.tariff_per_kwh = time_of_use_tariff(n, st.dt_h, start_clock_h=start_clock_h)
    return st


def default_scenario() -> Scenario:
    return Scenario(default_evs(), default_station())
