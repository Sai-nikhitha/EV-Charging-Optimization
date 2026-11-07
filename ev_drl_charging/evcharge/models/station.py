"""Charging-station model: capacity, grid limit, solar profile, tariff profile.

Terminology (kept deliberately separate):
  * station_power_kw : max total power the chargers can deliver to all EVs together
  * grid_limit_kw    : max power imported from the grid
  * solar_kw[t]      : solar power available in interval t (can't be exported/stored here)
  * EV max power     : per-vehicle limit (lives in EVConfig)

Deliverable power in interval t = min(station_power_kw, solar_kw[t] + grid_limit_kw).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StationConfig:
    station_power_kw: float
    grid_limit_kw: float
    interval_min: int = 15
    duration_h: float = 12.0
    solar_kw: list[float] = field(default_factory=list)      # one value per step; [] = no solar
    tariff_per_kwh: list[float] = field(default_factory=list)  # Rs/kWh per step

    # ---- time grid ----------------------------------------------------------
    @property
    def dt_h(self) -> float:
        return self.interval_min / 60.0

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_h / self.dt_h))

    def time_h(self, step: int) -> float:
        return step * self.dt_h

    # ---- per-step lookups (safe outside the range: last value is held) -------
    def solar_at(self, step: int) -> float:
        if not self.solar_kw:
            return 0.0
        return self.solar_kw[min(max(step, 0), len(self.solar_kw) - 1)]

    def tariff_at(self, step: int) -> float:
        idx = min(max(step, 0), len(self.tariff_per_kwh) - 1)
        return self.tariff_per_kwh[idx]

    def max_deliverable_kw(self, step: int) -> float:
        """Upper bound on total EV charging power at this step."""
        return min(self.station_power_kw, self.solar_at(step) + self.grid_limit_kw)

    def validate(self) -> list[str]:
        e: list[str] = []
        if self.station_power_kw <= 0:
            e.append("Station power capacity must be > 0 kW.")
        if self.grid_limit_kw < 0:
            e.append("Grid power limit must be >= 0 kW.")
        if self.interval_min <= 0 or 60 % self.interval_min != 0:
            e.append("Interval must be a positive divisor of 60 minutes (e.g. 5, 10, 15, 30, 60).")
        if self.duration_h <= 0:
            e.append("Simulation duration must be > 0 h.")
        elif self.interval_min > 0 and abs(self.duration_h / self.dt_h - round(self.duration_h / self.dt_h)) > 1e-9:
            e.append("Duration must be a whole number of intervals.")
        n = self.n_steps if not e else 0
        for name, prof in (("Solar", self.solar_kw), ("Tariff", self.tariff_per_kwh)):
            if any((not math.isfinite(x)) or x < 0 for x in prof):
                e.append(f"{name} profile must contain finite values >= 0.")
        if not self.tariff_per_kwh:
            e.append("Tariff profile must not be empty.")
        for name, prof in (("Solar", self.solar_kw), ("Tariff", self.tariff_per_kwh)):
            if prof and n and len(prof) not in (1, n):
                e.append(f"{name} profile length ({len(prof)}) must be 1 or {n} (one per interval).")
        return e

    # ---- profile builders (used by the UI and by training later) --------------
    def expand(self) -> None:
        """Broadcast length-1 profiles to one value per interval."""
        n = self.n_steps
        if len(self.solar_kw) == 1:
            self.solar_kw = self.solar_kw * n
        if len(self.tariff_per_kwh) == 1:
            self.tariff_per_kwh = self.tariff_per_kwh * n


def constant_profile(value: float, n_steps: int) -> list[float]:
    return [float(value)] * n_steps


def bell_solar_profile(peak_kw: float, n_steps: int, dt_h: float,
                       sunrise_h: float = 6.0, sunset_h: float = 18.0) -> list[float]:
    """Half-sine solar curve between sunrise and sunset (illustrative, not measured data)."""
    out = []
    for k in range(n_steps):
        t = (k + 0.5) * dt_h
        if sunrise_h <= t <= sunset_h:
            out.append(peak_kw * math.sin(math.pi * (t - sunrise_h) / (sunset_h - sunrise_h)))
        else:
            out.append(0.0)
    return out


def time_of_use_tariff(n_steps: int, dt_h: float, start_clock_h: float = 0.0,
                       off_peak: float = 5.0, standard: float = 8.0, peak: float = 11.0,
                       peak_windows: tuple[tuple[float, float], ...] = ((18.0, 22.0),),
                       off_peak_windows: tuple[tuple[float, float], ...] = ((22.0, 24.0), (0.0, 6.0))
                       ) -> list[float]:
    """Time-of-use tariff in Rs/kWh by clock hour (illustrative values, user-configurable)."""
    out = []
    for k in range(n_steps):
        clock = (start_clock_h + k * dt_h) % 24.0
        if any(a <= clock < b for a, b in peak_windows):
            out.append(peak)
        elif any(a <= clock < b for a, b in off_peak_windows):
            out.append(off_peak)
        else:
            out.append(standard)
    return out
