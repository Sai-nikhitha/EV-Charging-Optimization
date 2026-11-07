"""Energy-accounting primitives.

Assumptions (documented, consistent across baseline and DRL):
  1. Charging losses are NOT modeled: energy stored in the battery == energy drawn
     from the charger (100 % efficiency). Easy to extend with an efficiency factor later.
  2. Power is constant within one interval, so energy_kWh = power_kW * dt_h.
  3. Solar-first policy: solar covers charging demand first; grid supplies the remainder.
     Unused solar is not stored or exported (no battery storage / feed-in in this project).
  4. Cost counts only grid energy: cost_Rs = grid_kWh * tariff_Rs_per_kWh. Solar is free.
"""
from __future__ import annotations

from typing import NamedTuple


class EnergySplit(NamedTuple):
    solar_kwh: float
    grid_kwh: float


def required_energy_kwh(current_soc: float, target_soc: float, battery_kwh: float) -> float:
    """E_required = max(0, (target - current) / 100 * capacity)."""
    return max(0.0, (target_soc - current_soc) / 100.0 * battery_kwh)


def energy_from_power(power_kw: float, dt_h: float) -> float:
    if power_kw < 0 or dt_h <= 0:
        raise ValueError("Power must be >= 0 and dt_h > 0.")
    return power_kw * dt_h


def soc_after_charging(current_soc: float, energy_kwh: float, battery_kwh: float) -> float:
    """New SOC in percent, clipped to 100."""
    if battery_kwh <= 0:
        raise ValueError("Battery capacity must be > 0.")
    return min(100.0, current_soc + energy_kwh / battery_kwh * 100.0)


def split_solar_grid(total_power_kw: float, solar_kw: float, dt_h: float) -> EnergySplit:
    """Split total charging power into solar and grid energy for one interval (solar first)."""
    if total_power_kw < 0 or solar_kw < 0:
        raise ValueError("Powers must be >= 0.")
    solar_power = min(total_power_kw, solar_kw)
    return EnergySplit(solar_power * dt_h, (total_power_kw - solar_power) * dt_h)


def interval_cost_rs(grid_kwh: float, tariff_rs_per_kwh: float) -> float:
    if grid_kwh < 0 or tariff_rs_per_kwh < 0:
        raise ValueError("Energy and tariff must be >= 0.")
    return grid_kwh * tariff_rs_per_kwh


def split_ev_energy(ev_energy_kwh: dict[str, float], solar_share: float) -> dict[str, EnergySplit]:
    """Attribute the station's solar share pro-rata to each EV (for per-EV schedule rows)."""
    solar_share = min(max(solar_share, 0.0), 1.0)
    return {k: EnergySplit(v * solar_share, v * (1.0 - solar_share)) for k, v in ev_energy_kwh.items()}
