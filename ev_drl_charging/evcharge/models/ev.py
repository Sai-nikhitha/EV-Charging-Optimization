"""EV model: immutable configuration + mutable charging state.

Time convention: all times are hours since simulation start (float).
An EV is *present* in the half-open window [arrival_h, departure_h).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

EPS = 1e-9


class VehicleType(str, Enum):
    SCOOTER = "Scooter"
    CAR = "Car"


@dataclass(frozen=True)
class EVConfig:
    """User-editable parameters of one vehicle (independent per EV)."""
    ev_id: str
    vehicle_type: VehicleType
    battery_kwh: float
    initial_soc: float      # percent, 0..100
    target_soc: float       # percent, 0..100
    max_power_kw: float
    arrival_h: float
    departure_h: float

    def validate(self) -> list[str]:
        """Return a list of human-readable errors (empty list = valid)."""
        e: list[str] = []
        if not self.ev_id or not self.ev_id.strip():
            e.append("Vehicle ID must not be empty.")
        if not isinstance(self.vehicle_type, VehicleType):
            e.append(f"{self.ev_id}: vehicle type must be Scooter or Car.")
        if self.battery_kwh <= 0:
            e.append(f"{self.ev_id}: battery capacity must be > 0 kWh.")
        for name, v in (("initial SOC", self.initial_soc), ("target SOC", self.target_soc)):
            if not 0 <= v <= 100:
                e.append(f"{self.ev_id}: {name} must be between 0 and 100 %.")
        if self.target_soc < self.initial_soc:
            e.append(f"{self.ev_id}: target SOC must be >= initial SOC.")
        if self.max_power_kw <= 0:
            e.append(f"{self.ev_id}: max charging power must be > 0 kW.")
        if self.arrival_h < 0:
            e.append(f"{self.ev_id}: arrival time must be >= 0 h.")
        if self.departure_h <= self.arrival_h:
            e.append(f"{self.ev_id}: departure must be after arrival.")
        return e


@dataclass
class EVState:
    """Mutable charging state. SOC is derived from stored energy (single source of truth)."""
    config: EVConfig
    energy_kwh: float = field(init=False)
    delivered_kwh: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.energy_kwh = self.config.initial_soc / 100.0 * self.config.battery_kwh

    # ---- derived quantities -------------------------------------------------
    @property
    def soc(self) -> float:
        """Current SOC in percent."""
        return self.energy_kwh / self.config.battery_kwh * 100.0

    @property
    def target_energy_kwh(self) -> float:
        return self.config.target_soc / 100.0 * self.config.battery_kwh

    @property
    def required_energy_kwh(self) -> float:
        """E_required = max(0, (target - current) / 100 * capacity)."""
        return max(0.0, self.target_energy_kwh - self.energy_kwh)

    @property
    def target_reached(self) -> bool:
        return self.required_energy_kwh <= 1e-6

    def is_present(self, t_h: float) -> bool:
        return self.config.arrival_h - EPS <= t_h < self.config.departure_h - EPS

    def remaining_time_h(self, t_h: float) -> float:
        """Hours until departure, never negative."""
        return max(0.0, self.config.departure_h - t_h)

    # ---- mutation -----------------------------------------------------------
    def add_energy(self, kwh: float) -> float:
        """Add energy to the battery, capped at the *target* SOC. Returns kWh actually stored."""
        if kwh < 0:
            raise ValueError("Energy to add must be >= 0.")
        stored = min(kwh, self.required_energy_kwh)
        self.energy_kwh += stored
        self.delivered_kwh += stored
        return stored
