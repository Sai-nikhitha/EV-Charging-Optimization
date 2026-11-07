"""Stage 7 - run complete simulations with the baseline or a trained PPO model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baseline import baseline_requests
from .core import ChargingCore, scenario_signature
from .env import ChargingEnv, MAX_EVS
from .models.scenario import Scenario

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODEL_DIR / "ppo_seed0.zip"


@dataclass
class SimResult:
    policy: str
    schedule: pd.DataFrame
    station: pd.DataFrame
    summary: dict
    signature: str
    scenario: Scenario


def run_baseline(scenario: Scenario) -> SimResult:
    core = ChargingCore(scenario, record=True)
    while not core.done:
        core.step(baseline_requests(core))
    return SimResult("Baseline", core.schedule_df(), core.station_df(), core.summary(),
                     scenario_signature(scenario), scenario)


def load_model(path: str | Path = DEFAULT_MODEL):
    from stable_baselines3 import PPO
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No trained PPO model at {p}. Train one first: python -m evcharge.train")
    return PPO.load(str(p), device="cpu")


def run_drl(scenario: Scenario, model) -> SimResult:
    if len(scenario.evs) > MAX_EVS:
        raise ValueError(f"The trained policy supports at most {MAX_EVS} EVs.")
    env = ChargingEnv(scenario=scenario, record=True)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    core = env.core
    return SimResult("DRL (PPO)", core.schedule_df(), core.station_df(), core.summary(),
                     scenario_signature(scenario), scenario)
