"""Stage 6 - train PPO on randomly generated scenarios.

    python -m evcharge.train --seed 0 --timesteps 600000

Each episode draws a NEW random scenario (1..5 EVs, random SOCs, deadlines, station limits, solar,
tariffs, intervals), so the policy is not fitted to one fixed three-EV example.
Evaluation uses different random seeds than training (see evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .env import MAX_EVS, ChargingEnv

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

HPARAMS = dict(n_steps=1024, batch_size=256, gamma=0.995, gae_lambda=0.95, learning_rate=3e-4,
               ent_coef=0.003, n_epochs=10, clip_range=0.2, policy_kwargs=dict(net_arch=[128, 128]))


def train(seed: int = 0, timesteps: int = 600_000, n_envs: int = 4, out_dir: Path = MODEL_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: ChargingEnv(max_evs=MAX_EVS), n_envs=n_envs, seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **HPARAMS)
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    path = out_dir / f"ppo_seed{seed}.zip"
    model.save(str(path))
    meta = {"seed": seed, "timesteps": timesteps, "n_envs": n_envs, "max_evs": MAX_EVS,
            "train_seconds": round(time.time() - t0, 1),
            "hparams": {k: v for k, v in HPARAMS.items() if k != "policy_kwargs"}, "net_arch": [128, 128]}
    (out_dir / f"ppo_seed{seed}.json").write_text(json.dumps(meta, indent=2))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--envs", type=int, default=4)
    a = ap.parse_args()
    print("saved", train(a.seed, a.timesteps, a.envs))
