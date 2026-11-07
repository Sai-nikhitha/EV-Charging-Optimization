# DRL-Based Intelligent EV Charging Optimization (simulation)

A configurable, **simulation-only** platform that distributes charging power among several EVs and compares a
deterministic **baseline** policy with a **PPO** (Deep Reinforcement Learning) policy under identical conditions.
It is not connected to any real charger, and it does not claim to model every real station.

## 1. Install and run

```powershell
cd ev_drl_charging
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m pytest -q                                   # automated tests
python -m evcharge.train --seed 0 --timesteps 2000000 # train a PPO model (repeat for seeds 1, 2)
python -m evcharge.evaluate --scenarios 100           # held-out multi-scenario evaluation -> results/
python -m streamlit run app.py                        # web app at http://localhost:8501
```

The app works without a trained model (baseline only). `models/ppo_seed*.zip` enables the DRL buttons.

## 2. Project layout

| Path | Purpose |
|---|---|
| `evcharge/models/` | `EVConfig`/`EVState`, `StationConfig`, `Scenario` (validation, default 2 scooters + 1 car) |
| `evcharge/energy.py` | Energy accounting primitives (required energy, SOC, solar/grid split, cost) |
| `evcharge/allocation.py` | Constraint enforcement shared by all policies; order-independent water-filling |
| `evcharge/priority.py` | Dynamic priority engine (score, class, feasibility, explanation) |
| `evcharge/baseline.py` | Baseline policy |
| `evcharge/core.py` | Interval-based simulation core (one code path for baseline, Gym env and UI) |
| `evcharge/scenario_gen.py` | Random scenario generator (training + held-out evaluation) |
| `evcharge/env.py` | Gymnasium environment (observation, action, reward) |
| `evcharge/train.py`, `evaluate.py`, `compare.py`, `simulator.py`, `plots.py` | Training, evaluation, comparison, runners, Plotly charts |
| `app.py` | Streamlit dashboard |
| `tests/` | pytest suite |

## 3. Assumptions (documented, applied identically to both policies)

* Times are minutes in the UI, hours internally; power in kW, energy in kWh, money in Rs.
* **Charging losses are not modeled**: energy stored in the battery = allocated power x time.
* Power is constant within an interval; allocation is recomputed at every interval boundary.
  Arrivals/departures take effect at the next boundary; an EV arriving/leaving mid-interval is charged only for the overlapping time.
* **Solar-first**: solar covers charging first, the grid supplies the rest. Unused solar is wasted (no battery, no export). Solar is free; cost = grid kWh x tariff.
* Deliverable power in an interval = `min(station capacity, solar + grid limit)`; each EV is also capped by its own max power and by the energy still needed to reach its target SOC (no charging beyond target, before arrival or after departure).
* Four separate constraints: station power capacity, grid limit, solar availability, EV max power.
* A deadline is *met* iff the target SOC is reached before departure.

## 4. Algorithms

**Constraint enforcement (`allocation.py`)** - every policy emits *requests*; `allocate_power` clips each to the EV cap
and, if the sum exceeds the deliverable limit, scales all allocations by the same factor. Constraints are enforced in code, not by reward penalties.

**Dynamic priority (`priority.py`)** - `E_req = max(0,(target-SOC)/100*capacity)`, `T_rem = max(0, departure-max(t, arrival))`,
`P_req = E_req/T_rem` (inf if no time is left). Feasibility is reported separately from priority:
*Target reached / Missed deadline / Infeasible (EV power limit) / At risk (station limit) / Tight / Feasible*.
Score = 100 x (0.5 urgency + 0.2 energy need + 0.2 deadline closeness + 0.1 continuity), classes Critical>=70, High>=50, Medium>=30, Low.
The weights are documented defaults, not fitted to data; change them via `PriorityWeights`.

**Baseline (`baseline.py`)** - urgency-weighted proportional allocation: weights = priority score + 1, power shared proportionally with
per-EV caps and redistribution of leftovers. Deterministic, independent of EV list order, price-blind (charges as soon as possible).

**PPO environment (`env.py`)** - `MAX_EVS = 5` slots; inactive slots have all-zero observations and their actions are ignored.
Observation (69 values): per slot 12 features (chargeable, arrived, completed, SOC, target, required-energy fraction, remaining time,
max power, required/available power ratio, priority score, previous allocation, cap fraction) plus 9 global (station capacity, solar, grid limit,
tariff now/mean/min remaining, solar next hour, elapsed fraction, active-EV fraction). Action: `Box(0,1,(5,))` = fraction of each EV's max power;
converted to physically valid power by `allocate_power`.

**Reward (per step; `R` = total energy required at start)**: `+1.0*delivered/R` (so not charging never pays) `- 0.5*cost/(R*mean tariff)`
`+ 0.1*solar_used/R + 0.1` per EV reaching target `- (1.0 + 3.0*unmet_i/R)` per EV that leaves short `- 0.05*limit_excess/station`.

**Training scenarios** - each episode draws a new random scenario: 1-5 EVs, scooter/car parameters, random initial/target SOC, arrivals,
deadlines (about 12% physically infeasible), station capacity 0.35-1.1 x total EV power, grid limit, solar (none/constant/bell), fixed or time-of-use tariff,
interval in {5, 10, 15, 30} min, duration 3-6 h. Evaluation uses a different random seed than training.

## 5. Supported configurations (what is and is not claimed)

* The UI edits **3 EVs** (2 scooters + 1 car). The environment/policy support **1-5 EVs**; scenarios outside the training ranges
  (interval not in {5,10,15,30}, duration > 6 h) trigger a warning because the policy has not seen them.
* Not modeled: charging losses, battery degradation, V2G, solar storage/export, real charger dynamics or communication.

## 6. Measured results (reproducible; nothing here is hand-entered)

Training: PPO (Stable-Baselines3), 3 seeds (0, 1, 2), 2,000,000 steps each, `net_arch=[128,128]`, ~12 min/seed on CPU
(`models/ppo_seed*.json` holds the exact settings). Evaluation: `python -m evcharge.evaluate --scenarios 100`
on 100 held-out random scenarios (test seed 424242, never used in training); baseline and PPO run on the same scenario objects.
Raw rows: `results/eval_scenarios.csv`; summary: `results/eval_summary.csv`.

| PPO seed | mean cost reduction vs baseline | DRL cheaper in | DRL costlier in | mean targets reached (DRL - baseline) | cheaper *but worse service* | savings when service is matched |
|---|---|---|---|---|---|---|
| 0 | +3.8 % | 41 % of scenarios | 8 % | -0.12 EVs | 34 % | +0.98 % (60 scenarios) |
| 1 | +3.6 % | 37 % | 8 % | -0.18 EVs | 30 % | +1.05 % (61 scenarios) |
| 2 | +1.8 % | 37 % | 7 % | -0.22 EVs | 30 % | +0.84 % (62 scenarios) |

**How to read this honestly**

* PPO lowers *average* cost by roughly 2-4 %, but a large part of that comes from delivering less energy and missing more targets/deadlines
  than the baseline (it is cheaper-but-worse in about 30 % of scenarios). Where PPO serves at least as well as the baseline,
  the saving shrinks to about **1 %** (a few paise to a fraction of a rupee per scenario).
* In 7-8 % of scenarios PPO costs *more* than the baseline. The median saving is 0 %: in most scenarios both policies pay the same.
* The baseline itself misses at least one deadline in 51 % of the generated scenarios because ~12 % of EVs are physically infeasible and
  stations are often oversubscribed, so some misses are unavoidable for any policy.
* Results vary between seeds (cost reduction 1.8-3.8 %), so a single trained model should not be quoted as "the" result.

**The UI demo scenario (3 vehicles, 15 kW station, 4 h):** with a fixed tariff and constant solar, baseline and PPO cost exactly the same
(Rs 164.40): both must deliver the same energy at the same price, so there is nothing to optimise. With a time-of-use tariff the scenario is
so time-constrained that gains are small: starting at 16:00 both cost Rs 188.40; starting at 17:00 baseline Rs 207.30 vs PPO Rs 207.43 /
207.30 / 204.09 (seed 2 is 1.5 % cheaper but leaves one EV short of target, i.e. it is not a like-for-like saving).
PPO is most likely to differ when there is slack between charging need and departure time and the tariff or solar changes during the stay.

**Reward iteration (disclosed):** version 1 used a miss penalty of `0.5 + 2.0*unmet/R` and 800k steps; on a first test seed (987654) it showed
+1.4-1.8 % mean savings with 0.22-0.28 fewer targets reached per scenario. Because service was worse, the penalty was raised to
`1.0 + 3.0*unmet/R` and training lengthened to 2M steps, and the final numbers above use a *fresh* test seed. The archived v1 models and results are in
`models/archive_v1/` and `results/archive_v1/`. No other tuning was done.

## 7. Limitations

* Simulation only; scenario generator ranges are illustrative, not measured station data. Results do not transfer automatically to real stations.
* PPO does not reliably match the baseline's service level; it is a learned approximation and can miss deadlines the baseline meets (and vice-versa).
* Fixed 5-slot architecture; trained on 3-6 h episodes with 5-30 min intervals. Outside that range quality is unknown (the UI warns).
* Reward weights and priority weights are design choices, not optimised or fitted.
* No charging losses, degradation, V2G, solar storage/export or feeder-level constraints.
* 3 seeds x 100 scenarios gives indicative, not statistically conclusive, evidence.

## 8. Demonstration script

1. `python -m streamlit run app.py`; open **Dashboard** (default 2 scooters + 1 car) and edit a value; show the validation error when target SOC < initial SOC.
2. **Station Configuration**: set *Time-dependent tariff* and *Predefined profile* solar, save.
3. **Run Simulation** -> *Compare Both Methods*; open **Dynamic Priority** and drag the time slider; set an impossible deadline (e.g. CAR-01 departing after 30 min) to see "Infeasible (EV power limit)".
4. **Results & Comparison**: savings/percentage (may be negative or zero), trade-off verdict, final SOC chart; then the *Multi-scenario evaluation* tab.
5. **Charging Schedule** -> download the CSV; **Visualizations** -> the six charts.
6. Show `python -m pytest -q` (73 tests).