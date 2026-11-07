"""EV Charge Optimizer - Streamlit UI (Stage 1: configuration + validation; no simulation yet)."""
from __future__ import annotations

import html
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from evcharge.models import EVConfig, Scenario, StationConfig, VehicleType
from evcharge import plots
from evcharge.compare import compare_results, comparison_table
from evcharge.core import scenario_signature
from evcharge.evaluate import RESULTS_DIR
from evcharge.models import EVState
from evcharge.models.station import bell_solar_profile, constant_profile, time_of_use_tariff
from evcharge.priority import priority_table
from evcharge.scenario_gen import DURATIONS_H, INTERVALS_MIN
from evcharge.simulator import MODEL_DIR, load_model, run_baseline, run_drl

st.set_page_config(page_title="EV Charge Optimizer | Dashboard", page_icon="⚡", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
:root{--blue:#1769e0;--ink:#172b4d;--muted:#6b7c93;--line:#dce6f3}
.stApp{background:#f4f7fc;color:var(--ink)}
header[data-testid="stHeader"]{display:none}
.block-container{padding:24px 28px 30px;max-width:1500px}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] > div{padding-top:10px}
.brand{display:flex;align-items:center;gap:11px;padding:4px 6px 18px}
.brand-icon{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#1872ee,#51a2ff);color:#fff;display:grid;place-items:center;font-size:23px}
.brand h2{font-size:15px;line-height:1.25;margin:0;color:#123c7d}.brand small{color:var(--muted);font-size:11px}
.nav-label{font-size:10px;color:#91a0b6;letter-spacing:1.2px;font-weight:800;padding:4px 6px 6px}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:2px}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:10px 12px;border-radius:10px;width:100%;cursor:pointer}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#f4f7fc}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{display:none}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,#e5efff,#f1f6ff)}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{color:#145bd0;font-weight:750}
section[data-testid="stSidebar"] div[role="radiogroup"] p{font-size:13px;color:#536783}
.side-note{background:#f5f8ff;border:1px solid var(--line);padding:13px;border-radius:12px;font-size:12px;color:#657792;line-height:1.5;margin-top:18px}
.side-note b{color:#244a83;display:block;margin-bottom:3px}
.eyebrow{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase}
.page-title{margin:5px 0 5px;font-size:25px;font-weight:800;letter-spacing:-.5px;color:#142f59}
.subtitle{font-size:13px;color:var(--muted);margin-bottom:18px}
.pill{border:1px solid #d5e5fa;background:#fff;color:#57708f;padding:9px 12px;border-radius:10px;font-size:12px;display:inline-block}
.avatar{display:inline-grid;height:38px;width:38px;border-radius:50%;background:#dceaff;color:#145bd0;place-items:center;font-weight:800;margin-left:8px}
.metric{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 8px 24px rgba(26,68,128,.06)}
.metric-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:11px;font-weight:650}
.metric-icon{width:31px;height:31px;display:grid;place-items:center;border-radius:9px;background:#eaf2ff;font-size:16px}
.metric-value{font-size:25px;font-weight:800;letter-spacing:-.6px;margin:12px 0 4px;color:#143768}
.metric-value small{font-size:13px}.metric-foot{font-size:11px;color:#8a9ab0}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:0 8px 24px rgba(26,68,128,.06)}
.ph h3{margin:0;font-size:14px;color:#183c72}.ph .ps{font-size:11px;color:#8292a8;margin-top:5px}
.ph{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:12px}
.tag{font-size:10px;font-weight:750;border-radius:20px;padding:6px 9px;background:#eff5ff;color:#2865bb;white-space:nowrap}
.tag.g{background:#e6f8f0;color:#13865d}.tag.o{background:#fff4e4;color:#b77812}
.notice{padding:12px 14px;border:1px solid #d7e7ff;background:#f0f6ff;color:#41638f;border-radius:10px;font-size:12px;line-height:1.5;margin:10px 0}
.notice.warn{background:#fff8eb;border-color:#f6dfb3;color:#8c641e}
.notice.err{background:#fff0f0;border-color:#ffd9d9;color:#b83a3a}
.empty{padding:27px 14px;text-align:center;color:#8090a7;font-size:12px;background:#fbfdff;border:1px dashed #d9e4f2;border-radius:10px}
.empty .big{font-size:26px;margin-bottom:8px}
.tw{overflow:auto}table.t{width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap}
table.t th{text-align:left;background:#f5f8fd;color:#647792;font-weight:750;padding:11px 10px;border-bottom:1px solid var(--line)}
table.t td{padding:11px 10px;border-bottom:1px solid #edf1f7;color:#344b6b}
.idc{font-weight:750;color:#183b70}.vt{background:#eef4ff;color:#3269b7;padding:4px 7px;border-radius:6px;font-size:10px}
.tile{padding:15px;background:#f8fbff;border:1px solid var(--line);border-radius:11px}
.tile small{display:block;color:#71829a;font-size:11px}.tile strong{display:block;font-size:20px;color:#173c74;margin-top:8px}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{border-radius:9px;font-weight:650;white-space:nowrap;background:#edf3fc;color:#214c89;border:1px solid #dce7f7}
.stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover{background:#e2ecfa;color:#214c89;border-color:#c5d8f3}
.stButton>button p,.stFormSubmitButton>button p,.stDownloadButton>button p{color:inherit!important;font-size:14px!important}
.stButton>button[kind="primary"] p,.stFormSubmitButton>button[kind="primary"] p{color:#fff!important}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#1769e0;border:0;color:#fff}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover{background:#125bc6;color:#fff}
.footer{text-align:center;color:#95a2b5;font-size:10px;padding:24px 0 4px}
label p{font-size:11px!important;color:#61738c!important;font-weight:650!important}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- state
DEFAULT_STATION = dict(capacity=15.0, grid=12.0, solar_mode="Manual input", solar_kw=3.0,
                       tariff_mode="Fixed tariff", tariff=8.0, interval=5, duration=240, clock0=8)
DEFAULT_EVS = [
    dict(id="SCOOTER-01", type="Scooter", capacity=3.5, initial=30.0, target=100.0, power=1.5, arrival=0.0, deadline=180.0),
    dict(id="SCOOTER-02", type="Scooter", capacity=3.5, initial=40.0, target=100.0, power=1.5, arrival=30.0, deadline=200.0),
    dict(id="CAR-01", type="Car", capacity=40.0, initial=30.0, target=100.0, power=7.0, arrival=0.0, deadline=240.0),
]
ss = st.session_state
ss.setdefault("station", dict(DEFAULT_STATION))
ss.setdefault("evs", [dict(e) for e in DEFAULT_EVS])
ss.setdefault("ver", 0)          # bumped on reset so form widgets reload defaults
ss.setdefault("sel_ev", DEFAULT_EVS[0]["id"])   # ID of the EV being edited
if "pending_sel" in ss:                          # applied before the radio is created
    ss["sel_ev"] = ss.pop("pending_sel")
ss.setdefault("flash", {})       # page -> (kind, message)

PAGES = {
    "▦  Dashboard": ("Charging Overview", "Monitor your station setup and configure the initial EV fleet."),
    "⚡  Station Configuration": ("Station Configuration", "Configure station capacity, grid limits, solar and tariffs."),
    "🚘  EV Configuration": ("EV Configuration", "Set individual battery, SOC, charging power and departure parameters."),
    "◉  Dynamic Priority": ("Dynamic Priority", "Preview energy demand and feasibility from the configured inputs."),
    "▷  Run Simulation": ("Run Simulation", "Launch the future baseline and PPO simulation workflow."),
    "▥  Results & Comparison": ("Results & Comparison", "Compare measured performance from identical scenario inputs."),
    "▤  Charging Schedule": ("Charging Schedule", "Inspect interval-level charging allocations and export data."),
    "⌁  Visualizations": ("Visualizations", "Explore power, SOC, energy and utilization charts."),
    "ⓘ  About Project": ("About Project", "Project scope, assumptions and implementation status."),
}


def go(page_label: str) -> None:
    ss["nav"] = page_label


# ----------------------------------------------------------------------------- domain helpers
def build_scenario() -> Scenario:
    """Convert UI state (minutes) into validated domain objects (hours)."""
    s = ss.station
    dt_h = s["interval"] / 60.0
    n = int(round(s["duration"] / s["interval"]))
    c0 = float(s["clock0"])
    if s["solar_mode"] == "No solar":
        solar: list[float] = []
    elif s["solar_mode"] == "Manual input":
        solar = constant_profile(s["solar_kw"], n)
    else:
        solar = bell_solar_profile(s["solar_kw"], n, dt_h, 6.0 - c0, 18.0 - c0)
    tariff = (constant_profile(s["tariff"], n) if s["tariff_mode"] == "Fixed tariff"
              else time_of_use_tariff(n, dt_h, c0))
    station = StationConfig(s["capacity"], s["grid"], int(s["interval"]), s["duration"] / 60.0, solar, tariff)
    evs = [EVConfig(e["id"], VehicleType(e["type"]), e["capacity"], e["initial"], e["target"],
                    e["power"], e["arrival"] / 60.0, e["deadline"] / 60.0) for e in ss.evs]
    return Scenario(evs, station)


def required_kwh(e: dict) -> float:
    return max(0.0, (e["target"] - e["initial"]) / 100 * e["capacity"])


def avg_power(e: dict) -> float:
    mins = e["deadline"] - e["arrival"]
    return required_kwh(e) / (mins / 60) if mins > 0 else float("inf")


def feasible_alone(e: dict) -> bool:
    return avg_power(e) <= e["power"] + 1e-9


def esc(x) -> str:
    return html.escape(str(x))


def tag(text: str, kind: str = "") -> str:
    return f'<span class="tag {kind}">{esc(text)}</span>'


def table(headers: list[str], rows: list[list[str]]) -> None:
    """rows contain pre-escaped HTML."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<div class="tw"><table class="t"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>',
                unsafe_allow_html=True)


def head(title: str, sub: str, tag_html: str = "") -> None:
    st.markdown(f'<div class="ph"><div><h3>{esc(title)}</h3><div class="ps">{esc(sub)}</div></div>{tag_html}</div>',
                unsafe_allow_html=True)


def notice(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="notice {kind}">{text}</div>', unsafe_allow_html=True)


def flash(page: str) -> None:
    if page in ss.flash:
        kind, msg = ss.flash.pop(page)
        notice(msg, kind)


def metric(label: str, icon: str, value: str, unit: str, foot: str) -> str:
    u = f" <small>{unit}</small>" if unit else ""
    return (f'<div class="metric"><div class="metric-head"><span>{label}</span><span class="metric-icon">{icon}</span></div>'
            f'<div class="metric-value">{value}{u}</div><div class="metric-foot">{foot}</div></div>')


def fmt(x: float) -> str:
    return f"{x:g}"


def fleet_rows() -> list[list[str]]:
    return [[f'<span class="idc">{esc(e["id"])}</span>', f'<span class="vt">{e["type"]}</span>', f'{fmt(e["capacity"])} kWh',
             f'{fmt(e["initial"])}%', f'{fmt(e["target"])}%', f'{fmt(e["power"])} kW'] for e in ss.evs]


def feas_tag(e: dict, long: bool = False) -> str:
    ok = feasible_alone(e)
    if long:
        return tag("Feasible alone" if ok else "Infeasible alone", "g" if ok else "o")
    return tag("Vehicle-only feasible" if ok else "Check feasibility", "g" if ok else "o")


def ap(e: dict) -> str:
    p = avg_power(e)
    return f"{p:.2f} kW" if p != float("inf") else "—"


# ----------------------------------------------------------------------------- forms
def station_form(prefix: str) -> None:
    s = ss.station
    v = ss.ver
    with st.form(f"station_{prefix}_{v}"):
        a, b = st.columns(2)
        cap = a.number_input("Total station capacity (kW)", 0.0, 1000.0, float(s["capacity"]), 0.5, key=f"{prefix}cap{v}")
        grid = b.number_input("Grid power limit (kW)", 0.0, 1000.0, float(s["grid"]), 0.5, key=f"{prefix}grid{v}")
        modes = ["Manual input", "Predefined profile", "No solar"]
        sm = a.selectbox("Solar energy mode", modes, index=modes.index(s["solar_mode"]), key=f"{prefix}sm{v}")
        sk = b.number_input("Solar power / peak (kW)", 0.0, 1000.0, float(s["solar_kw"]), 0.5, key=f"{prefix}sk{v}")
        tmodes = ["Fixed tariff", "Time-dependent tariff"]
        tm = a.selectbox("Electricity tariff mode", tmodes, index=tmodes.index(s["tariff_mode"]), key=f"{prefix}tm{v}")
        tr = b.number_input("Tariff (₹/kWh, fixed mode)", 0.0, 1000.0, float(s["tariff"]), 0.5, key=f"{prefix}tr{v}")
        ints = [1, 5, 10, 15, 30]
        iv = a.selectbox("Time interval (minutes)", ints, index=ints.index(s["interval"]), key=f"{prefix}iv{v}")
        du = b.number_input("Simulation duration (minutes)", 5.0, 4320.0, float(s["duration"]), 5.0, key=f"{prefix}du{v}")
        c0 = a.number_input("Clock hour at t = 0 (for solar/ToU profiles)", 0, 23, int(s["clock0"]), 1, key=f"{prefix}c0{v}")
        c1, c2, _ = st.columns([1, 1, 3])
        save = c1.form_submit_button("Save station settings", type="primary")
        reset = c2.form_submit_button("Reset defaults")
    if reset:
        ss.station = dict(DEFAULT_STATION)
        ss.ver += 1
        ss.flash[ss.nav] = ("", "Default station settings restored.")
        st.rerun()
    if save:
        old = dict(ss.station)
        ss.station = dict(capacity=cap, grid=grid, solar_mode=sm, solar_kw=sk, tariff_mode=tm,
                          tariff=tr, interval=int(iv), duration=du, clock0=int(c0))
        errs = build_scenario().station.validate()
        if errs:
            ss.station = old
            ss.flash[ss.nav] = ("err", "Not saved:<br>" + "<br>".join(esc(x) for x in errs))
        else:
            ss.flash[ss.nav] = ("", "Station settings saved in this browser session. No simulation was run.")
        st.rerun()


def vehicle_form() -> None:
    v = ss.ver
    ids = [x["id"] for x in ss.evs]
    idx = ids.index(ss.sel_ev) if ss.sel_ev in ids else 0
    e = ss.evs[idx]
    k = f"ev{idx}_{v}_"
    with st.form(f"vehicle_{idx}_{v}"):
        a, b = st.columns(2)
        vid = a.text_input("Vehicle ID", e["id"], key=k + "id")
        vt = b.selectbox("Vehicle type", ["Scooter", "Car"], index=0 if e["type"] == "Scooter" else 1, key=k + "t")
        cap = a.number_input("Battery capacity (kWh)", 0.0, 500.0, float(e["capacity"]), 0.5, key=k + "c")
        pw = b.number_input("Maximum charging power (kW)", 0.0, 500.0, float(e["power"]), 0.5, key=k + "p")
        s0 = a.number_input("Initial SOC (%)", -10.0, 110.0, float(e["initial"]), 1.0, key=k + "s0")
        s1 = b.number_input("Target SOC (%)", -10.0, 110.0, float(e["target"]), 1.0, key=k + "s1")
        ar = a.number_input("Arrival time (minutes)", -60.0, 4320.0, float(e["arrival"]), 5.0, key=k + "ar")
        dp = b.number_input("Departure deadline (minutes)", -60.0, 4320.0, float(e["deadline"]), 5.0, key=k + "dp")
        c1, c2, _ = st.columns([1, 1, 3])
        save = c1.form_submit_button("Save vehicle settings", type="primary")
        reset = c2.form_submit_button("Reset scenario")
    if reset:
        ss.evs = [dict(x) for x in DEFAULT_EVS]
        ss["pending_sel"] = DEFAULT_EVS[0]["id"]
        ss.ver += 1
        ss.flash[ss.nav] = ("", "Default three-EV scenario restored.")
        st.rerun()
    if save:
        new = dict(id=vid.strip(), type=vt, capacity=cap, initial=s0, target=s1, power=pw, arrival=ar, deadline=dp)
        old = ss.evs[idx]
        ss.evs[idx] = new
        errs = build_scenario().validate()
        if errs:
            ss.evs[idx] = old
            ss.flash[ss.nav] = ("err", "Not saved:<br>" + "<br>".join(esc(x) for x in errs))
        else:
            ss["pending_sel"] = new["id"]
            ss.flash[ss.nav] = ("", "Vehicle settings saved in this browser session. No simulation was run.")
        st.rerun()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">ϟ</div><div><h2>EV Charge<br>Optimizer</h2>'
                '<small>Intelligent energy control</small></div></div><div class="nav-label">WORKSPACE</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
    st.markdown('<div class="side-note"><b>● Prototype mode</b>Configuration and validation run on the real Python '
                'models. Simulation and PPO backend are added in later stages.</div>', unsafe_allow_html=True)

title, subtitle = PAGES[page]
h1, h2 = st.columns([4, 1])
h1.markdown(f'<div class="eyebrow">ENERGY MANAGEMENT PLATFORM</div><div class="page-title">{title}</div>'
            f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
h2.markdown('<div style="text-align:right;padding-top:8px"><span class="pill">◉ Demo workspace</span>'
            '<span class="avatar">EV</span></div>', unsafe_allow_html=True)

scenario = build_scenario()
errors = scenario.validate()
S = ss.station

# ----------------------------------------------------------------------------- results helpers
ss.setdefault("results", {})


def fresh(key: str):
    """Return a stored result only if it was produced from the CURRENT inputs (else None)."""
    res = ss.results.get(key)
    if res is None or errors:
        return None
    return res if res.signature == scenario_signature(scenario) else None


@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_model(path)


def empty(icon: str, text: str) -> None:
    st.markdown(f'<div class="empty"><div class="big">{icon}</div>{text}</div>', unsafe_allow_html=True)


def states_at(sc: Scenario, res, step: int):
    """EV states (and previous allocations) at the start of `step`, taken from a finished simulation."""
    states = [EVState(c) for c in sc.evs]
    prev: dict[str, float] = {}
    if res is not None:
        sched = res.schedule
        rows = sched[sched["Step"] == step]
        for s in states:
            r = rows[rows["EV ID"] == s.config.ev_id].iloc[0]
            s.energy_kwh = float(r["SOC start (%)"]) / 100.0 * s.config.battery_kwh
        if step > 0:
            pr = sched[sched["Step"] == step - 1]
            prev = dict(zip(pr["EV ID"], pr["Allocated power (kW)"]))
    return states, prev


def drl_scope_warnings(sc: Scenario) -> list[str]:
    w = []
    if sc.station.interval_min not in INTERVALS_MIN:
        w.append(f"Interval {sc.station.interval_min} min was not used in training {INTERVALS_MIN}; expect weaker DRL behaviour.")
    if sc.station.duration_h > max(DURATIONS_H) + 1e-9:
        w.append(f"Duration {sc.station.duration_h:g} h is longer than any training episode (max {max(DURATIONS_H):g} h).")
    return w


def summary_block(res) -> None:
    s = res.summary
    cols = st.columns(4)
    items = [("Grid energy", f"{s['grid_kwh']:.2f} kWh"), ("Solar energy used", f"{s['solar_kwh']:.2f} kWh"),
             ("Delivered to batteries", f"{s['delivered_kwh']:.2f} kWh"), ("Grid electricity cost", f"Rs {s['cost_rs']:.2f}")]
    for c, (l, v) in zip(cols, items):
        c.markdown(f'<div class="tile"><small>{l}</small><strong>{v}</strong></div>', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(4)
    items = [("Targets reached", f"{s['targets_reached']}/{s['n_evs']} ({s['targets_pct']:.0f}%)"),
             ("Deadlines met", f"{s['deadlines_met']}/{s['n_evs']} ({s['deadlines_pct']:.0f}%)"),
             ("Station utilization", f"{s['station_utilization_pct']:.1f} %"), ("Status", s["completion"])]
    for c, (l, v) in zip(cols, items):
        c.markdown(f'<div class="tile"><small>{l}</small><strong style="font-size:15px">{v}</strong></div>', unsafe_allow_html=True)
    st.write("")
    tg = {c.ev_id: c.target_soc for c in res.scenario.evs}
    st.dataframe(pd.DataFrame([{"EV": k, "Final SOC (%)": round(v, 2), "Target SOC (%)": tg[k],
                                "Target reached": "Yes" if s["target_reached"][k] else "No",
                                "Deadline": "Met" if s["target_reached"][k] else "Missed"} for k, v in s["final_soc"].items()]),
                 hide_index=True, width="stretch")


def eval_panel() -> None:
    f_s, f_d = RESULTS_DIR / "eval_summary.csv", RESULTS_DIR / "eval_scenarios.csv"
    if not f_s.exists():
        empty("▥", "No evaluation file yet. Run <code>python -m evcharge.evaluate --scenarios 100</code>.")
        return
    summ, det = pd.read_csv(f_s), pd.read_csv(f_d)
    st.caption(f"{int(summ['scenarios'].iloc[0])} held-out random scenarios, same scenarios for every model. "
               "Generated by scenario_gen.py - simulated, not real stations.")
    st.dataframe(summ.round(3), hide_index=True, width="stretch")
    pct = det.dropna(subset=["savings_pct"])
    fig = px.box(pct, x="model", y="savings_pct", points="all")
    fig.update_layout(height=340, yaxis_title="Cost reduction vs baseline (%); negative = DRL costs more",
                      xaxis_title="", paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=20, b=10))
    fig.add_hline(y=0, line_dash="dot", line_color="#e45757")
    st.plotly_chart(fig, width="stretch")
    st.caption("Read with the service columns (mean_delivered_diff_kwh, mean_targets_diff, mean_miss_diff): "
               "a cost cut with fewer delivered kWh or more missed deadlines is not a real saving.")


# ----------------------------------------------------------------------------- pages
if page.endswith("Dashboard"):
    cols = st.columns(5)
    solar_shown = 0 if S["solar_mode"] == "No solar" else S["solar_kw"]
    cards = [("Station capacity", "⚡", fmt(S["capacity"]), "kW", "Total station power limit"),
             ("Grid power limit", "⌁", fmt(S["grid"]), "kW", "Maximum grid supply"),
             ("Solar available", "☼", fmt(solar_shown), "kW", "Configured solar input"),
             ("Configured EVs", "🚘", f"{len(ss.evs)} / 3", "", "Initial demonstration fleet"),
             ("Simulation horizon", "◷", fmt(S["duration"]), "min", "Configured duration")]
    for c, args in zip(cols, cards):
        c.markdown(metric(*args), unsafe_allow_html=True)
    st.write("")
    flash("▦  Dashboard")
    if errors:
        notice("<b>Current scenario has problems:</b><br>" + "<br>".join(esc(x) for x in errors), "err")
    left, right = st.columns(2)
    with left, st.container(border=True):
        head("Station Configuration", "Set the operating limits and energy source", tag("Editable"))
        station_form("d")
    with right, st.container(border=True):
        head("EV Fleet Overview", "Three vehicles in the initial scenario", tag(f"{len(ss.evs)} vehicles", "g"))
        table(["Vehicle", "Type", "Battery", "Initial SOC", "Target", "Max Power"], fleet_rows())
        st.button("Edit EV configuration →", on_click=go, args=("🚘  EV Configuration",))
        notice("Editable example inputs. Results appear only after you run a simulation.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        head("Dynamic Priority Preview", "Energy demand and vehicle-only feasibility from current inputs")
        table(["Vehicle", "Required energy", "Avg. required power", "Vehicle limit check"],
              [[f'<span class="idc">{esc(e["id"])}</span>', f"{required_kwh(e):.2f} kWh", ap(e), feas_tag(e)] for e in ss.evs])
        st.markdown('<div class="ps" style="font-size:11px;color:#8292a8;margin-top:10px">Vehicle-only check at arrival. '
                    'The full time-varying priority score is on the Dynamic Priority page.</div>', unsafe_allow_html=True)
        st.button("View details", on_click=go, args=("◉  Dynamic Priority",))
    with right, st.container(border=True):
        b0, d0 = fresh("base"), fresh("drl")
        head("Simulation Status", "Baseline and PPO workflow",
             tag("Results available" if (b0 or d0) else "Not run", "g" if (b0 or d0) else "o"))
        if b0 or d0:
            lines = []
            if b0:
                lines.append(f"Baseline: Rs {b0.summary['cost_rs']:.2f}, {b0.summary['targets_reached']}/{b0.summary['n_evs']} targets reached")
            if d0:
                lines.append(f"DRL: Rs {d0.summary['cost_rs']:.2f}, {d0.summary['targets_reached']}/{d0.summary['n_evs']} targets reached")
            notice("<br>".join(lines))
        else:
            empty("⌁", "<b>No simulation results yet</b><p>Open the simulation controls and run the baseline or DRL.</p>")
        st.button("Open simulation controls", type="primary", on_click=go, args=("▷  Run Simulation",))

elif page.endswith("Station Configuration"):
    with st.container(border=True):
        head("Charging Station Setup", "Station inputs are independent of vehicle parameters.", tag("Configuration"))
        flash(page)
        station_form("s")
        notice("Important: station capacity, grid limit, solar availability and each EV's charging power are four "
               "separate constraints. The allocation engine (Stage 2) enforces all of them. "
               "Deliverable power = min(station capacity, solar + grid limit).")

elif page.endswith("EV Configuration"):
    with st.container(border=True):
        head("Vehicle Configuration", "Adjust each vehicle independently. Defaults are illustrative, not universal vehicle specifications.",
             tag("3 EV slots"))
        st.radio("Vehicle", [x["id"] for x in ss.evs],
                 horizontal=True, key="sel_ev", label_visibility="collapsed")
        flash(page)
        vehicle_form()
    st.write("")
    with st.container(border=True):
        head("Configuration Summary", "Calculated directly from the editable inputs, not a simulation.")
        table(["Vehicle ID", "Type", "Required energy (kWh)", "Available time (h)", "Required average power (kW)",
               "Vehicle-only feasibility"],
              [[f'<span class="idc">{esc(e["id"])}</span>', e["type"], f"{required_kwh(e):.2f}",
                f"{max(0, e['deadline'] - e['arrival']) / 60:.2f}",
                (f"{avg_power(e):.2f}" if avg_power(e) != float("inf") else "—"), feas_tag(e, True)] for e in ss.evs])

elif page.endswith("Dynamic Priority"):
    with st.container(border=True):
        head("Dynamic Priority Engine", "Score = 100 x (0.5 urgency + 0.2 energy need + 0.2 deadline closeness + 0.1 continuity). "
             "Urgency and feasibility are reported separately.",
             tag("Live" if fresh("base") else "Initial state", "g" if fresh("base") else "o"))
        if errors:
            notice("Fix the configuration errors first:<br>" + "<br>".join(esc(x) for x in errors), "err")
        else:
            n_steps = scenario.station.n_steps
            ivl = scenario.station.interval_min
            step = 0
            if n_steps > 1:
                tmin = st.slider("Simulation time (min)", 0, (n_steps - 1) * ivl, 0, step=ivl)
                step = tmin // ivl
            base = fresh("base")
            states, prev = states_at(scenario, base, step)
            df = priority_table(states, step * scenario.station.dt_h, scenario.station.max_deliverable_kw(step), prev)
            if base:
                notice("SOC values come from the <b>baseline simulation</b> at the selected time. Move the slider to watch the table update.")
            else:
                notice("No simulation has run, so this shows the <b>initial SOC with no charging applied</b>. "
                       "Run the baseline to see priorities evolve as vehicles charge.", "warn")
            st.dataframe(df, hide_index=True, width="stretch")
            notice("Feasibility compares required average power with min(EV max power, station power available now). "
                   "It does not include competition between EVs - the allocation step reveals that.")

elif page.endswith("Run Simulation"):
    with st.container(border=True):
        head("Simulation Controls", "Baseline and DRL run on identical scenario inputs.", tag("Ready" if not errors else "Fix inputs", "g" if not errors else "o"))
        notice("Both policies use the same EVs, station limits, solar profile, tariff profile and interval. Constraints are enforced "
               "by the allocation code for both. DRL needs a trained model in <code>models/</code> (run <code>python -m evcharge.train</code>).")
        if errors:
            notice("Fix the configuration errors first:<br>" + "<br>".join(esc(x) for x in errors), "err")
        models = sorted(MODEL_DIR.glob("ppo_seed*.zip"))
        chosen = st.selectbox("PPO model", models, index=0 if models else None, format_func=lambda p: p.stem,
                              disabled=not models, placeholder="No trained model found")
        if not models:
            notice("No trained PPO model found. Run <code>python -m evcharge.train --seed 0</code> first. "
                   "Baseline simulation still works.", "warn")
        r1, r2 = st.columns(2), st.columns(2)
        do_base = r1[0].button("▶ Run Baseline Simulation", width="stretch", disabled=bool(errors))
        do_drl = r1[1].button("⚡ Run DRL Simulation", type="primary", width="stretch", disabled=bool(errors) or not models)
        do_cmp = r2[0].button("⇄ Compare Both Methods", width="stretch", disabled=bool(errors) or not models)
        if r2[1].button("↻ Reset Scenario", width="stretch"):
            ss.station = dict(DEFAULT_STATION)
            ss.evs = [dict(x) for x in DEFAULT_EVS]
            ss["pending_sel"] = DEFAULT_EVS[0]["id"]
            ss.ver += 1
            ss.results = {}
            ss["nav"] = "▦  Dashboard"
            st.rerun()
        if (do_drl or do_cmp) and chosen:
            for w_ in drl_scope_warnings(scenario):
                notice(w_, "warn")
        try:
            if do_base or do_cmp:
                with st.spinner("Running baseline..."):
                    ss.results["base"] = run_baseline(scenario)
            if do_drl or do_cmp:
                with st.spinner("Running PPO policy..."):
                    ss.results["drl"] = run_drl(scenario, get_model(str(chosen)))
                ss.results["model"] = chosen.stem
            if do_base or do_drl or do_cmp:
                notice("Simulation finished. See <b>Results & Comparison</b>, <b>Charging Schedule</b> and <b>Visualizations</b>.")
        except Exception as exc:  # surface errors instead of crashing the page
            notice(f"Simulation failed: {esc(exc)}", "err")
        st.markdown("**Export results to CSV**")
        e1, e2, e3 = st.columns(3)
        b_, d_ = fresh("base"), fresh("drl")
        e1.download_button("Baseline schedule CSV", b_.schedule.to_csv(index=False) if b_ else "", "baseline_schedule.csv",
                           "text/csv", disabled=b_ is None, width="stretch")
        e2.download_button("DRL schedule CSV", d_.schedule.to_csv(index=False) if d_ else "", "drl_schedule.csv",
                           "text/csv", disabled=d_ is None, width="stretch")
        e3.download_button("Comparison CSV", comparison_table(b_, d_).to_csv(index=False) if (b_ and d_) else "",
                           "comparison.csv", "text/csv", disabled=not (b_ and d_), width="stretch")

elif page.endswith("Results & Comparison"):
    b_, d_ = fresh("base"), fresh("drl")
    if ss.results and not (b_ or d_):
        notice("Inputs changed since the last run. Run the simulation again to refresh results.", "warn")
    t1, t2, t3, t4 = st.tabs(["Comparison", "Baseline results", "DRL results", "Multi-scenario evaluation"])
    with t1, st.container(border=True):
        head("Baseline vs DRL (this scenario)", "All values are computed from the two simulations.",
             tag("Measured" if (b_ and d_) else "No results yet", "g" if (b_ and d_) else "o"))
        if not (b_ and d_):
            empty("⇄", "Run <b>Compare Both Methods</b> on the Run Simulation page to see results.")
        else:
            c = compare_results(b_, d_)
            pct = "n/a (baseline cost is 0)" if c["savings_pct"] is None else f"{c['savings_pct']:+.2f} %"
            cols = st.columns(4)
            for col, (lbl, val) in zip(cols, [("Baseline cost", f"Rs {c['baseline_cost']:.2f}"), ("DRL cost", f"Rs {c['drl_cost']:.2f}"),
                                              ("Savings (Baseline - DRL)", f"Rs {c['savings_rs']:+.2f}"), ("Cost reduction", pct)]):
                col.markdown(f'<div class="tile"><small>{lbl}</small><strong>{val}</strong></div>', unsafe_allow_html=True)
            notice(f"<b>{esc(c['verdict'])}</b><br>Energy delivered (DRL - Baseline): {c['delivered_diff_kwh']:+.2f} kWh · "
                   f"Targets reached: {c['targets_diff']:+d} · Deadline misses: {c['deadline_miss_diff']:+d}. "
                   "Cost alone is not a fair comparison unless service is equal.",
                   "warn" if (c["cheaper_but_worse_service"] or c["savings_rs"] < 0) else "")
            st.dataframe(comparison_table(b_, d_).round(3), hide_index=True, width="stretch")
            g1, g2 = st.columns(2)
            g1.plotly_chart(plots.cost_comparison(b_, d_), width="stretch")
            g2.plotly_chart(plots.final_soc_compare(b_, d_), width="stretch")
            st.caption(f"Model: {ss.results.get('model', '?')}. A single scenario is an example, not proof of general performance - "
                       "see the Multi-scenario evaluation tab.")
    for tab, res, nm in ((t2, b_, "Baseline"), (t3, d_, "DRL (PPO)")):
        with tab, st.container(border=True):
            head(f"{nm} results", "Cost, energy usage, final SOC and deadline performance.")
            if res is None:
                empty("▥", f"No {nm} results yet. Run it from the Run Simulation page.")
            else:
                summary_block(res)
    with t4, st.container(border=True):
        head("Multi-scenario, multi-seed evaluation", "Held-out random scenarios; every PPO seed vs the baseline.")
        eval_panel()

elif page.endswith("Charging Schedule"):
    b_, d_ = fresh("base"), fresh("drl")
    with st.container(border=True):
        head("Charging Schedule", "Interval-by-interval allocation from the actual simulation.", tag("Measured" if (b_ or d_) else "No data", "g" if (b_ or d_) else "o"))
        if not (b_ or d_):
            empty("▤", "No schedule yet. Configure the scenario, then run a simulation.")
        else:
            tabs = st.tabs(["DRL schedule", "Baseline schedule"])
            for tab, res, nm, fn in ((tabs[0], d_, "DRL", "drl_schedule.csv"), (tabs[1], b_, "Baseline", "baseline_schedule.csv")):
                with tab:
                    if res is None:
                        empty("▤", f"{nm} has not been run.")
                        continue
                    pick = st.multiselect("Filter EVs", [c.ev_id for c in res.scenario.evs], default=[c.ev_id for c in res.scenario.evs], key=f"f{nm}")
                    only = st.checkbox("Hide rows where the EV is not arrived / departed", key=f"h{nm}")
                    df = res.schedule[res.schedule["EV ID"].isin(pick)]
                    if only:
                        df = df[~df["Status"].isin(["Not arrived", "Departed"])]
                    st.dataframe(df, hide_index=True, width="stretch", height=420)
                    st.download_button(f"Download {nm} schedule (CSV, all rows)", res.schedule.to_csv(index=False), fn, "text/csv")

elif page.endswith("Visualizations"):
    b_, d_ = fresh("base"), fresh("drl")
    if not (b_ or d_):
        with st.container(border=True):
            head("Visualizations", "Charts use actual simulation records.", tag("No data"))
            empty("⌁", "Run a simulation to populate the charts.")
    else:
        opts = [n for n, r in (("DRL (PPO)", d_), ("Baseline", b_)) if r]
        which = st.radio("Show charts for", opts, horizontal=True)
        res = d_ if which.startswith("DRL") else b_
        for a, b in ((plots.power_per_ev, plots.station_power), (plots.soc_per_ev, plots.solar_vs_grid)):
            c1, c2 = st.columns(2)
            with c1, st.container(border=True):
                st.plotly_chart(a(res), width="stretch")
            with c2, st.container(border=True):
                st.plotly_chart(b(res), width="stretch")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.plotly_chart(plots.utilization(res), width="stretch")
        with c2, st.container(border=True):
            if b_ and d_:
                st.plotly_chart(plots.cost_comparison(b_, d_), width="stretch")
            else:
                st.info("Run both methods (Compare) to see baseline vs DRL cost.")

else:  # About
    with st.container(border=True):
        st.markdown("### About this project")
        st.write("A configurable, simulation-based EV charging optimization platform intended to compare a deterministic "
                 "baseline charging policy with a Proximal Policy Optimization (PPO) agent. It covers dynamic priority "
                 "management, solar energy integration, time-dependent tariffs and departure deadlines.")
        a, b = st.columns(2)
        with a, st.container(border=True):
            st.markdown("**Implemented**")
            st.caption("EV/station models, constraint allocator, priority engine, baseline policy, simulation core, Gymnasium environment, PPO training, evaluation, comparison, charts, CSV export and automated tests.")
        with b, st.container(border=True):
            st.markdown("**Limits**")
            st.caption("Simulation only. PPO supports up to 5 EVs and was trained on generated scenarios (3-6 h, 5-30 min intervals). This UI edits 3 EVs. No charging losses, no V2G, no real hardware.")
        notice("Assumption: charging losses are omitted; battery energy gain equals allocated charging power × interval duration. "
               "Simulation only — not connected to any real charging hardware.")

st.markdown('<div class="footer">EV Charge Optimizer · Simulation only · No hardware connection · No measured optimization claims</div>',
            unsafe_allow_html=True)
