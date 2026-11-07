"""Stage 10 - Plotly figures. Every figure is built from actual SimResult data."""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .simulator import SimResult

COLORS = ["#1769e0", "#16a673", "#e89a22", "#8154df", "#e45757", "#0ea5b7"]
BASE_C, DRL_C = "#6b7c93", "#1769e0"


def _layout(fig: go.Figure, title: str, ytitle: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#183c72")), height=height,
                      margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font=dict(color="#344b6b"), legend=dict(orientation="h", y=-0.2),
                      xaxis=dict(title="Time (min)", gridcolor="#edf1f7"), yaxis=dict(title=ytitle, gridcolor="#edf1f7"))
    return fig


def power_per_ev(res: SimResult) -> go.Figure:
    fig = go.Figure()
    for i, (ev, g) in enumerate(res.schedule.groupby("EV ID", sort=False)):
        fig.add_trace(go.Scatter(x=g["Time (min)"], y=g["Allocated power (kW)"], name=ev, mode="lines",
                                 line=dict(shape="hv", color=COLORS[i % len(COLORS)])))
    return _layout(fig, f"Charging power vs time per EV - {res.policy}", "Power (kW)")


def station_power(res: SimResult) -> go.Figure:
    s = res.station
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["Time (min)"], y=s["Total power (kW)"], name="Total station power",
                             line=dict(shape="hv", color=DRL_C), fill="tozeroy", fillcolor="rgba(23,105,224,.12)"))
    fig.add_trace(go.Scatter(x=s["Time (min)"], y=s["Deliverable limit (kW)"], name="Deliverable limit (min(station, solar+grid))",
                             line=dict(shape="hv", color="#e89a22", dash="dash")))
    fig.add_trace(go.Scatter(x=s["Time (min)"], y=s["Station capacity (kW)"], name="Station capacity",
                             line=dict(color="#e45757", dash="dot")))
    return _layout(fig, f"Total station power vs time - {res.policy}", "Power (kW)")


def soc_per_ev(res: SimResult) -> go.Figure:
    fig = go.Figure()
    targets = {c.ev_id: c.target_soc for c in res.scenario.evs}
    for i, (ev, g) in enumerate(res.schedule.groupby("EV ID", sort=False)):
        col = COLORS[i % len(COLORS)]
        t_end = g["Time (min)"].iloc[-1] + res.scenario.station.interval_min
        x = [g["Time (min)"].iloc[0]] + (g["Time (min)"] + res.scenario.station.interval_min).tolist()
        y = [g["SOC start (%)"].iloc[0]] + g["SOC updated (%)"].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, name=ev, line=dict(color=col)))
        fig.add_trace(go.Scatter(x=[0, t_end], y=[targets[ev]] * 2, name=f"{ev} target", showlegend=False,
                                 line=dict(color=col, dash="dot", width=1)))
    fig = _layout(fig, f"EV SOC vs time (dotted = target SOC) - {res.policy}", "SOC (%)")
    fig.update_yaxes(range=[0, 105])
    return fig


def solar_vs_grid(res: SimResult) -> go.Figure:
    s = res.station
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["Time (min)"], y=s["Solar energy (kWh)"], name="Solar energy", marker_color="#e89a22"))
    fig.add_trace(go.Bar(x=s["Time (min)"], y=s["Grid energy (kWh)"], name="Grid energy", marker_color=DRL_C))
    fig.update_layout(barmode="stack")
    return _layout(fig, f"Solar vs grid energy per interval - {res.policy}", "Energy (kWh)")


def cost_comparison(base: SimResult, drl: SimResult) -> go.Figure:
    fig = go.Figure()
    for r, c in ((base, BASE_C), (drl, DRL_C)):
        s = r.station
        fig.add_trace(go.Scatter(x=s["Time (min)"], y=s["Interval cost (Rs)"].cumsum(), name=r.policy,
                                 line=dict(shape="hv", color=c)))
    return _layout(fig, "Cumulative grid electricity cost: baseline vs DRL", "Cost (Rs)")


def utilization(res: SimResult) -> go.Figure:
    s = res.station
    util = s["Total power (kW)"] / s["Station capacity (kW)"] * 100
    fig = go.Figure(go.Scatter(x=s["Time (min)"], y=util, line=dict(shape="hv", color="#8154df"), name="Utilization",
                               fill="tozeroy", fillcolor="rgba(129,84,223,.12)"))
    fig = _layout(fig, f"Station power utilization - {res.policy}", "Utilization (% of station capacity)")
    fig.update_yaxes(range=[0, 105])
    return fig


def utilization_compare(base: SimResult, drl: SimResult) -> go.Figure:
    fig = go.Figure()
    for r, c in ((base, BASE_C), (drl, DRL_C)):
        s = r.station
        fig.add_trace(go.Scatter(x=s["Time (min)"], y=s["Total power (kW)"] / s["Station capacity (kW)"] * 100,
                                 name=r.policy, line=dict(shape="hv", color=c)))
    fig = _layout(fig, "Station power utilization: baseline vs DRL", "Utilization (%)")
    fig.update_yaxes(range=[0, 105])
    return fig


def final_soc_compare(base: SimResult, drl: SimResult) -> go.Figure:
    ids = list(base.summary["final_soc"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ids, y=[base.summary["final_soc"][i] for i in ids], name="Baseline", marker_color=BASE_C))
    fig.add_trace(go.Bar(x=ids, y=[drl.summary["final_soc"][i] for i in ids], name="DRL (PPO)", marker_color=DRL_C))
    tg = {c.ev_id: c.target_soc for c in base.scenario.evs}
    fig.add_trace(go.Scatter(x=ids, y=[tg[i] for i in ids], mode="markers", name="Target SOC",
                             marker=dict(symbol="line-ew", size=28, color="#e45757", line=dict(width=3, color="#e45757"))))
    fig = _layout(fig, "Final SOC per EV vs target", "SOC (%)")
    fig.update_xaxes(title="")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 105])
    return fig
