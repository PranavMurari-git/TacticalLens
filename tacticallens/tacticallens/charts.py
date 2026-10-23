"""Matplotlib / mplsoccer charts. They read from MatchAnalytics so numbers match the tools."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

from .analytics import CHANNELS, THIRDS, MatchAnalytics

CHARTS = ("shot_map", "pass_network", "zone_heatmap", "xg_timeline")
COLORS = ["#1f6feb", "#e5534b"]
_PITCH = dict(pitch_type="statsbomb", pitch_color="#fbfbf8", line_color="#8a8a80", linewidth=1.2)


def _surname(name: str) -> str:
    return name.split()[-1] if name else ""


def shot_map(a: MatchAnalytics, team: str):
    team = a.resolve_team(team)
    s = a._of(team, "Shot").dropna(subset=["x", "y"])
    pitch = Pitch(half=True, **_PITCH)
    fig, ax = pitch.draw(figsize=(7, 5.2))
    goals = s[s["shot_outcome"] == "Goal"]
    other = s[s["shot_outcome"] != "Goal"]
    pitch.scatter(other.x, other.y, s=60 + other["shot_xg"].fillna(0) * 900, ax=ax,
                  c="#9aa5b1", edgecolors="#33414d", alpha=0.8, label="Shot")
    pitch.scatter(goals.x, goals.y, s=60 + goals["shot_xg"].fillna(0) * 900, ax=ax, marker="*",
                  c="#f2b01e", edgecolors="#33414d", label="Goal", zorder=3)
    for _, r in s[(s["shot_xg"] >= 0.3) | (s["shot_outcome"] == "Goal")].iterrows():
        ax.annotate(f"{_surname(r['player'])} {r['shot_xg']:.2f}", (r["x"], r["y"]),
                    xytext=(6, -10), textcoords="offset points", fontsize=7.5)
    ax.set_title(f"{team}: shots (marker size = xG) | total xG {s['shot_xg'].sum():.2f}", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    return fig


def pass_network(a: MatchAnalytics, team: str):
    net = a.passing_network(team, min_passes=2)
    pitch = Pitch(**_PITCH)
    fig, ax = pitch.draw(figsize=(8, 5.4))
    pos = {n["player"]: (n["avg_x"], n["avg_y"]) for n in net["nodes"]}
    peak = max([l["passes"] for l in net["top_links"]] or [1])
    for l in net["top_links"]:
        if l["from"] in pos and l["to"] in pos:
            (x1, y1), (x2, y2) = pos[l["from"]], pos[l["to"]]
            pitch.lines(x1, y1, x2, y2, lw=1 + 5 * l["passes"] / peak, color="#1f6feb",
                        alpha=0.45, ax=ax, zorder=2)
    for n in net["nodes"]:
        pitch.scatter(n["avg_x"], n["avg_y"], s=120 + 25 * n["passes"], ax=ax, c="#e5534b",
                      edgecolors="#222", zorder=3)
        ax.annotate(_surname(n["player"]), (n["avg_x"], n["avg_y"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8, zorder=4)
    ax.set_title(f"{net['team']}: passing network (until first substitution)", fontsize=11)
    return fig


def zone_heatmap(a: MatchAnalytics, team: str, action: str = "passes"):
    z = a.zone_usage(team, action)
    pitch = Pitch(**_PITCH)
    fig, ax = pitch.draw(figsize=(8, 5.4))
    peak = max(v for t in THIRDS for v in z["percent_by_zone"][t].values()) or 1
    for i, t in enumerate(THIRDS):
        for j, c in enumerate(CHANNELS):
            v = z["percent_by_zone"][t][c]
            ax.add_patch(Rectangle((40 * i, 80 / 3 * j), 40, 80 / 3, color="#1f6feb",
                                   alpha=0.08 + 0.7 * v / peak, zorder=1))
            ax.text(40 * i + 20, 80 / 3 * j + 80 / 6, f"{v:.1f}%", ha="center", va="center",
                    fontsize=10, zorder=3)
    ax.set_title(f"{z['team']}: where {action} start (attacking to the right)", fontsize=11)
    return fig


def xg_timeline(a: MatchAnalytics, team: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    end = float(a.events["minute"].max()) + 1
    for team_name, color in zip(a.teams, COLORS):
        s = a._of(team_name, "Shot")
        t = (s["minute"] + s["second"] / 60).to_numpy()
        cum = s["shot_xg"].fillna(0).cumsum().to_numpy()
        ax.step(np.r_[0, t, end], np.r_[0, cum, cum[-1] if len(cum) else 0], where="post",
                color=color, lw=2, label=f"{team_name} ({cum[-1] if len(cum) else 0:.2f} xG)")
    for g in a.goals()["goals"]:
        m, sec = g["clock"].split(":")
        ax.axvline(int(m) + int(sec) / 60, color="#888", ls=":", lw=1)
        ax.text(int(m) + int(sec) / 60, ax.get_ylim()[1] * 0.98, f"{g['scorer'].split()[-1]}",
                rotation=90, va="top", ha="right", fontsize=7, color="#555")
    ax.set_xlabel("Match clock (minutes)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Cumulative xG (dotted lines = goals)", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def render(a: MatchAnalytics, chart: str, team: str | None = None):
    if chart not in CHARTS:
        raise ValueError(f"chart must be one of {CHARTS}")
    if chart == "xg_timeline":
        return xg_timeline(a)
    team = team or a.teams[0]
    return {"shot_map": shot_map, "pass_network": pass_network, "zone_heatmap": zone_heatmap}[chart](a, team)
