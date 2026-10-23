"""Deterministic football analytics over StatsBomb event data.

Every number the LLM ever shows to a user comes from a function in this file.
Coordinates: pitch is 120 x 80, each team attacks towards x = 120, y = 0 is the attacker's
LEFT touchline. Units are roughly yards.

Every public method returns a JSON-serialisable dict. Many include an "evidence" list: the
actual match events (id, clock, player) behind the numbers, so claims can be checked.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

GOAL_X, GOAL_Y = 120.0, 40.0
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102.0, 18.0, 62.0
FINAL_THIRD_X = 80.0
BIG_CHANCE_XG = 0.30
ON_TARGET = {"Goal", "Saved", "Saved to Post"}
RESTART_PASSES = {"Corner", "Goal Kick", "Kick Off"}
ON_BALL_TYPES = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*"}
PROGRESSIVE_MIN_RATIO = 0.25   # end point must be >= 25% closer to goal centre ...
PROGRESSIVE_MIN_GAIN = 10.0    # ... and at least 10 units closer
FORWARD_PASS_MIN_DX = 5.0

ZONE_ACTIONS = {
    "passes": "Pass", "carries": "Carry", "shots": "Shot", "pressures": "Pressure",
    "ball_recoveries": "Ball Recovery", "ball_receipts": "Ball Receipt*",
}
THIRDS = ("defensive third", "middle third", "attacking third")
CHANNELS = ("left", "centre", "right")


def _r(x: Any, nd: int = 2):
    if x is None:
        return None
    try:
        if math.isnan(float(x)):
            return None
    except (TypeError, ValueError):
        return x
    return round(float(x), nd)


def _dist_to_goal(x, y):
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def _in_box(x, y):
    return (x >= BOX_X_MIN) & (y >= BOX_Y_MIN) & (y <= BOX_Y_MAX)


def _pct(num: float, den: float):
    return _r(100.0 * num / den) if den else None


class MatchAnalytics:
    def __init__(self, events: pd.DataFrame, label: str | None = None):
        self.had_shootout = bool((events["period"] == 5).any())
        df = events[events["period"] <= 4].sort_values("index").reset_index(drop=True)
        self.events = df
        self.teams = [t for t in df["team"].dropna().unique()]
        self.label = label or " vs ".join(self.teams)
        self._seq_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ helpers
    def resolve_team(self, name: str) -> str:
        """Match a user/LLM-supplied team name to a real team (case-insensitive, substring ok)."""
        low = (name or "").strip().lower()
        for t in self.teams:
            if t.lower() == low:
                return t
        hits = [t for t in self.teams if low and (low in t.lower() or t.lower() in low)]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f"Unknown team '{name}'. Valid teams: {self.teams}")

    def opponent(self, team: str) -> str:
        return [t for t in self.teams if t != team][0]

    def _of(self, team: str, etype: str) -> pd.DataFrame:
        d = self.events
        return d[(d["team"] == team) & (d["type"] == etype)]

    @staticmethod
    def _detail(r) -> str:
        if r["type"] == "Shot":
            xg = f"{r['shot_xg']:.2f}" if pd.notna(r["shot_xg"]) else "n/a"
            return f"{r['shot_outcome']}, xG {xg}, {r['shot_type']}, {r['shot_body_part']}"
        if r["type"] == "Pass":
            res = "complete" if pd.isna(r["pass_outcome"]) else r["pass_outcome"]
            return f"to {r['pass_recipient'] or 'unknown'} ({res})"
        return ""

    def _evidence(self, rows: pd.DataFrame, limit: int = 6) -> list[dict]:
        out = []
        for _, r in rows.head(limit).iterrows():
            out.append({
                "event_id": r["id"], "clock": f"{int(r['minute'])}:{int(r['second']):02d}",
                "period": int(r["period"]), "team": r["team"], "player": r["player"],
                "type": r["type"], "detail": self._detail(r),
            })
        return out

    @staticmethod
    def _period(period) -> int | None:
        mapping = {None: None, "all": None, "first_half": 1, "second_half": 2,
                   1: 1, 2: 2, "1": 1, "2": 2}
        if period not in mapping:
            raise ValueError("period must be one of: all, first_half, second_half")
        return mapping[period]

    # ------------------------------------------------------------------ goals / overview
    def goals(self) -> dict:
        """All goals in chronological order, including own goals (credited to the beneficiary)."""
        d = self.events
        rows = []
        for _, r in d[(d["type"] == "Shot") & (d["shot_outcome"] == "Goal")].iterrows():
            rows.append((r["index"], r["team"], r["player"], False, r))
        for _, r in d[d["type"] == "Own Goal Against"].iterrows():
            rows.append((r["index"], self.opponent(r["team"]), r["player"], True, r))
        rows.sort(key=lambda t: t[0])
        goals = [{
            "clock": f"{int(r['minute'])}:{int(r['second']):02d}", "team": team,
            "scorer": player, "own_goal": og, "event_id": r["id"],
            "xg": None if og else _r(r["shot_xg"]),
        } for _, team, player, og, r in rows]
        return {"goals": goals, "total": len(goals)}

    def _goals_for(self, team: str) -> int:
        return sum(1 for g in self.goals()["goals"] if g["team"] == team)

    def overview(self) -> dict:
        teams = {}
        total_passes = int((self.events["type"] == "Pass").sum())
        for t in self.teams:
            shots, passes = self._of(t, "Shot"), self._of(t, "Pass")
            teams[t] = {
                "goals": self._goals_for(t),
                "shots": len(shots),
                "shots_on_target": int(shots["shot_outcome"].isin(ON_TARGET).sum()),
                "xg": _r(shots["shot_xg"].sum()),
                "passes_attempted": len(passes),
                "passes_completed": int(passes["pass_outcome"].isna().sum()),
                "pass_accuracy_pct": _pct(passes["pass_outcome"].isna().sum(), len(passes)),
                "pass_share_pct": _pct(len(passes), total_passes),
            }
        a, b = self.teams[0], self.teams[1]
        note = "Pass share is a rough proxy for possession, not official possession time."
        if self.had_shootout:
            note += " A penalty shootout took place; its events are excluded from all statistics."
        return {
            "match": self.label,
            "score": f"{a} {teams[a]['goals']}-{teams[b]['goals']} {b}",
            "teams": teams,
            "went_to_extra_time": bool((self.events["period"] >= 3).any()),
            "notes": note,
        }

    # ------------------------------------------------------------------ shots
    def shots(self, team: str | None = None, min_xg: float = 0.0, limit: int = 15) -> dict:
        d = self.events[self.events["type"] == "Shot"]
        if team:
            d = d[d["team"] == self.resolve_team(team)]
        d = d[d["shot_xg"].fillna(0) >= min_xg]
        rows = [{
            "clock": f"{int(r['minute'])}:{int(r['second']):02d}", "team": r["team"],
            "player": r["player"], "xg": _r(r["shot_xg"], 3), "outcome": r["shot_outcome"],
            "shot_type": r["shot_type"], "body_part": r["shot_body_part"],
            "distance_to_goal": _r(_dist_to_goal(r["x"], r["y"]), 1),
            "play_pattern": r["play_pattern"], "event_id": r["id"],
        } for _, r in d.head(limit).iterrows()]
        return {"count": len(d), "shown": len(rows), "shots": rows}

    def team_shot_analysis(self, team: str) -> dict:
        team = self.resolve_team(team)
        s = self._of(team, "Shot")
        n = len(s)
        pens = s[s["shot_type"] == "Penalty"]
        dist = _dist_to_goal(s["x"], s["y"]) if n else pd.Series(dtype=float)
        inside = _in_box(s["x"], s["y"]) if n else pd.Series(dtype=bool)
        top = (s.groupby("player").agg(shots=("id", "size"), xg=("shot_xg", "sum"))
                .sort_values(["shots", "xg"], ascending=False).head(5))
        key_passes = self._of(team, "Pass")
        key_passes = key_passes[key_passes["pass_shot_assist"] | key_passes["pass_goal_assist"]]
        kp_players = key_passes.groupby("player").size().sort_values(ascending=False).head(5)
        goals, xg = self._goals_for(team), float(s["shot_xg"].sum())
        big = s[s["shot_xg"] >= BIG_CHANCE_XG].sort_values("shot_xg", ascending=False)
        return {
            "team": team, "shots": n,
            "shots_on_target": int(s["shot_outcome"].isin(ON_TARGET).sum()),
            "goals_scored_incl_own_goals": goals,
            "xg": _r(xg), "npxg": _r(s[s["shot_type"] != "Penalty"]["shot_xg"].sum()),
            "xg_per_shot": _r(xg / n) if n else None,
            "goals_minus_xg": _r(goals - xg),
            "big_chances_xg_ge_0.30": len(big),
            "shots_inside_box": int(inside.sum()) if n else 0,
            "avg_shot_distance": _r(dist.mean(), 1) if n else None,
            "by_shot_type": {k: int(v) for k, v in s["shot_type"].value_counts().items()},
            "by_body_part": {k: int(v) for k, v in s["shot_body_part"].value_counts().items()},
            "by_outcome": {k: int(v) for k, v in s["shot_outcome"].value_counts().items()},
            "penalties": len(pens),
            "top_shooters": [{"player": p, "shots": int(r.shots), "xg": _r(r.xg)}
                             for p, r in top.iterrows()],
            "key_passes_total": len(key_passes),
            "top_chance_creators": [{"player": p, "key_passes": int(c)} for p, c in kp_players.items()],
            "evidence": self._evidence(big if len(big) else s.sort_values("shot_xg", ascending=False)),
        }

    # ------------------------------------------------------------------ passing
    def passing_stats(self, team: str, period="all") -> dict:
        team = self.resolve_team(team)
        p = self._of(team, "Pass")
        per = self._period(period)
        if per:
            p = p[p["period"] == per]
        done = p["pass_outcome"].isna()
        valid = p.dropna(subset=["x", "end_x"])
        completed = p[done].dropna(subset=["x", "end_x"])
        into_final = completed[(completed["x"] < FINAL_THIRD_X) & (completed["end_x"] >= FINAL_THIRD_X)]
        into_box = completed[~_in_box(completed["x"], completed["y"]) & _in_box(completed["end_x"], completed["end_y"])]
        return {
            "team": team, "period": str(period or "all"),
            "attempted": len(p), "completed": int(done.sum()),
            "accuracy_pct": _pct(done.sum(), len(p)),
            "avg_pass_length": _r(p["pass_length"].mean(), 1),
            "forward_pass_pct": _pct(((valid["end_x"] - valid["x"]) >= FORWARD_PASS_MIN_DX).sum(), len(valid)),
            "completed_into_final_third": len(into_final),
            "completed_into_box": len(into_box),
            "under_pressure_pct": _pct(p["under_pressure"].sum(), len(p)),
            "evidence": self._evidence(into_box.sort_values("index")),
        }

    def compare_halves(self, team: str) -> dict:
        a, b = self.passing_stats(team, "first_half"), self.passing_stats(team, "second_half")
        for r in (a, b):
            r.pop("evidence", None)
        return {"team": a["team"], "first_half": a, "second_half": b}

    def progressive_passes(self, team: str, top_n: int = 5) -> dict:
        """Completed, non-restart passes ending >=25% closer to goal centre AND >=10 units closer."""
        team = self.resolve_team(team)
        p = self._of(team, "Pass")
        p = p[p["pass_outcome"].isna() & ~p["pass_type"].isin(RESTART_PASSES)]
        p = p.dropna(subset=["x", "y", "end_x", "end_y"]).copy()
        d0, d1 = _dist_to_goal(p["x"], p["y"]), _dist_to_goal(p["end_x"], p["end_y"])
        p["gain"] = d0 - d1
        prog = p[(d1 <= (1 - PROGRESSIVE_MIN_RATIO) * d0) & (p["gain"] >= PROGRESSIVE_MIN_GAIN)]
        by_player = prog.groupby("player").agg(progressive_passes=("id", "size"), total_gain=("gain", "sum"))
        by_player = by_player.sort_values(["progressive_passes", "total_gain"], ascending=False).head(top_n)
        return {
            "team": team, "total_progressive_passes": len(prog),
            "definition": "completed non-restart pass ending >=25% closer to goal centre and >=10 units closer",
            "top_players": [{"player": k, "progressive_passes": int(r.progressive_passes),
                             "total_distance_gained": _r(r.total_gain, 1)} for k, r in by_player.iterrows()],
            "evidence": self._evidence(prog.sort_values("gain", ascending=False)),
        }

    def passing_network(self, team: str, min_passes: int = 3) -> dict:
        """Completed pass links until the team's first substitution (a stable line-up)."""
        team = self.resolve_team(team)
        subs = self.events[(self.events["type"] == "Substitution") & (self.events["team"] == team)]
        cutoff = subs["index"].min() if len(subs) else self.events["index"].max() + 1
        p = self._of(team, "Pass")
        p = p[(p["index"] < cutoff) & p["pass_outcome"].isna() & p["pass_recipient"].notna()]
        edges = (p.groupby(["player", "pass_recipient"]).size().reset_index(name="passes")
                  .query("passes >= @min_passes").sort_values("passes", ascending=False))
        nodes = p.groupby("player").agg(avg_x=("x", "mean"), avg_y=("y", "mean"), passes=("id", "size"))
        return {
            "team": team, "built_from": "completed passes before the team's first substitution",
            "nodes": [{"player": k, "avg_x": _r(r.avg_x, 1), "avg_y": _r(r.avg_y, 1), "passes": int(r.passes)}
                      for k, r in nodes.iterrows()],
            "top_links": [{"from": r.player, "to": r.pass_recipient, "passes": int(r.passes)}
                          for r in edges.head(10).itertuples()],
        }

    # ------------------------------------------------------------------ possessions / sequences
    def _sequences(self, team: str) -> pd.DataFrame:
        if team in self._seq_cache:
            return self._seq_cache[team]
        d = self.events[self.events["possession_team"] == team]
        rows = []
        for poss, g in d.groupby("possession", sort=True):
            mine = g[g["team"] == team]
            shots = mine[mine["type"] == "Shot"]
            first, last = g.iloc[0], g.iloc[-1]
            clock = lambda r: r["minute"] * 60 + r["second"]
            outcome = None
            if len(shots):
                outcome = "Goal" if (shots["shot_outcome"] == "Goal").any() else shots.iloc[-1]["shot_outcome"]
            rows.append({
                "possession": int(poss), "clock": f"{int(first['minute'])}:{int(first['second']):02d}",
                "play_pattern": first["play_pattern"], "start_x": first["x"], "start_y": first["y"],
                "passes": int((mine["type"] == "Pass").sum()),
                "duration_s": int(max(clock(last) - clock(first), 0)),
                "players": sorted(set(mine[mine["type"].isin(ON_BALL_TYPES)]["player"].dropna())),
                "ends_in_shot": bool(len(shots)), "xg": float(shots["shot_xg"].sum()),
                "outcome": outcome,
                "shot_event_ids": list(shots["id"]),
            })
        out = pd.DataFrame(rows)
        self._seq_cache[team] = out
        return out

    def attacking_sequences(self, team: str, top_n: int = 5) -> dict:
        team = self.resolve_team(team)
        seq = self._sequences(team)
        shot_seq = seq[seq["ends_in_shot"]]
        top = shot_seq.sort_values("xg", ascending=False).head(top_n)
        shots = self.events.set_index("id")
        evidence = []
        for ids in top["shot_event_ids"]:
            evidence += self._evidence(shots.loc[ids].reset_index().head(1), limit=1)
        return {
            "team": team, "possessions": len(seq), "shot_ending_sequences": len(shot_seq),
            "shot_sequence_pct": _pct(len(shot_seq), len(seq)),
            "avg_passes_in_shot_sequences": _r(shot_seq["passes"].mean(), 1) if len(shot_seq) else None,
            "avg_passes_in_other_possessions": _r(seq[~seq["ends_in_shot"]]["passes"].mean(), 1),
            "shot_sequences_by_play_pattern": {k: int(v) for k, v in shot_seq["play_pattern"].value_counts().items()},
            "top_sequences_by_xg": [{
                "starts_at": r.clock, "play_pattern": r.play_pattern, "passes": int(r.passes),
                "duration_s": int(r.duration_s), "players_involved": r.players,
                "sequence_xg": _r(r.xg), "outcome": r.outcome,
            } for r in top.itertuples()],
            "evidence": evidence,
        }

    def player_involvement(self, team: str, top_n: int = 8) -> dict:
        team = self.resolve_team(team)
        d = self.events[(self.events["team"] == team) & self.events["player"].notna()]
        on_ball = d[d["type"].isin(ON_BALL_TYPES)]
        seq = self._sequences(team)
        in_shot_seq = pd.Series(
            [p for plist in seq[seq["ends_in_shot"]]["players"] for p in plist]).value_counts()
        rows = []
        for player, g in on_ball.groupby("player"):
            passes = g[g["type"] == "Pass"]
            rows.append({
                "player": player,
                "main_position": d[d["player"] == player]["position"].mode().iat[0],
                "on_ball_events": len(g), "passes_attempted": len(passes),
                "passes_completed": int(passes["pass_outcome"].isna().sum()),
                "carries": int((g["type"] == "Carry").sum()),
                "shots": int((g["type"] == "Shot").sum()),
                "xg": _r(g["shot_xg"].sum()),
                "key_passes": int(passes["pass_shot_assist"].sum() + passes["pass_goal_assist"].sum()),
                "shot_sequences_involved": int(in_shot_seq.get(player, 0)),
            })
        rows.sort(key=lambda r: r["on_ball_events"], reverse=True)
        by_seq = sorted(rows, key=lambda r: r["shot_sequences_involved"], reverse=True)[:3]
        return {
            "team": team, "players": rows[:top_n],
            "most_involved_in_shot_sequences": [{"player": r["player"], "sequences": r["shot_sequences_involved"]} for r in by_seq],
            "definition": "on_ball_events = passes + carries + shots + dribbles + ball receipts",
        }

    # ------------------------------------------------------------------ zones
    def zone_usage(self, team: str, action: str = "passes") -> dict:
        team = self.resolve_team(team)
        if action not in ZONE_ACTIONS:
            raise ValueError(f"action must be one of {list(ZONE_ACTIONS)}")
        d = self._of(team, ZONE_ACTIONS[action]).dropna(subset=["x", "y"])
        third = pd.cut(d["x"], [-0.1, 40, 80, 120.1], labels=THIRDS)
        chan = pd.cut(d["y"], [-0.1, 80 / 3, 160 / 3, 80.1], labels=CHANNELS)
        counts = pd.crosstab(third, chan).reindex(index=THIRDS, columns=CHANNELS, fill_value=0)
        pct = (100 * counts / max(counts.values.sum(), 1)).round(1)
        best = counts.stack().idxmax() if counts.values.sum() else (None, None)
        return {
            "team": team, "action": action, "total": int(counts.values.sum()),
            "percent_by_zone": {t: {c: float(pct.loc[t, c]) for c in CHANNELS} for t in THIRDS},
            "percent_by_third": {t: _r(pct.loc[t].sum(), 1) for t in THIRDS},
            "percent_by_channel": {c: _r(pct[c].sum(), 1) for c in CHANNELS},
            "busiest_zone": f"{best[0]}, {best[1]} channel",
            "note": "Location is where the action started; 'left' is the team's own left when attacking.",
        }

    # ------------------------------------------------------------------ generic lookup
    def find_events(self, team: str | None = None, event_type: str | None = None,
                    player: str | None = None, minute_from: int | None = None,
                    minute_to: int | None = None, limit: int = 15) -> dict:
        d = self.events
        if team:
            d = d[d["team"] == self.resolve_team(team)]
        if event_type:
            d = d[d["type"].str.lower() == event_type.lower()]
        if player:
            d = d[d["player"].fillna("").str.lower().str.contains(player.lower())]
        if minute_from is not None:
            d = d[d["minute"] >= minute_from]
        if minute_to is not None:
            d = d[d["minute"] <= minute_to]
        return {"count": len(d), "events": self._evidence(d, limit=limit),
                "available_types": sorted(self.events["type"].unique())[:40]}
