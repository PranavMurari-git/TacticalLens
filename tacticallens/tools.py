"""Tool definitions the LLM can call, plus the dispatcher that runs them.

The LLM never computes statistics. It chooses a tool and arguments (a structured analytical
request); the Python function in analytics.py does the maths; the result goes back to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .analytics import ZONE_ACTIONS, MatchAnalytics
from .charts import CHARTS

_TEAM = {"type": "string", "description": "Team name as it appears in the match, e.g. 'France'."}
_TOP_N = {"type": "integer", "description": "How many rows to return (default 5).", "minimum": 1, "maximum": 15}


def _spec(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props or {}, "required": required or []}}


TOOL_SPECS: list[dict] = [
    _spec("get_match_overview",
          "Score, shots, shots on target, xG, passes, pass accuracy and pass share for both teams. "
          "Start here for broad questions about the match."),
    _spec("get_goals", "Every goal in order: clock, scorer, team, xG, own-goal flag."),
    _spec("get_shots",
          "List individual shots (clock, player, xG, outcome, type, distance). Use for 'best chances' "
          "or 'which shots...' questions.",
          {"team": _TEAM, "min_xg": {"type": "number", "description": "Only shots with xG >= this."},
           "limit": {"type": "integer", "description": "Max shots to list (default 15)."}}),
    _spec("get_team_shot_analysis",
          "Chance-creation summary for one team: shots, on target, goals, xG, npxG, xG per shot, big "
          "chances, shots in the box, shot types, top shooters, key passes and chance creators.",
          {"team": _TEAM}, ["team"]),
    _spec("get_passing_stats",
          "Passing summary for one team: attempted, completed, accuracy, average length, forward pass %, "
          "passes into final third and box. Use period='compare_halves' for first vs second half.",
          {"team": _TEAM, "period": {"type": "string", "enum": ["all", "first_half", "second_half", "compare_halves"]}},
          ["team"]),
    _spec("get_progressive_passes",
          "Progressive passes (completed, non-restart, ending >=25% and >=10 units closer to goal) with the "
          "top players. Use for 'who moved the ball forward best' questions.",
          {"team": _TEAM, "top_n": _TOP_N}, ["team"]),
    _spec("get_player_involvement",
          "Per-player on-ball involvement (passes, carries, shots, key passes) and involvement in "
          "shot-ending possession sequences. Use for 'which players were most involved'.",
          {"team": _TEAM, "top_n": _TOP_N}, ["team"]),
    _spec("get_attacking_sequences",
          "Possession sequences that ended in shots: how many, how they started (open play, counter, "
          "set piece), passes per sequence, and the highest-xG sequences with players involved.",
          {"team": _TEAM, "top_n": _TOP_N}, ["team"]),
    _spec("get_zone_usage",
          "3x3 pitch-zone breakdown (thirds x left/centre/right channels) of where a team's actions start.",
          {"team": _TEAM, "action": {"type": "string", "enum": list(ZONE_ACTIONS)}}, ["team"]),
    _spec("get_passing_network",
          "Average player positions and the most frequent pass links until the first substitution.",
          {"team": _TEAM}, ["team"]),
    _spec("find_events",
          "Look up raw events by team, event type (e.g. 'Pass', 'Shot', 'Foul Committed', 'Substitution'), "
          "player name fragment and minute range. Use for 'what happened between minute X and Y'.",
          {"team": _TEAM, "event_type": {"type": "string"}, "player": {"type": "string"},
           "minute_from": {"type": "integer"}, "minute_to": {"type": "integer"},
           "limit": {"type": "integer"}}),
    _spec("show_chart",
          "Display a chart to the user under your answer. shot_map / pass_network / zone_heatmap need a team; "
          "xg_timeline shows both teams.",
          {"chart": {"type": "string", "enum": list(CHARTS)}, "team": _TEAM}, ["chart"]),
]


@dataclass
class ToolContext:
    """Collects side effects of a question: charts to show and events cited as evidence."""
    charts: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    def add_evidence(self, items: list[dict]) -> None:
        seen = {e["event_id"] for e in self.evidence}
        self.evidence += [e for e in items if e["event_id"] not in seen]


def json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def dispatch(a: MatchAnalytics, name: str, args: dict, ctx: ToolContext) -> dict:
    """Run one tool call. Errors are returned as data so the LLM can correct itself."""
    handlers: dict[str, Callable[..., dict]] = {
        "get_match_overview": lambda: a.overview(),
        "get_goals": lambda: a.goals(),
        "get_shots": lambda: a.shots(args.get("team"), args.get("min_xg", 0.0), args.get("limit", 15)),
        "get_team_shot_analysis": lambda: a.team_shot_analysis(args["team"]),
        "get_passing_stats": lambda: (a.compare_halves(args["team"]) if args.get("period") == "compare_halves"
                                      else a.passing_stats(args["team"], args.get("period", "all"))),
        "get_progressive_passes": lambda: a.progressive_passes(args["team"], args.get("top_n", 5)),
        "get_player_involvement": lambda: a.player_involvement(args["team"], args.get("top_n", 8)),
        "get_attacking_sequences": lambda: a.attacking_sequences(args["team"], args.get("top_n", 5)),
        "get_zone_usage": lambda: a.zone_usage(args["team"], args.get("action", "passes")),
        "get_passing_network": lambda: a.passing_network(args["team"]),
        "find_events": lambda: a.find_events(args.get("team"), args.get("event_type"), args.get("player"),
                                             args.get("minute_from"), args.get("minute_to"), args.get("limit", 15)),
        "show_chart": lambda: _show_chart(a, args, ctx),
    }
    if name not in handlers:
        return {"error": f"Unknown tool '{name}'. Available: {list(handlers)}"}
    try:
        out = handlers[name]()
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    ctx.add_evidence(out.get("evidence", []) + out.get("events", []))
    return out


def _show_chart(a: MatchAnalytics, args: dict, ctx: ToolContext) -> dict:
    chart = args["chart"]
    if chart not in CHARTS:
        raise ValueError(f"chart must be one of {CHARTS}")
    team = a.resolve_team(args["team"]) if args.get("team") else None
    if chart != "xg_timeline" and team is None:
        raise ValueError(f"{chart} needs a team")
    ctx.charts.append({"chart": chart, "team": team})
    return {"status": "chart will be displayed under your answer", "chart": chart, "team": team}
