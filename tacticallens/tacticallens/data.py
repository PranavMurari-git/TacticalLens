"""Load StatsBomb Open Data (free JSON files on GitHub) into a flat pandas DataFrame.

Data licence: StatsBomb open data is free to use but requires attribution to StatsBomb.
Pitch coordinates are 120 x 80, and every team always attacks towards x = 120.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE_DIR = Path(
    os.environ.get("TACTICALLENS_CACHE", Path(__file__).resolve().parent.parent / "data_cache")
)

COLUMNS = [
    "id", "index", "period", "minute", "second", "type", "team", "player", "position",
    "possession", "possession_team", "play_pattern", "x", "y", "end_x", "end_y",
    "under_pressure", "pass_recipient", "pass_outcome", "pass_type", "pass_length",
    "pass_shot_assist", "pass_goal_assist", "shot_xg", "shot_outcome", "shot_type",
    "shot_body_part",
]


def _fetch(path: str):
    """GET a JSON file from the StatsBomb repo, with an on-disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / path.replace("/", "__")
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    resp = requests.get(f"{BASE_URL}/{path}", timeout=60)
    resp.raise_for_status()
    cache_file.write_bytes(resp.content)
    return json.loads(resp.content.decode("utf-8"))


def list_competitions() -> pd.DataFrame:
    df = pd.DataFrame(_fetch("competitions.json"))
    cols = ["competition_id", "season_id", "competition_name", "season_name", "country_name"]
    return df[cols].sort_values(["competition_name", "season_name"], ascending=[True, False])


def list_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    rows = _fetch(f"matches/{competition_id}/{season_id}.json")
    out = []
    for m in rows:
        out.append({
            "match_id": m["match_id"],
            "date": m["match_date"],
            "home": m["home_team"]["home_team_name"],
            "away": m["away_team"]["away_team_name"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "stage": (m.get("competition_stage") or {}).get("name", ""),
            "competition": m["competition"]["competition_name"],
            "season": m["season"]["season_name"],
        })
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def load_raw_events(match_id: int) -> list[dict]:
    return _fetch(f"events/{match_id}.json")


def _name(d: dict | None) -> str | None:
    return d.get("name") if d else None


def flatten_events(raw: list[dict]) -> pd.DataFrame:
    """Turn nested StatsBomb event JSON into one row per event with the fields we use."""
    rows = []
    for e in raw:
        loc = e.get("location") or [None, None]
        p, s, c = e.get("pass", {}), e.get("shot", {}), e.get("carry", {})
        end = p.get("end_location") or s.get("end_location") or c.get("end_location") or [None, None]
        rows.append({
            "id": e["id"], "index": e["index"], "period": e["period"],
            "minute": e["minute"], "second": e["second"], "type": e["type"]["name"],
            "team": _name(e.get("team")), "player": _name(e.get("player")),
            "position": _name(e.get("position")), "possession": e["possession"],
            "possession_team": _name(e.get("possession_team")),
            "play_pattern": _name(e.get("play_pattern")),
            "x": loc[0], "y": loc[1], "end_x": end[0], "end_y": end[1],
            "under_pressure": bool(e.get("under_pressure")),
            "pass_recipient": _name(p.get("recipient")),
            "pass_outcome": _name(p.get("outcome")),  # None means the pass was completed
            "pass_type": _name(p.get("type")),
            "pass_length": p.get("length"),
            "pass_shot_assist": bool(p.get("shot_assist")),
            "pass_goal_assist": bool(p.get("goal_assist")),
            "shot_xg": s.get("statsbomb_xg"),
            "shot_outcome": _name(s.get("outcome")),
            "shot_type": _name(s.get("type")),
            "shot_body_part": _name(s.get("body_part")),
        })
    return make_events_df(rows)


def make_events_df(rows: list[dict]) -> pd.DataFrame:
    """Build an events DataFrame with every expected column present (handy for tests)."""
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].copy()
    for col in ("x", "y", "end_x", "end_y", "pass_length", "shot_xg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("under_pressure", "pass_shot_assist", "pass_goal_assist"):
        df[col] = df[col].fillna(False).astype(bool)
    return df


def load_events(match_id: int) -> pd.DataFrame:
    return flatten_events(load_raw_events(match_id))
