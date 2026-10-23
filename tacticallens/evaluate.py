"""Evaluation: does the tool-using agent beat the same LLM without tools?

Ground truth is computed by INDEPENDENT plain-Python loops over the raw StatsBomb JSON (not by
analytics.py), so a bug in the analytics engine cannot silently agree with itself.

Usage:
  python -m tacticallens.evaluate --match-id 8658 --dry-run          # show questions + truth, no API
  python -m tacticallens.evaluate --match-id 8658 --mode both         # needs ANTHROPIC_API_KEY
Tip: famous matches leak into LLM training data. Also run on an obscure match to see the real gap.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from .agent import TacticalAgent
from .analytics import MatchAnalytics
from .data import flatten_events, load_raw_events

ON_TARGET = {"Goal", "Saved", "Saved to Post"}
LIMITATION_WORDS = ("tracking", "off-ball", "cannot", "can't", "not available", "does not include",
                    "doesn't include", "unable", "limitation", "no data", "not possible")


def _pname(e, key):  # nested name helper
    return (e.get(key) or {}).get("name")


def build_questions(raw: list[dict]) -> list[dict]:
    """Return question dicts: {id, category, question, numbers, tol, text, tool}."""
    ev = [e for e in raw if e["period"] <= 4]
    teams = list(dict.fromkeys(_pname(e, "team") for e in ev if _pname(e, "team")))
    A, B = teams[0], teams[1]
    shots = lambda t: [e for e in ev if _pname(e, "type") == "Shot" and _pname(e, "team") == t]
    passes = lambda t: [e for e in ev if _pname(e, "type") == "Pass" and _pname(e, "team") == t]
    done = lambda p: "outcome" not in p["pass"]
    goals = {t: sum(1 for s in shots(t) if s["shot"]["outcome"]["name"] == "Goal") for t in teams}
    for e in ev:  # own goals count for the beneficiary
        if _pname(e, "type") == "Own Goal For":
            goals[_pname(e, "team")] += 1

    qs = []
    add = lambda **k: qs.append({"numbers": [], "tol": 0.0, "text": [], "tool": None, **k})

    add(id="score", category="lookup", question="What was the final score of this match (excluding any shootout)?",
        numbers=[goals[A], goals[B]], tool="get_match_overview")
    add(id="shots_A", category="lookup", question=f"How many shots did {A} take?",
        numbers=[len(shots(A))], tool="get_match_overview")
    add(id="sot_B", category="lookup", question=f"How many of {B}'s shots were on target?",
        numbers=[sum(s["shot"]["outcome"]["name"] in ON_TARGET for s in shots(B))], tool="get_match_overview")
    add(id="xg_A", category="metric", question=f"What was {A}'s total expected goals (xG) in this match?",
        numbers=[sum(s["shot"]["statsbomb_xg"] for s in shots(A))], tol=0.06, tool="get_team_shot_analysis")
    pa = passes(A)
    add(id="pass_acc_A", category="metric", question=f"What was {A}'s pass completion percentage?",
        numbers=[100 * sum(done(p) for p in pa) / len(pa)], tol=0.6, tool="get_match_overview")
    p2 = [p for p in pa if p["period"] == 2]
    add(id="completed_2h_A", category="metric", question=f"How many passes did {A} complete in the second half?",
        numbers=[sum(done(p) for p in p2)], tool="get_passing_stats")
    cnt = Counter(_pname(s, "player") for s in shots(B))
    top, n = max(cnt.items(), key=lambda kv: kv[1])
    ties = [p for p, c in cnt.items() if c == n]
    if len(ties) == 1:
        add(id="top_shooter_B", category="ranking", question=f"Which {B} player took the most shots, and how many?",
            numbers=[n], text=[top.split()[-1]], tool="get_team_shot_analysis")
    best = max(shots(A), key=lambda s: s["shot"]["statsbomb_xg"])
    add(id="best_chance_A", category="ranking", question=f"Which {A} shot had the highest xG, who took it and what was the xG?",
        numbers=[best["shot"]["statsbomb_xg"]], tol=0.006, text=[_pname(best, "player").split()[-1]], tool="get_shots")

    def prog(t):
        n = 0
        for p in passes(t):
            if not done(p) or _pname(p["pass"], "type") in {"Corner", "Goal Kick", "Kick Off"}:
                continue
            (x, y), (ex, ey) = p["location"], p["pass"]["end_location"]
            d0, d1 = math.hypot(120 - x, 40 - y), math.hypot(120 - ex, 40 - ey)
            n += (d1 <= 0.75 * d0) and (d0 - d1 >= 10)
        return n
    add(id="prog_B", category="defined-metric",
        question=f"How many progressive passes did {B} make? (Use the standard tool definition.)",
        numbers=[prog(B)], tool="get_progressive_passes")
    first = min((e for e in ev if (_pname(e, "type") == "Shot" and e["shot"]["outcome"]["name"] == "Goal")
                 or _pname(e, "type") == "Own Goal Against"), key=lambda e: e["index"])
    add(id="first_goal_minute", category="lookup",
        question="In which minute value (the 'minute' field, 0-indexed from kick-off) of the data was the first goal scored?",
        numbers=[first["minute"]], tol=1.0, tool="get_goals")
    add(id="limitation", category="limitation",
        question=f"What was {B}'s average defensive line height, measured in metres from their own goal?",
        text=list(LIMITATION_WORDS), tool=None)
    return qs


def extract_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


def grade(q: dict, answer: str) -> bool:
    low = answer.lower()
    if q["category"] == "limitation":
        return any(w in low for w in q["text"])
    nums = extract_numbers(answer)
    ok_nums = all(any(abs(v - want) <= q["tol"] + 1e-9 for v in nums) for want in q["numbers"])
    ok_text = all(t.lower() in low for t in q["text"]) if q["text"] else True
    return ok_nums and ok_text


def run(match_id: int, mode: str, model: str | None, label: str | None, dry: bool) -> dict:
    raw = load_raw_events(match_id)
    qs = build_questions(raw)
    if dry:
        for q in qs:
            print(f"[{q['category']:<14}] {q['question']}\n    truth: numbers={[round(n, 3) for n in q['numbers']]} text={q['text'][:3]}")
        return {}
    analytics = MatchAnalytics(flatten_events(raw), label)
    agent = TacticalAgent(analytics, model=model)
    results = []
    for q in qs:
        row = {"id": q["id"], "category": q["category"], "question": q["question"]}
        if mode in ("tools", "both"):
            r = agent.ask(q["question"])
            row |= {"tools_answer": r.answer, "tools_correct": grade(q, r.answer),
                    "tools_used": [c["name"] for c in r.tool_calls],
                    "right_tool": q["tool"] is None or q["tool"] in [c["name"] for c in r.tool_calls]}
        if mode in ("baseline", "both"):
            r = agent.ask_without_tools(q["question"])
            row |= {"baseline_answer": r.answer, "baseline_correct": grade(q, r.answer)}
        results.append(row)
        print(f"{q['id']:<18} tools={row.get('tools_correct', '-')!s:<6} baseline={row.get('baseline_correct', '-')}")
    summary = {}
    for key in ("tools_correct", "baseline_correct", "right_tool"):
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[key] = f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)"
    out = {"match_id": match_id, "model": agent.model, "summary": summary, "results": results}
    folder = Path(__file__).resolve().parent.parent / "eval_results"
    folder.mkdir(exist_ok=True)
    path = folder / f"match{match_id}_{int(time.time())}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary), "\nSaved:", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", type=int, default=8658)
    ap.add_argument("--mode", choices=["tools", "baseline", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None, help="Match description given to the LLM, e.g. 'France vs Croatia, 2018 World Cup final'")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.match_id, a.mode, a.model, a.label, a.dry_run)
