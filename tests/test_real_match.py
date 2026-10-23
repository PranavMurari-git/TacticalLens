"""Integration test on the 2018 World Cup final (skipped when offline).

Cross-checks analytics.py against the independent ground truth used by the evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tacticallens.analytics import MatchAnalytics
from tacticallens.data import flatten_events, load_raw_events
from tacticallens.evaluate import build_questions


@pytest.fixture(scope="module")
def real():
    try:
        raw = load_raw_events(8658)
    except Exception:
        pytest.skip("StatsBomb data not reachable")
    return raw, MatchAnalytics(flatten_events(raw), "France vs Croatia")


def test_known_facts(real):
    _, a = real
    assert a.overview()["score"] == "France 4-2 Croatia"
    assert a.teams == ["France", "Croatia"] or set(a.teams) == {"France", "Croatia"}


def test_analytics_agree_with_independent_ground_truth(real):
    raw, a = real
    by_id = {q["id"]: q for q in build_questions(raw)}
    ov = a.overview()["teams"]
    assert ov["France"]["shots"] == by_id["shots_A"]["numbers"][0]
    assert ov["Croatia"]["shots_on_target"] == by_id["sot_B"]["numbers"][0]
    assert a.team_shot_analysis("France")["xg"] == pytest.approx(by_id["xg_A"]["numbers"][0], abs=0.006)
    assert ov["France"]["pass_accuracy_pct"] == pytest.approx(by_id["pass_acc_A"]["numbers"][0], abs=0.01)
    assert a.passing_stats("France", "second_half")["completed"] == by_id["completed_2h_A"]["numbers"][0]
    assert a.progressive_passes("Croatia")["total_progressive_passes"] == by_id["prog_B"]["numbers"][0]
