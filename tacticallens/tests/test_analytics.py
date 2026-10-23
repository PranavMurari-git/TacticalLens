"""Unit tests on a tiny hand-built match, so expected values are known by construction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tacticallens.analytics import MatchAnalytics
from tacticallens.data import make_events_df
from tacticallens.tools import ToolContext, dispatch


def _ev(i, type_, team, player=None, minute=1, period=1, **kw):
    return {"id": f"e{i}", "index": i, "period": period, "minute": minute, "second": 0, "type": type_,
            "team": team, "player": player, "possession": kw.pop("possession", 1),
            "possession_team": kw.pop("possession_team", team), "play_pattern": "Regular Play", **kw}


@pytest.fixture
def match():
    rows = [
        # Red: 4 passes (3 complete); one progressive (60,40)->(100,40); one into box
        _ev(1, "Pass", "Red", "Ann", x=60, y=40, end_x=100, end_y=40, pass_recipient="Bea", pass_length=40.0),
        _ev(2, "Pass", "Red", "Bea", x=100, y=40, end_x=108, end_y=40, pass_recipient="Cy", pass_shot_assist=True),
        _ev(3, "Pass", "Red", "Cy", x=30, y=10, end_x=33, end_y=12, pass_outcome="Incomplete"),
        _ev(4, "Pass", "Red", "Ann", x=20, y=70, end_x=22, end_y=70, pass_recipient="Bea"),
        _ev(5, "Shot", "Red", "Cy", x=108, y=40, shot_xg=0.5, shot_outcome="Goal", shot_type="Open Play", shot_body_part="Head"),
        # Blue: one shot saved, one penalty goal, one corner pass that must not count as progressive
        _ev(6, "Shot", "Blue", "Dee", x=100, y=30, shot_xg=0.1, shot_outcome="Saved", shot_type="Open Play",
            shot_body_part="Right Foot", possession=2),
        _ev(7, "Shot", "Blue", "Dee", x=108, y=40, shot_xg=0.76, shot_outcome="Goal", shot_type="Penalty",
            shot_body_part="Right Foot", possession=3, minute=50, period=2),
        _ev(8, "Pass", "Blue", "Eli", x=120, y=0, end_x=108, end_y=40, pass_type="Corner", possession=4, minute=60, period=2),
        # Own goal by Blue's player credited to Red
        _ev(9, "Own Goal Against", "Blue", "Fay", minute=70, period=2, possession=5),
        # Shootout event must be ignored everywhere
        _ev(10, "Shot", "Blue", "Dee", period=5, shot_xg=0.78, shot_outcome="Goal", shot_type="Penalty", minute=120),
    ]
    return MatchAnalytics(make_events_df(rows), "Red vs Blue")


def test_overview_score_includes_own_goal_and_ignores_shootout(match):
    ov = match.overview()
    assert ov["score"] == "Red 2-1 Blue"
    assert ov["teams"]["Blue"]["shots"] == 2  # the shootout shot is excluded
    assert match.had_shootout


def test_xg_and_shots_on_target(match):
    red = match.team_shot_analysis("Red")
    assert red["shots"] == 1 and red["xg"] == 0.5 and red["shots_on_target"] == 1
    blue = match.team_shot_analysis("blue")  # case-insensitive
    assert blue["xg"] == pytest.approx(0.86) and blue["npxg"] == pytest.approx(0.1)
    assert blue["shots_on_target"] == 2 and blue["penalties"] == 1


def test_pass_accuracy_and_halves(match):
    p = match.passing_stats("Red")
    assert (p["attempted"], p["completed"], p["accuracy_pct"]) == (4, 3, 75.0)
    assert match.passing_stats("Red", "second_half")["attempted"] == 0


def test_progressive_passes_rule(match):
    prog = match.progressive_passes("Red")
    assert prog["total_progressive_passes"] == 1  # only Ann's 60->100 pass
    assert prog["top_players"][0]["player"] == "Ann"
    assert match.progressive_passes("Blue")["total_progressive_passes"] == 0  # corner excluded


def test_zone_percentages_sum_to_100(match):
    z = match.zone_usage("Red", "passes")
    total = sum(v for t in z["percent_by_zone"].values() for v in t.values())
    assert total == pytest.approx(100, abs=0.2)


def test_sequences_and_goals(match):
    seq = match.attacking_sequences("Red")
    assert seq["possessions"] == 1 and seq["shot_ending_sequences"] == 1
    goals = match.goals()["goals"]
    assert [g["team"] for g in goals] == ["Red", "Blue", "Red"] and goals[2]["own_goal"]


def test_unknown_team_gives_helpful_error(match):
    with pytest.raises(ValueError, match="Valid teams"):
        match.resolve_team("Green")


def test_dispatch_returns_errors_as_data_and_queues_charts(match):
    ctx = ToolContext()
    assert "error" in dispatch(match, "get_team_shot_analysis", {"team": "Green"}, ctx)
    assert "error" in dispatch(match, "nope", {}, ctx)
    out = dispatch(match, "show_chart", {"chart": "shot_map", "team": "red"}, ctx)
    assert out["team"] == "Red" and ctx.charts == [{"chart": "shot_map", "team": "Red"}]
    dispatch(match, "get_team_shot_analysis", {"team": "Red"}, ctx)
    assert ctx.evidence and ctx.evidence[0]["event_id"] == "e5"
