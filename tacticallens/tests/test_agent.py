"""Tests the agent loop with a scripted fake LLM client (no API key, no network)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tacticallens.agent import TacticalAgent
from tests.test_analytics import match  # noqa: F401  (pytest fixture)


class FakeClient:
    """First call asks for a tool, second call writes the final answer."""
    def __init__(self):
        self.calls = []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        usage = NS(input_tokens=10, output_tokens=5)
        if len(self.calls) == 1:
            block = NS(type="tool_use", id="t1", name="get_team_shot_analysis", input={"team": "red"})
            return NS(stop_reason="tool_use", content=[block], usage=usage)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Red had 1 shot.")], usage=usage)


def test_agent_executes_tool_and_returns_answer(match):  # noqa: F811
    agent = TacticalAgent.__new__(TacticalAgent)
    agent.analytics, agent.model, agent.max_steps, agent.client = match, "fake", 4, FakeClient()
    res = agent.ask("How many shots did Red have?")
    assert res.answer == "Red had 1 shot."
    assert [c["name"] for c in res.tool_calls] == ["get_team_shot_analysis"]
    assert res.tool_calls[0]["output"]["shots"] == 1
    # second request must carry the tool_result back to the model
    last_user = agent.client.calls[1]["messages"][-1]
    assert last_user["content"][0]["type"] == "tool_result"
    assert res.input_tokens == 20 and res.evidence
