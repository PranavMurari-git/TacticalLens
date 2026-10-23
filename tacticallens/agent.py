"""The LLM layer: question -> tool calls -> grounded answer (plus a no-tools baseline)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic

from .analytics import MatchAnalytics
from .tools import TOOL_SPECS, ToolContext, dispatch, json_default

DEFAULT_MODEL = os.environ.get("TACTICALLENS_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are TacticalLens, a football analyst for ONE match: {label}.
Your only source of truth is the analytics tools, which compute everything from StatsBomb open event data.

Rules
1. Never state a statistic, count, xG value, minute or player involvement that did not come from a tool result in this conversation. If you need a number, call a tool. Only compare or subtract numbers the tools returned.
2. Pick the tools that fit the question and call several when needed (for example, both teams for a comparison).
3. Structure answers like this: a one-line headline answer; then "Measured facts" (numbers, with the metric definition when it is non-obvious, and clock references such as 37:56 taken from tool evidence); then "Tactical interpretation", clearly labelled as interpretation and hedged.
4. Event data has no off-ball player positions, so it cannot show defensive-line height, pressing shape, or compactness. If asked, say so plainly and offer the closest measurable proxy.
5. Clocks are elapsed match time in mm:ss (17:56 is in the 18th minute). Penalty-shootout events are excluded. Own goals are credited to the beneficiary team.
6. If a chart would help, call show_chart and refer to it in your answer.
7. Keep answers under about 250 words unless asked for more. If a tool returns an error, fix the call and retry."""

BASELINE_PROMPT = ("You are a football analyst. Answer questions about this match: {label}. "
                   "You have no data tools; use only what you already know. Give specific numbers when asked.")


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class TacticalAgent:
    def __init__(self, analytics: MatchAnalytics, model: str | None = None,
                 api_key: str | None = None, max_steps: int = 8):
        self.analytics = analytics
        self.model = model or DEFAULT_MODEL
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.max_steps = max_steps

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """Answer with tools. `history` is a list of earlier {'role', 'content': str} turns."""
        ctx, calls = ToolContext(), []
        messages = list(history or []) + [{"role": "user", "content": question}]
        tokens_in = tokens_out = 0
        resp = None
        for _ in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1800, tools=TOOL_SPECS, messages=messages,
                system=SYSTEM_PROMPT.format(label=self.analytics.label))
            tokens_in += resp.usage.input_tokens
            tokens_out += resp.usage.output_tokens
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                out = dispatch(self.analytics, block.name, dict(block.input), ctx)
                calls.append({"name": block.name, "input": dict(block.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out, default=json_default),
                                "is_error": "error" in out})
            messages.append({"role": "user", "content": results})
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "tool_use" or not text:
            text = text or "I could not finish the analysis within the step limit. Try a narrower question."
        return AgentResult(text, calls, ctx.charts, ctx.evidence, tokens_in, tokens_out)

    def ask_without_tools(self, question: str) -> AgentResult:
        """Baseline for evaluation: the same LLM, but no tools and no data."""
        resp = self.client.messages.create(
            model=self.model, max_tokens=1000,
            system=BASELINE_PROMPT.format(label=self.analytics.label),
            messages=[{"role": "user", "content": question}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return AgentResult(text, [], [], [], resp.usage.input_tokens, resp.usage.output_tokens)
