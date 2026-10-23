# TacticalLens: an evidence-grounded AI football tactics analyst

Ask questions about a football match in plain English. An LLM turns each question into calls to
deterministic Python analytics functions (shots, xG, passing, progressive passes, possession
sequences, pitch zones), then explains the *measured* results and labels its tactical
*interpretation* separately. It cannot invent statistics: every number comes from a tool.

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data) (free, attribution required).

## Quick start
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
streamlit run app.py
```
The Dashboard tab works without a key. The first load of a match downloads its JSON into `data_cache/`.

## How it works
```
question -> LLM (tool schemas) -> tool call -> analytics.py (pandas) -> JSON result -> LLM -> answer
                                       \-> evidence events (id, clock, player) shown under the answer
```
| File | Role |
|---|---|
| `tacticallens/data.py` | Download + cache StatsBomb JSON, flatten to one DataFrame row per event |
| `tacticallens/analytics.py` | All metrics. Pure functions, JSON out, evidence events attached |
| `tacticallens/tools.py` | Tool schemas for the LLM + dispatcher (errors returned as data) |
| `tacticallens/agent.py` | System prompt, tool-use loop, no-tools baseline |
| `tacticallens/charts.py` | Shot map, passing network, zone heatmap, xG timeline |
| `tacticallens/evaluate.py` | Verified question set, independent ground truth, tools vs baseline |
| `app.py` | Streamlit UI (chat + dashboard) |

## Metric definitions (so you can defend them in an interview)
- **Pitch**: 120 x 80, every team attacks toward x = 120, y = 0 is the attacker's left.
- **xG**: StatsBomb's `statsbomb_xg` per shot (npxG excludes penalties). Penalty shootout excluded.
- **Goals**: shot goals + own goals credited to the beneficiary team.
- **Pass accuracy**: passes with no outcome tag / all passes. **Pass share** is only a proxy for possession.
- **Progressive pass**: completed, non-restart pass ending >= 25% and >= 10 units closer to goal centre.
- **Shot sequence**: a possession (StatsBomb `possession` id) containing at least one shot by that team.
- **Zones**: thirds (x < 40, 40-80, > 80) x channels (y thirds) where the action *starts*.

## Known limits (the assistant says these out loud)
Event data has no off-ball positions, so it cannot measure defensive-line height, compactness or
pressing shape. 360 data exists for some matches but is not used here (a good extension).

## Evaluate
```bash
python -m tacticallens.evaluate --match-id 8658 --dry-run     # questions + ground truth, no API
python -m tacticallens.evaluate --match-id 8658 --mode both   # tools vs no-tools baseline
```
Famous matches leak into LLM training data, so also evaluate an obscure match (pick any `match_id`
from the app) and pass `--label "TeamA vs TeamB, League Season"`. Report accuracy per category.

## Tests
```bash
python -m pytest -q      # unit tests on a hand-built mini match, agent loop with a fake LLM, real-match cross-check
```

## Ideas to extend
StatsBomb 360 freeze frames for compactness; PPDA/pressing from `Pressure` events; multi-match season
queries; retrieval over match reports; LLM-as-judge for "measured fact vs interpretation" scoring.
