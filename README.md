# TacticalLens: AI Football Tactical Analyst

An LLM-powered football analytics assistant that answers natural-language questions about matches using Python analytics, match-event data, and interactive visualisations.

## Project Overview

TacticalLens combines Large Language Models (LLMs) with deterministic analytics tools to transform raw football match data into interpretable tactical insights. Instead of manually analysing statistics, users can ask questions about chance creation, passing patterns, ball progression, and attacking performance.

## Key Features

- **Conversational Analysis:** Ask tactical questions in natural language using an LLM-powered agent.
- **Expected Goals (xG):** Analyse shot quality and compare scoring opportunities.
- **Passing Analytics:** Explore passing networks and progressive passes.
- **Pitch-Zone Analysis:** Investigate where teams perform their actions across the pitch.
- **Match Flow:** Visualise xG timelines and attacking sequences.
- **Evidence-Grounded Responses:** Use Python tools to calculate statistics and provide supporting match-event evidence.
- **Evaluation Framework:** Compare tool-assisted answers against a no-tools baseline.

## Architecture

The application follows a tool-augmented LLM workflow:

`User Question → LLM Agent → Python Analytics Tools → Match Data → Evidence → LLM Response`

The LLM interprets questions and selects analytical tools, while Python performs the calculations to reduce reliance on generated numerical claims.

## Technology Stack

- **Language:** Python
- **LLM:** Anthropic API
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, mplsoccer
- **Interface:** Streamlit
- **Data Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- **Testing:** Pytest

## Run Locally

```bash
git clone https://github.com/PranavMurari-git/TacticalLens.git
cd TacticalLens

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The dashboard works without an LLM API key. Conversational analysis requires a valid Anthropic API key configured through the environment.

## Limitations

- Analyses only matches available in the selected dataset.
- Event data does not provide continuous player-tracking information.
- Tactical conclusions require interpretation; statistics alone do not establish causation.
- LLM-generated explanations may still contain interpretive errors.

## Future Improvements

- Broader competition and season coverage.
- Player-tracking and off-ball movement analysis.
- Advanced pressing and possession-value metrics.
- Multi-match tactical comparisons.
- More rigorous evaluation and deployment.

---

**TacticalLens — turning football match data into evidence-backed tactical insights.**
