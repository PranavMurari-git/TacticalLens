"""TacticalLens Streamlit app. Run:  streamlit run app.py"""
import os

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

from tacticallens import charts  # noqa: E402
from tacticallens.agent import DEFAULT_MODEL, TacticalAgent  # noqa: E402
from tacticallens.analytics import ZONE_ACTIONS, MatchAnalytics  # noqa: E402
from tacticallens.data import list_competitions, list_matches, load_events  # noqa: E402

st.set_page_config(page_title="TacticalLens", page_icon="⚽", layout="wide")

EXAMPLES = [
    "Compare the two teams' chance creation in this match.",
    "Which players moved the ball forward best?",
    "Compare each team's passing in the first and second half.",
    "Which areas of the pitch did each team use most?",
    "Which players were most involved in attacking sequences?",
    "How compact was the defence?",
]


@st.cache_data(show_spinner=False)
def _competitions():
    return list_competitions()


@st.cache_data(show_spinner=False)
def _matches(comp_id, season_id):
    return list_matches(comp_id, season_id)


@st.cache_data(show_spinner="Loading match events...")
def _events(match_id):
    return load_events(match_id)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("⚽ TacticalLens")
    st.caption("Evidence-grounded tactics analysis on StatsBomb Open Data.")
    comps = _competitions()
    default_comp = list(comps["competition_name"].unique()).index("FIFA World Cup")
    comp_name = st.selectbox("Competition", comps["competition_name"].unique(), index=default_comp)
    seasons = comps[comps["competition_name"] == comp_name]
    season_name = st.selectbox("Season", seasons["season_name"])
    row = seasons[seasons["season_name"] == season_name].iloc[0]
    matches = _matches(int(row.competition_id), int(row.season_id))
    labels = [f"{m.home} {m.home_score}-{m.away_score} {m.away} ({m.date})" for m in matches.itertuples()]
    pick = st.selectbox("Match", range(len(labels)), format_func=lambda i: labels[i],
                        index=len(labels) - 1)
    m = matches.iloc[pick]
    api_key = st.text_input("Anthropic API key", type="password",
                            value=os.environ.get("ANTHROPIC_API_KEY", ""))
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("Data: StatsBomb Open Data (attribution required).")

events = _events(int(m.match_id))
label = f"{m.home} vs {m.away}, {m.competition} {m.season} {m.stage}, {m.date}"
analytics = MatchAnalytics(events, label)

if st.session_state.get("match_id") != int(m.match_id):  # new match: reset the chat
    st.session_state.update(match_id=int(m.match_id), messages=[])

st.header(f"{m.home} {m.home_score}-{m.away_score} {m.away}")
st.caption(f"{m.competition} {m.season} · {m.stage} · {m.date}")
tab_chat, tab_dash = st.tabs(["💬 Ask TacticalLens", "📊 Dashboard"])


def show_charts(chart_list, key_prefix):
    for i, c in enumerate(chart_list):
        st.pyplot(charts.render(analytics, c["chart"], c.get("team")), clear_figure=True)


# ------------------------------------------------------------------ chat tab
with tab_chat:
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(EXAMPLES):
        if cols[i % 3].button(q, key=f"ex{i}", width="stretch"):
            clicked = q
    for k, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                show_charts(msg.get("charts", []), f"h{k}")
                if msg.get("evidence"):
                    with st.expander(f"Supporting events ({len(msg['evidence'])})"):
                        st.dataframe(pd.DataFrame(msg["evidence"]).drop(columns=["event_id"]), hide_index=True)
                if msg.get("tool_calls"):
                    with st.expander(f"Tools called ({len(msg['tool_calls'])})"):
                        for c in msg["tool_calls"]:
                            st.code(f"{c['name']}({c['input']})", language="python")
    question = st.chat_input("Ask about this match...") or clicked
    if question:
        if not api_key:
            st.warning("Add an Anthropic API key in the sidebar to chat. The Dashboard tab works without one.")
        else:
            history = [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages]
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Running analytics..."):
                    try:
                        res = TacticalAgent(analytics, model=model, api_key=api_key).ask(question, history)
                    except Exception as exc:  # network / auth / rate limit
                        st.error(f"LLM call failed: {exc}")
                        st.stop()
                st.markdown(res.answer)
                show_charts(res.charts, "new")
                if res.evidence:
                    with st.expander(f"Supporting events ({len(res.evidence)})"):
                        st.dataframe(pd.DataFrame(res.evidence).drop(columns=["event_id"]), hide_index=True)
                if res.tool_calls:
                    with st.expander(f"Tools called ({len(res.tool_calls)})"):
                        for c in res.tool_calls:
                            st.code(f"{c['name']}({c['input']})", language="python")
            st.session_state.messages.append({"role": "assistant", "content": res.answer, "charts": res.charts,
                                              "evidence": res.evidence, "tool_calls": res.tool_calls})

# ------------------------------------------------------------------ dashboard tab
with tab_dash:
    ov = analytics.overview()
    st.dataframe(pd.DataFrame(ov["teams"]).rename(index=lambda s: s.replace("_", " ")), width="stretch")
    st.caption(ov["notes"])
    team = st.radio("Team", analytics.teams, horizontal=True)
    c1, c2 = st.columns(2)
    c1.pyplot(charts.shot_map(analytics, team), clear_figure=True)
    c2.pyplot(charts.pass_network(analytics, team), clear_figure=True)
    c3, c4 = st.columns(2)
    action = c3.selectbox("Zone heatmap action", list(ZONE_ACTIONS))
    c3.pyplot(charts.zone_heatmap(analytics, team, action), clear_figure=True)
    c4.pyplot(charts.xg_timeline(analytics), clear_figure=True)
    with st.expander("Goals"):
        st.dataframe(pd.DataFrame(analytics.goals()["goals"]).drop(columns=["event_id"]), hide_index=True)
    with st.expander("Progressive passers"):
        st.dataframe(pd.DataFrame(analytics.progressive_passes(team, 8)["top_players"]), hide_index=True)
