# Usage dashboard: what the RAG pipelines cost, in the style of OpenRouter's activity page.
#
# Streamlit shows every file in pages/ as an extra page in the sidebar, so this needs no
# change to app.py. Run the app as usual (streamlit run app.py) and pick "Usage".
#
# Where the numbers come from:
#   RAG/cost_log.csv   one row per question (or indexed document), written by RAG/cost.py.
#                      Every pipeline in the app is tracked: the Ext ones by RAG/app_bridge.py,
#                      the RAGs/ ones by app.py (run_tracked) + the hook in initialize_api().
#                      The dollar amounts are what OpenRouter reported it charged.
#   OpenRouter API     the key's limit/spend and the account's credits (the "Account" row),
#                      to check the tracked total against what was actually billed.

import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from RAG.cost import COST_LOG

load_dotenv()
st.set_page_config(page_title="Usage", page_icon="💰", layout="wide")

# ---------- Pipelines and their colours ----------
# Each pipeline keeps the same colour everywhere on the page, whatever the filters show
# (colour follows the pipeline, not its rank). The 8 colours are a validated colour-blind-safe
# palette, with steps for light and dark mode. Indexing and summaries share the last slot.
PIPELINES = [  # (name in cost_log.csv, label on the page)
    ("naive", "Ext Naive RAG"),
    ("crag", "Ext Corrective RAG"),
    ("basic", "Basic RAG"),
    ("parent_child", "Parent-Child RAG"),
    ("hybrid", "Hybrid RAG"),
    ("crag_parent_child", "Corrective Parent-Child RAG"),
    ("crag_hybrid", "Corrective Hybrid RAG"),
]
OTHER = "Indexing & summaries"            # ingest, ingest_global, history_summary
LABELS = [label for _, label in PIPELINES] + [OTHER]
QUESTION_SYSTEMS = {name for name, _ in PIPELINES}
COLOURS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
PAGE_BG = {"light": "#ffffff", "dark": "#0e1117"}     # Streamlit's backgrounds: gaps between bars
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}   # value labels on the charts

mode = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
COLOUR_OF = dict(zip(LABELS, COLOURS[mode]))


# ---------- Data ----------
@st.cache_data
def load_log(path, modified):
    """cost_log.csv as a table (re-read whenever the file changes: `modified` is its mtime)."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    names = dict(PIPELINES)
    df["pipeline"] = df["system"].map(lambda s: names.get(s, OTHER))
    df["is_question"] = df["system"].isin(QUESTION_SYSTEMS)
    df["output_tokens"] = df["completion_tokens"] - df["reasoning_tokens"]   # answer text only
    df["tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    return df


@st.cache_data(ttl=300)
def account_usage():
    """This key's limit and spend, and the account's credits, from OpenRouter (5-minute cache).
    Returns None when there is no key or OpenRouter can't be reached."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}"}
    try:
        k = requests.get("https://openrouter.ai/api/v1/key", headers=headers, timeout=10).json()["data"]
        c = requests.get("https://openrouter.ai/api/v1/credits", headers=headers, timeout=10).json()["data"]
    except (requests.RequestException, ValueError, KeyError):
        return None
    limit = k.get("limit")
    spent = k.get("usage")
    if spent is None and limit is not None and k.get("limit_remaining") is not None:
        spent = limit - k["limit_remaining"]
    return {"key_spent": spent, "key_limit": limit,
            "credits_left": c.get("total_credits", 0) - c.get("total_usage", 0)}


def usd(x):
    """Small amounts need more decimals: $0.00012, $0.0234, $1.25."""
    if x == 0:
        return "$0"
    return f"${x:,.5f}" if x < 0.01 else f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def step_rows(df):
    """One row per (question, step) from the steps JSON column, with the model of that step:
    embedding steps use the embedding model, all other steps the chat model."""
    rows = []
    for r in df.itertuples():
        for step, s in json.loads(r.steps).items():
            model = r.embedding_model if step.startswith("embed") else r.chat_model
            rows.append({"pipeline": r.pipeline, "step": step, "model": model, "calls": s["calls"],
                         "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
                         "reasoning_tokens": s["reasoning_tokens"], "usd": s["usd"]})
    return pd.DataFrame(rows)


# ---------- Page ----------
st.title("💰 Usage")
st.caption("What every question cost, across all RAG pipelines: tokens and US dollars as "
           f"reported by OpenRouter. Source: `{COST_LOG.relative_to(COST_LOG.parent.parent).as_posix()}`")

if not COST_LOG.exists():
    st.info("No usage recorded yet. Ask a question on the main page (any pipeline), "
            "then come back here.")
    st.stop()

log = load_log(str(COST_LOG), COST_LOG.stat().st_mtime)

# Filters, above everything they control: period and model, then the pipelines as toggles
f1, f2 = st.columns([3, 2])
with f1:
    period = st.segmented_control("Period", ["24 hours", "7 days", "30 days", "All time"],
                                  default="30 days")
with f2:
    models = sorted(log["chat_model"].unique())
    model = st.selectbox("Chat model", ["All models"] + models)
present = [l for l in LABELS if l in set(log["pipeline"])]
chosen = st.pills("Pipelines (click to hide or show)", present, selection_mode="multi",
                  default=present)

df = log
days = {"24 hours": 1, "7 days": 7, "30 days": 30}.get(period or "All time")
if days:
    df = df[df["time"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
df = df[df["pipeline"].isin(chosen or present)]
if model != "All models":
    df = df[df["chat_model"] == model]

if df.empty:
    st.info("Nothing recorded for these filters.")
    st.stop()

# ---------- Headline numbers ----------
questions = df[df["is_question"]]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total spend", usd(df["total_usd"].sum()),
          help="Everything in the filtered rows: questions plus indexing and summaries.")
k2.metric("Requests", f"{int(df['n_calls'].sum()):,}",
          help="Model requests (chat and embedding). One question makes 2-8 of them.")
k3.metric("Tokens", f"{int(df['tokens'].sum()):,}",
          help=f"{int(df['prompt_tokens'].sum()):,} in / {int(df['output_tokens'].sum()):,} out / "
               f"{int(df['reasoning_tokens'].sum()):,} reasoning")
k4.metric("Questions", f"{len(questions):,}")
k5.metric("Avg cost / question", usd(questions["total_usd"].mean()) if len(questions) else "–")

# OpenRouter's own numbers: the tracked total above should be at most what the key spent
acct = account_usage()
if acct:
    limit = f" of {usd(acct['key_limit'])} limit" if acct["key_limit"] is not None else ""
    spent = usd(acct["key_spent"]) if acct["key_spent"] is not None else "–"
    st.caption(f"**Account (from OpenRouter):** this API key has spent {spent}{limit} · "
               f"credits left on the account {usd(acct['credits_left'])} · "
               f"tracked here (all time): {usd(log['total_usd'].sum())}")

st.divider()

# ---------- Charts over time ----------
# Hourly bars when the data covers two days or less, otherwise daily
hourly = df["time"].max() - df["time"].min() <= pd.Timedelta(days=2)
unit, unit_title = ("yearmonthdatehours", "Hour") if hourly else ("yearmonthdate", "Day")
shown = [l for l in LABELS if l in set(df["pipeline"])]
colour = alt.Color("pipeline:N", title="Pipeline", legend=None,      # shared legend below
                   scale=alt.Scale(domain=shown, range=[COLOUR_OF[l] for l in shown]),
                   sort=shown)
order = alt.Order("order:Q")
df = df.assign(order=df["pipeline"].map({l: i for i, l in enumerate(LABELS)}))

# One legend for both charts (same colours): a colour square + the name in normal text colour
st.markdown(" &nbsp; ".join(
    f'<span style="white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;'
    f'border-radius:2px;background:{COLOUR_OF[l]};margin-right:6px"></span>{l}</span>'
    for l in shown), unsafe_allow_html=True)


def over_time(field, title, fmt):
    """Stacked bars per hour/day, one colour per pipeline, with a tooltip on every segment."""
    return (alt.Chart(df).mark_bar(stroke=PAGE_BG[mode], strokeWidth=2)
            .encode(x=alt.X(f"{unit}(time):T", title=unit_title),
                    y=alt.Y(f"sum({field}):Q", title=title, axis=alt.Axis(format=fmt)),
                    color=colour, order=order,
                    tooltip=[alt.Tooltip(f"{unit}(time):T", title=unit_title),
                             alt.Tooltip("pipeline:N", title="Pipeline"),
                             alt.Tooltip(f"sum({field}):Q", title=title, format=fmt),
                             alt.Tooltip("count():Q", title="Rows")])
            .properties(height=280))


c1, c2 = st.columns(2)
with c1:
    st.subheader("Spend")
    st.altair_chart(over_time("total_usd", "USD", "$.5f"), width="stretch")
with c2:
    st.subheader("Token volume")
    st.altair_chart(over_time("tokens", "Tokens", ",.0f"), width="stretch")

# ---------- Comparison: average cost per question ----------
if len(questions):
    st.subheader("Average cost per question")
    per = (questions.groupby("pipeline")
           .agg(avg_usd=("total_usd", "mean"), avg_requests=("n_calls", "mean"),
                avg_tokens=("tokens", "mean"), questions=("total_usd", "size"))
           .reset_index())
    per["label"] = per["avg_usd"].map(usd)
    q_shown = [l for l in LABELS if l in set(per["pipeline"])]
    base = alt.Chart(per).encode(
        y=alt.Y("pipeline:N", title=None, sort=q_shown, axis=alt.Axis(labelLimit=260)),
        x=alt.X("avg_usd:Q", title="USD per question", axis=alt.Axis(format="$.5f", tickCount=5)))
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18).encode(
        color=alt.Color("pipeline:N", legend=None,
                        scale=alt.Scale(domain=q_shown, range=[COLOUR_OF[l] for l in q_shown])),
        tooltip=[alt.Tooltip("pipeline:N", title="Pipeline"),
                 alt.Tooltip("avg_usd:Q", title="Avg USD", format="$.6f"),
                 alt.Tooltip("avg_requests:Q", title="Avg requests", format=".1f"),
                 alt.Tooltip("avg_tokens:Q", title="Avg tokens", format=",.0f"),
                 alt.Tooltip("questions:Q", title="Questions")])
    # The value at the end of each bar, in the normal (secondary) text colour, not the bar's
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=TEXT_SECONDARY[mode]).encode(
        text="label:N")
    st.altair_chart((bars + labels).properties(height=44 * len(per) + 30), width="stretch")

# ---------- Tables ----------
tab1, tab2, tab3, tab4 = st.tabs(["By pipeline", "By step", "By model", "Recent questions"])

with tab1:
    t = (df.groupby("pipeline")
         .agg(rows=("total_usd", "size"), requests=("n_calls", "sum"),
              tokens_in=("prompt_tokens", "sum"), tokens_out=("output_tokens", "sum"),
              reasoning=("reasoning_tokens", "sum"), spend=("total_usd", "sum"))
         .reindex([l for l in LABELS if l in set(df["pipeline"])]).reset_index())
    t["avg_per_row"] = t["spend"] / t["rows"]
    t["share"] = 100 * t["spend"] / t["spend"].sum()
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "pipeline": "Pipeline", "rows": st.column_config.NumberColumn("Questions / documents"),
        "requests": st.column_config.NumberColumn("Requests", format="localized"), "tokens_in": st.column_config.NumberColumn("Tokens in", format="localized"),
        "tokens_out": st.column_config.NumberColumn("Tokens out", format="localized"), "reasoning": st.column_config.NumberColumn("Reasoning tokens", format="localized"), "spend": st.column_config.NumberColumn("Spend", format="$%.6f"),
        "avg_per_row": st.column_config.NumberColumn("Avg per question", format="$%.6f"),
        "share": st.column_config.ProgressColumn("Share of spend", format="%.1f%%", min_value=0, max_value=100)})

steps = step_rows(df)
with tab2:
    st.caption("Where the money goes inside each pipeline. The RAGs/ pipelines only "
               "distinguish chat and embedding requests.")
    s = (steps.groupby(["pipeline", "step"], sort=False)
         .agg(calls=("calls", "sum"), tokens_in=("prompt_tokens", "sum"),
              tokens_out=("completion_tokens", "sum"), reasoning=("reasoning_tokens", "sum"),
              spend=("usd", "sum")).reset_index())
    s["share_of_pipeline"] = 100 * s["spend"] / s.groupby("pipeline")["spend"].transform("sum")
    st.dataframe(s, hide_index=True, width="stretch", column_config={
        "pipeline": "Pipeline", "step": "Step", "calls": "Requests", "tokens_in": st.column_config.NumberColumn("Tokens in", format="localized"),
        "tokens_out": st.column_config.NumberColumn("Tokens out (incl. reasoning)", format="localized"), "reasoning": st.column_config.NumberColumn("Reasoning tokens", format="localized"),
        "spend": st.column_config.NumberColumn("Spend", format="$%.6f"),
        "share_of_pipeline": st.column_config.ProgressColumn("Share of the pipeline's spend",
                                                             format="%.1f%%", min_value=0, max_value=100)})

with tab3:
    m = (steps.groupby("model")
         .agg(requests=("calls", "sum"), tokens_in=("prompt_tokens", "sum"),
              tokens_out=("completion_tokens", "sum"), reasoning=("reasoning_tokens", "sum"),
              spend=("usd", "sum")).sort_values("spend", ascending=False).reset_index())
    m["share"] = 100 * m["spend"] / m["spend"].sum()
    st.dataframe(m, hide_index=True, width="stretch", column_config={
        "model": "Model", "requests": "Requests", "tokens_in": st.column_config.NumberColumn("Tokens in", format="localized"),
        "tokens_out": st.column_config.NumberColumn("Tokens out (incl. reasoning)", format="localized"), "reasoning": st.column_config.NumberColumn("Reasoning tokens", format="localized"),
        "spend": st.column_config.NumberColumn("Spend", format="$%.6f"),
        "share": st.column_config.ProgressColumn("Share of spend", format="%.1f%%", min_value=0, max_value=100)})

with tab4:
    recent = df.sort_values("time", ascending=False)[
        ["time", "pipeline", "question", "n_calls", "prompt_tokens", "completion_tokens",
         "reasoning_tokens", "total_usd"]].head(200)
    st.dataframe(recent, hide_index=True, width="stretch", column_config={
        "time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "pipeline": "Pipeline", "question": st.column_config.TextColumn("Question / document", width="large"),
        "n_calls": "Requests", "prompt_tokens": st.column_config.NumberColumn("Tokens in", format="localized"), "completion_tokens": st.column_config.NumberColumn("Tokens out", format="localized"),
        "reasoning_tokens": st.column_config.NumberColumn("Reasoning", format="localized"), "total_usd": st.column_config.NumberColumn("Cost", format="$%.6f")})

st.download_button("Download cost_log.csv", COST_LOG.read_bytes(), file_name="cost_log.csv",
                   mime="text/csv")
