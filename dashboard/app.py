from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from app.db import fetch_recent, fetch_stats, init_db  # noqa: E402

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
init_db()
stats = fetch_stats()

st.title("LLM Cost Autopilot")
st.caption("Shadow-cost routing across free Groq, Gemini, and local Ollama models.")

pct = stats["cost_reduction_pct"]
st.metric(
    "Cost reduction vs always-premium (Groq 70B shadow price)",
    f"{pct:.1f}%",
    help="Headline portfolio metric: routed shadow cost vs sending every request to the highest-tier model.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Routed shadow $", f"{stats['routed_cost']:.6f}")
c2.metric("Always-premium $", f"{stats['baseline_cost']:.6f}")
c3.metric("Saved vs premium", f"${stats['saved_vs_premium']:.6f}")
c4.metric("Saved vs GPT-4o list", f"{stats['cost_reduction_vs_gpt4o_pct']:.1f}%")

c5, c6, c7 = st.columns(3)
c5.metric("Requests", stats["total_requests"])
c6.metric("Escalation rate", f"{stats['escalation_rate'] * 100:.1f}%")
c7.metric("Avg quality", f"{stats['avg_quality']:.2f}")

dist = stats["routing_distribution"]
if dist:
    st.subheader("Routing distribution")
    df = pd.DataFrame(dist)
    st.bar_chart(df.set_index("model")["count"])
    st.dataframe(df, use_container_width=True)

daily = stats["daily"]
if daily:
    st.subheader("Daily cost")
    ddf = pd.DataFrame(daily)
    st.line_chart(ddf.set_index("day")[["routed_cost", "baseline_cost"]])

scores = stats["quality_scores"]
if scores:
    st.subheader("Quality score distribution")
    st.bar_chart(pd.Series(scores).value_counts(bins=8).sort_index())

st.subheader("Recent requests")
recent = fetch_recent(40)
if recent:
    slim = [
        {
            "id": r["id"],
            "tier": r["complexity_tier"],
            "model": r["routed_model"],
            "cost": r["cost"],
            "latency_ms": round(r["latency_ms"], 1),
            "quality": r["quality_score"],
            "escalated": bool(r["escalated"]),
            "status": r["verify_status"],
        }
        for r in recent
    ]
    st.dataframe(pd.DataFrame(slim), use_container_width=True)
else:
    st.info("No requests yet. POST to /v1/completions or run scripts/load_test.py")

report = ROOT / "artifacts" / "load_test_report.json"
if report.exists():
    st.subheader("Latest load-test report")
    data = json.loads(report.read_text(encoding="utf-8"))
    st.json({k: data[k] for k in data if k != "results"})
