# Agentic AI CEO — Human-like, Graph-Aware 10-Agents FMEA (Streamlit Free)
# ------------------------------------------------------------------------
# Single-file Streamlit app that simulates a human-like CEO replacement:
# - 10 leadership-style agents
# - FMEA tied to YOUR Problem + CEO Decision
# - Knowledge-graph themed risk extraction (no heavy NLP)
# - Dynamic, contextual mitigation plans + 30/60/90 roadmap
# - No external APIs; free Streamlit Cloud friendly
#
# Dependencies (add to requirements.txt):
#   streamlit
#   pandas
#   numpy
#   altair
#
# ------------------------------------------------------------------------

from __future__ import annotations
import json
import math
import random
import re
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# App Config / Theme
# -----------------------------
st.set_page_config(
    page_title="Agentic AI CEO — Graph-Aware FMEA",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Constants & Knowledge Base
# -----------------------------
LEADERSHIP_STYLES: Dict[str, Dict] = {
    "Autocratic Leader Agentic AI Agent CEO": {
        "desc": "Decides alone, tight control, speed over consensus.",
        "bias": {"S": +1, "O": +1, "D": -1},
        "tone": "decisive, directive, short sentences",
    },
    "Democratic Leader Agentic AI Agent CEO": {
        "desc": "Seeks participation and consensus, inclusive decision-making.",
        "bias": {"S": 0, "O": +1, "D": 0},
        "tone": "inclusive, invites feedback",
    },
    "Laissez-Faire Leader Agentic AI Agent CEO": {
        "desc": "Hands-off, relies on team autonomy and initiative.",
        "bias": {"S": +1, "O": +2, "D": -1},
        "tone": "hands-off, empowers teams",
    },
    "Transformational Leader Agentic AI Agent CEO": {
        "desc": "Drives inspiring vision, change, and innovation.",
        "bias": {"S": +2, "O": +1, "D": -1},
        "tone": "vision-led, change-focused",
    },
    "Transactional Leader Agentic AI Agent CEO": {
        "desc": "Targets performance via incentives, KPIs, and compliance.",
        "bias": {"S": 0, "O": 0, "D": +1},
        "tone": "metrics-driven, operational",
    },
    "Servant Leader Agentic AI Agent CEO": {
        "desc": "Puts people first, grows teams, builds trust and community.",
        "bias": {"S": 0, "O": 0, "D": 0},
        "tone": "empathetic, supportive",
    },
    "Charismatic Leader Agentic AI Agent CEO": {
        "desc": "Inspires via presence and storytelling; rallies followers.",
        "bias": {"S": +2, "O": +1, "D": -1},
        "tone": "narrative, rallying",
    },
    "Situational Leader Agentic AI Agent CEO": {
        "desc": "Adapts style to team maturity and task complexity.",
        "bias": {"S": -1, "O": -1, "D": +1},
        "tone": "adaptive, pragmatic",
    },
    "Visionary Leader Agentic AI Agent CEO": {
        "desc": "Long-term strategic focus; bold bets and roadmaps.",
        "bias": {"S": +2, "O": +1, "D": -1},
        "tone": "strategic, long-horizon",
    },
    "Bureaucratic Leader Agentic AI Agent CEO": {
        "desc": "Follows rules and procedures; values consistency.",
        "bias": {"S": -1, "O": 0, "D": +2},
        "tone": "policy-first, controlled",
    },
}

# Thematic knowledge graph: themes -> risk keywords -> suggested actions.
THEMES: Dict[str, Dict] = {
    "Market": {
        "keywords": ["market", "demand", "pricing", "competition", "competitor", "share", "gtm", "channel", "customer"],
        "actions": [
            ("VP Sales", "Run win/loss + pricing experiments", "Revenue growth, win-rate"),
            ("PMM", "Segment customers; refine positioning & ICP", "Qualified pipeline")
        ],
    },
    "Product": {
        "keywords": ["product", "roadmap", "feature", "quality", "launch", "prototype", "mvp"],
        "actions": [
            ("Head of Product", "Freeze scope; deliver MVP in 60d", "MVP shipped"),
            ("QA Lead", "Critical bug burn-down", "Defect escape rate")
        ],
    },
    "Ops/Supply": {
        "keywords": ["supply", "logistics", "inventory", "supplier", "manufacturing", "capacity", "lead time"],
        "actions": [
            ("COO", "Dual-source critical parts", "Fill-rate"),
            ("Ops Analyst", "Safety stock re-calibration", "Service level")
        ],
    },
    "Cyber/Data": {
        "keywords": ["cyber", "data", "breach", "ransomware", "security", "privacy", "pii", "attack"],
        "actions": [
            ("CISO", "Patch critical CVEs; isolate crown-jewel systems", "Mean time to patch"),
            ("IT", "Backups + recovery drills", "RPO/RTO met")
        ],
    },
    "Legal/Compliance": {
        "keywords": ["compliance", "regulation", "regulatory", "gdpr", "hipaa", "licence", "licensing", "audit"],
        "actions": [
            ("GC", "Gap analysis vs applicable regs", "Open findings closed"),
            ("Compliance", "Implement policy & training", "Completion rate")
        ],
    },
    "Finance": {
        "keywords": ["revenue", "margin", "cost", "capex", "opex", "cash", "burn", "budget", "pricing"],
        "actions": [
            ("CFO", "Zero-based budget on non-critical spend", "Runway months"),
            ("RevOps", "Pricing test (A/B)", "Gross margin")
        ],
    },
    "People": {
        "keywords": ["hiring", "layoff", "attrition", "talent", "union", "culture", "training"],
        "actions": [
            ("CHRO", "Critical roles hiring freeze + backfill only", "Time-to-fill"),
            ("L&D", "Upskill program for impacted teams", "Certification rate")
        ],
    },
    "Brand/Comms": {
        "keywords": ["brand", "pr", "reputation", "media", "press", "investor", "ir", "communications"],
        "actions": [
            ("Comms", "Narrative + Q&A; spokesperson brief", "Share of voice"),
            ("IR", "Investor update with milestones", "Call sentiment")
        ],
    },
    "AI/Ethics": {
        "keywords": ["ai", "model", "bias", "ethics", "safety", "open-source", "llm"],
        "actions": [
            ("AI Lead", "Model evals + bias tests", "Eval pass rate"),
            ("Policy", "AI use policy & audit log", "Policy adoption")
        ],
    },
    "International/Geo": {
        "keywords": ["tariff", "export", "sanction", "customs", "geo", "country", "india", "china", "europe", "eu", "us"],
        "actions": [
            ("Trade", "Tariff exposure + reroute plan", "Landed cost variance"),
            ("Finance", "FX hedging program", "FX P&L impact")
        ],
    },
}

RISKY_KEYWORDS = {
    # severity, occurrence, detection bonus (d lowers detection i.e. harder to detect)
    "merger": (2, 1, -1), "acquisition": (2, 1, -1), "layoff": (2, 2, -1),
    "restructure": (1, 1, -1), "pivot": (2, 1, -1), "ai": (1, 1, -1),
    "cloud": (1, 0, 0), "shutdown": (3, 2, -2), "outsourcing": (1, 1, 0),
    "offshoring": (1, 1, 0), "automation": (1, 1, 0), "cybersecurity": (2, 1, 1),
    "compliance": (1, 0, 2), "regulation": (1, 0, 2), "expansion": (1, 1, -1)
}

URGENCY_HINTS = {
    "immediately": 2, "urgent": 2, "asap": 2, "deadline": 1, "shutdown": 3, "breach": 3,
}

# -----------------------------
# Utils
# -----------------------------

def clamp(v:int, lo:int=1, hi:int=10) -> int:
    return max(lo, min(hi, int(round(v))))

@st.cache_data(show_spinner=False)
def extract_themes(problem: str, decision: str) -> Dict[str, float]:
    """Return theme scores by counting keyword hits in problem+decision text."""
    text = f"{problem} {decision}".lower()
    scores = defaultdict(int)
    for theme, cfg in THEMES.items():
        for kw in cfg["keywords"]:
            hits = text.count(kw.lower())
            if hits:
                scores[theme] += hits
    # normalize to 0..1 for weighting
    total = sum(scores.values()) or 1
    return {k: v/total for k, v in scores.items()}

@st.cache_data(show_spinner=False)
def risky_shifts(problem: str, decision: str) -> Tuple[int,int,int,int]:
    """Lightweight scoring deltas from risky keywords and urgency; returns (dS,dO,dD, urgency)."""
    text = f"{problem} {decision}".lower()
    dS=dO=dD=0
    urgency = 0
    for kw,(s,o,d) in RISKY_KEYWORDS.items():
        if kw in text:
            dS += s; dO += o; dD += d
    for u,w in URGENCY_HINTS.items():
        if u in text:
            urgency += w
    return dS, dO, dD, urgency

@st.cache_data(show_spinner=False)
def fmea_scores(problem: str, decision: str, leader_name: str) -> Tuple[int,int,int,int]:
    """Compute S,O,D and RPN for one leader, influenced by text + leader bias."""
    # base depending on length (complexity proxy)
    baseS, baseO, baseD = 5, 5, 5
    length_factor = min(len(problem + decision)//200, 3)
    baseS += length_factor; baseO += length_factor

    dS, dO, dD, urgency = risky_shifts(problem, decision)

    # decision polarity: if decision contains no/stop/cancel, some risks lower occurrence but raise severity
    dec_l = decision.lower()
    if any(x in dec_l for x in ["stop", "cancel", "pause", "hold"]):
        dS += 1; baseO -= 1
    if any(x in dec_l for x in ["expand", "launch", "invest", "go", "scale"]):
        dO += 1

    bias = LEADERSHIP_STYLES[leader_name]["bias"]
    S = clamp(baseS + dS + bias["S"]) 
    O = clamp(baseO + dO + bias["O"]) 
    D = clamp(baseD + dD + bias["D"]) 

    # urgency reduces detection (harder) and increases severity
    S = clamp(S + urgency//2)
    D = clamp(D - urgency//2)
    RPN = S*O*D
    return S,O,D,RPN

@st.cache_data(show_spinner=False)
def actions_for_themes(theme_scores: Dict[str, float]) -> List[Tuple[str,str,str,str,float]]:
    """Return actions expanded with 30/60/90 buckets based on theme weight."""
    actions = []
    for theme, weight in theme_scores.items():
        for owner, action, metric in THEMES[theme]["actions"]:
            # weight -> timeline bucket: 0-30 if >=0.25 else 30-60 if >=0.1 else 60-90
            if weight >= 0.25: bucket = "0-30d"
            elif weight >= 0.10: bucket = "30-60d"
            else: bucket = "60-90d"
            actions.append((theme, owner, action, metric, weight))
    # sort by weight desc
    actions.sort(key=lambda x: x[-1], reverse=True)
    return actions

@st.cache_data(show_spinner=False)
def generate_mitigations(problem: str, decision: str, leader: str, theme_scores: Dict[str,float], S:int,O:int,D:int) -> List[Dict]:
    """Create mitigation items that explicitly reference the problem & decision and top themes."""
    # choose top 3 themes for this leader
    top = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    items = []
    for theme, w in top:
        base = THEMES[theme]["actions"][0]
        owner, action, metric = base
        timeline = "0-30d" if S*O >= 56 or w>=0.25 else ("30-60d" if S*O>=45 or w>=0.12 else "60-90d")
        text = (
            f"{owner}: {action}. Tie directly to this scenario — Problem: '{problem[:120]}', Decision: '{decision[:120]}'. "
            f"Track {metric}."
        )
        items.append({
            "Leader": leader,
            "Theme": theme,
            "Owner": owner,
            "Mitigation": text,
            "Timeline": timeline,
            "Metric": metric,
            "Weight": round(w,3)
        })
    return items

# -----------------------------
# UI — Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    delay = st.slider("Thinking delay per agent (s)", 0.0, 0.2, 0.05, 0.01)
    show_eli5 = st.checkbox("Show ELI5", value=True)
    st.markdown("""
    **Notes**
    - No external APIs; works on Streamlit Free.
    - Mitigations are tied to your text via the theme knowledge-graph and risk heuristics.
    """)

# -----------------------------
# Header & Inputs
# -----------------------------
st.title("Agentic AI CEO — 10 Leadership Agents, Graph-Aware FMEA")
st.caption("Type a Problem and the CEO Decision. Ten agents score FMEA and propose contextual mitigations + a 30/60/90 roadmap.")

cols = st.columns([2,1,1])
with cols[0]:
    st.subheader("Scenario Input")
with cols[1]:
    st.write("")
with cols[2]:
    st.write("")

problem = st.text_area("Problem", height=100, placeholder="Describe the business problem …")
decision = st.text_area("Decision taken by CEO", height=100, placeholder="Describe the exact decision …")

run = st.button("Run 10 Leadership Agents FMEA")

# -----------------------------
# Main Logic
# -----------------------------
if run:
    if not problem.strip() or not decision.strip():
        st.warning("Please provide both Problem and Decision.")
        st.stop()

    st.success("Running agents …")

    # 1) Extract themes & candidate actions
    theme_scores = extract_themes(problem, decision)
    if not theme_scores:
        # ensure at least some actions exist by seeding with neutral theme
        theme_scores = {"Market": 0.34, "Product": 0.33, "Finance": 0.33}
    actions_catalog = actions_for_themes(theme_scores)

    # 2) Per-agent FMEA
    per_agent_rows = []
    all_mitigations: List[Dict] = []

    for leader, cfg in LEADERSHIP_STYLES.items():
        block = st.expander(leader, expanded=False)
        with block:
            time.sleep(delay)
            S,O,D,RPN = fmea_scores(problem, decision, leader)
            per_agent_rows.append({
                "Leader": leader,
                "Severity": S,
                "Occurrence": O,
                "Detection": D,
                "RPN": RPN,
            })

            st.caption(cfg["desc"])            
            cols2 = st.columns(4)
            cols2[0].metric("Severity", S)
            cols2[1].metric("Occurrence", O)
            cols2[2].metric("Detection", D)
            cols2[3].metric("RPN", RPN)

            # Failure mode narrative tied to inputs
            fail_text = (
                f"Execution gaps and unintended consequences when applying the decision '{decision[:140]}' to the problem '{problem[:140]}', "
                f"viewed through the lens of {leader}."
            )
            st.markdown(f"**Failure Mode:** {fail_text}")
            st.markdown("**Effects:** Delays, cost overruns, quality issues, compliance risks, or missed market opportunities.")
            
            # Mitigations (contextual)
            items = generate_mitigations(problem, decision, leader, theme_scores, S,O,D)
            all_mitigations.extend(items)
            mit_df = pd.DataFrame(items)
            st.markdown("**Mitigation Strategy (contextual):**")
            st.dataframe(mit_df[["Theme","Owner","Mitigation","Timeline","Metric"]], use_container_width=True)

            if show_eli5:
                st.info(
                    f"ELI5: Severity is how big the ouch is, Occurrence how often it happens, Detection how quickly we spot it. RPN=S×O×D. "
                    f"For {leader.split()[0]}, style bias nudged these scores to S={S}, O={O}, D={D}."
                )

    st.success("All agents finished.")

    # 3) Summary
    st.subheader("Summary of Results")
    tab1, tab2, tab3 = st.tabs(["Scores", "Top Risks", "Combined Roadmap"])

    # Scores table
    with tab1:
        score_df = pd.DataFrame(per_agent_rows).sort_values("RPN", ascending=False)
        st.dataframe(score_df, use_container_width=True)
        chart = (
            alt.Chart(score_df)
            .mark_bar()
            .encode(x=alt.X("Leader:N", sort="-y"), y="RPN:Q", tooltip=["Leader","RPN","Severity","Occurrence","Detection"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # Top risks text
    with tab2:
        st.markdown("**Theme weights (from your input):**")
        weights_df = pd.DataFrame(
            sorted(theme_scores.items(), key=lambda x: x[1], reverse=True),
            columns=["Theme","Weight"],
        )
        st.dataframe(weights_df, use_container_width=True)
        st.markdown(
            "These weights are inferred from your Problem + Decision text using a knowledge-graph of business themes."
        )

    # Combined roadmap
    with tab3:
        # Risk-weighted aggregation: use RPN percentile as leader weight
        score_df = pd.DataFrame(per_agent_rows)
        max_rpn = max(score_df["RPN"]) or 1
        leader_weight = {r["Leader"]: (r["RPN"]/max_rpn) for r in per_agent_rows}

        # Combine + de-duplicate by (Theme, Owner, Metric)
        combo = {}
        for m in all_mitigations:
            key = (m["Theme"], m["Owner"], m["Metric"])
            w = leader_weight.get(m["Leader"], 0.5) * (1.0 + m["Weight"])  # agent weight * theme weight
            if key not in combo:
                combo[key] = {**m, "Score": w}
            else:
                combo[key]["Score"] += w
                # earliest timeline wins
                order = {"0-30d":0, "30-60d":1, "60-90d":2}
                if order[m["Timeline"]] < order[combo[key]["Timeline"]]:
                    combo[key]["Timeline"] = m["Timeline"]

        roadmap_df = pd.DataFrame(combo.values())
        if not roadmap_df.empty:
            roadmap_df = roadmap_df.sort_values(["Timeline","Score"], ascending=[True, False])
            st.dataframe(roadmap_df[["Theme","Owner","Mitigation","Timeline","Metric","Score"]], use_container_width=True)

            # downloads
            st.download_button(
                label="Download FMEA Scores (CSV)",
                data=pd.DataFrame(per_agent_rows).to_csv(index=False).encode(),
                file_name="fmea_scores.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download Roadmap (CSV)",
                data=roadmap_df.to_csv(index=False).encode(),
                file_name="mitigation_roadmap.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download Full Results (JSON)",
                data=json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "problem": problem,
                    "decision": decision,
                    "theme_scores": theme_scores,
                    "per_agent": per_agent_rows,
                    "mitigations": all_mitigations,
                    "roadmap": roadmap_df.to_dict(orient="records"),
                }, indent=2).encode(),
                file_name="agentic_ceo_results.json",
                mime="application/json",
            )
        else:
            st.info("No roadmap items generated. Try adding more detail to the Problem/Decision.")

    # 4) Knowledge Graph view (simple)
    st.subheader("Knowledge Graph (Themes ↔ Keywords)")
    edge_rows = []
    for theme, cfg in THEMES.items():
        for kw in cfg["keywords"]:
            edge_rows.append({"Theme": theme, "Keyword": kw})
    kg_df = pd.DataFrame(edge_rows)
    st.dataframe(kg_df, use_container_width=True)

else:
    st.info("Enter a Problem and a Decision, then click 'Run 10 Leadership Agents FMEA'.")
