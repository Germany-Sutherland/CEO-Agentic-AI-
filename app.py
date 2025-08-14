# Agentic AI CEO — 10 Leadership Agents with LLM Mitigations + Live Knowledge Graph (Streamlit Free)
# ---------------------------------------------------------------------------------
# What this app does
# - Takes your Problem + CEO Decision
# - Extracts themes (knowledge-graph style) and entities
# - Pulls light, free context from Wikipedia + ~5 RSS business/news items (no API keys)
# - Runs 10 leadership-style agents that each perform a contextual FMEA (S/O/D/RPN)
# - **Generates 10–20 line, agent-specific Mitigation Strategies** using a tiny local LLM
#   (optional; gracefully falls back to a fast rule-based writer if model is unavailable)
# - Builds a combined, risk-weighted 30/60/90 roadmap
# - Renders a small knowledge graph (Altair+NetworkX) that connects inputs→themes→entities→news
#
# Streamlit Free friendly: uses only free/open libs. The LLM is optional and will be used
# only if `transformers` is available at runtime.
#
# Minimal requirements.txt (safe for Free tier):
#   streamlit
#   pandas
#   numpy
#   altair
#   wikipedia
#   feedparser
#   requests
#   networkx
#
# Optional (ONLY if your free environment can install them):
#   transformers
#   torch
#
# ---------------------------------------------------------------------------------

from __future__ import annotations
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# light, free data sources
import wikipedia
import feedparser
import requests
import networkx as nx

# Try to enable a tiny local LLM if available (optional)
LLM_AVAILABLE = False
LLM = None
try:
    from transformers import pipeline
    # Use a very small model to keep memory low; distilgpt2 is ~82MB
    LLM = pipeline("text-generation", model="distilgpt2", device=-1)
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(
    page_title="Agentic AI CEO — LLM FMEA + Knowledge Graph",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------
# Knowledge Graph Themes
# ---------------------------------
THEMES: Dict[str, Dict] = {
    "Market": {
        "keywords": ["market", "demand", "pricing", "competition", "competitor", "share", "gtm", "channel", "customer"],
        "actions": [
            ("VP Sales", "Win/loss + pricing experiments", "Win-rate, gross margin"),
            ("PMM", "Segment ICP + reposition tiers", "Qualified pipeline")
        ],
    },
    "Product": {
        "keywords": ["product", "roadmap", "feature", "quality", "launch", "prototype", "mvp", "bug"],
        "actions": [
            ("Head of Product", "Scope freeze; ship MVP in 60d", "MVP shipped"),
            ("QA Lead", "Critical bug burn-down", "Defect escape rate")
        ],
    },
    "Ops/Supply": {
        "keywords": ["supply", "logistics", "inventory", "supplier", "manufacturing", "capacity", "lead time"],
        "actions": [
            ("COO", "Dual-source critical parts", "Fill-rate"),
            ("Ops Analyst", "Recalibrate safety stock", "Service level")
        ],
    },
    "Cyber/Data": {
        "keywords": ["cyber", "data", "breach", "ransomware", "security", "privacy", "pii", "attack"],
        "actions": [
            ("CISO", "Patch critical CVEs; isolate crown-jewel systems", "Mean-time-to-patch"),
            ("IT", "Backups + recovery drills", "RPO/RTO satisfied")
        ],
    },
    "Legal/Compliance": {
        "keywords": ["compliance", "regulation", "regulatory", "gdpr", "hipaa", "license", "licensing", "audit"],
        "actions": [
            ("GC", "Gap analysis vs applicable regs", "Open findings closed"),
            ("Compliance", "Policy + training rollout", "Completion rate")
        ],
    },
    "Finance": {
        "keywords": ["revenue", "margin", "cost", "capex", "opex", "cash", "burn", "budget", "pricing"],
        "actions": [
            ("CFO", "Zero-based budget on non-critical spend", "Runway months"),
            ("RevOps", "A/B pricing tests", "Gross margin")
        ],
    },
    "People": {
        "keywords": ["hiring", "layoff", "attrition", "talent", "union", "culture", "training"],
        "actions": [
            ("CHRO", "Critical roles freeze + backfill only", "Time-to-fill"),
            ("L&D", "Upskill program for impacted teams", "Certification rate")
        ],
    },
    "Brand/Comms": {
        "keywords": ["brand", "pr", "reputation", "media", "press", "investor", "ir", "communications"],
        "actions": [
            ("Comms", "Narrative + spokesperson brief", "Share of voice"),
            ("IR", "Investor update with milestones", "Call sentiment")
        ],
    },
    "AI/Ethics": {
        "keywords": ["ai", "model", "bias", "ethics", "safety", "open-source", "llm"],
        "actions": [
            ("AI Lead", "Model evals + bias tests", "Eval pass rate"),
            ("Policy", "AI-use policy + audit log", "Policy adoption")
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

LEADERSHIP: Dict[str, Dict] = {
    "Autocratic Leader Agentic AI Agent CEO": {"desc": "Decisive, tight control.", "bias": {"S":+1,"O":+1,"D":-1}, "voice":"brief, action-first"},
    "Democratic Leader Agentic AI Agent CEO": {"desc": "Inclusive, consensus-seeking.", "bias": {"S":0,"O":+1,"D":0}, "voice":"collaborative"},
    "Laissez-Faire Leader Agentic AI Agent CEO": {"desc": "Hands-off, autonomy.", "bias": {"S":+1,"O":+2,"D":-1}, "voice":"empowering"},
    "Transformational Leader Agentic AI Agent CEO": {"desc": "Vision + change.", "bias": {"S":+2,"O":+1,"D":-1}, "voice":"inspiring"},
    "Transactional Leader Agentic AI Agent CEO": {"desc": "KPIs and controls.", "bias": {"S":0,"O":0,"D":+1}, "voice":"metrics-driven"},
    "Servant Leader Agentic AI Agent CEO": {"desc": "People-first.", "bias": {"S":0,"O":0,"D":0}, "voice":"empathetic"},
    "Charismatic Leader Agentic AI Agent CEO": {"desc": "Storytelling.", "bias": {"S":+2,"O":+1,"D":-1}, "voice":"rallying"},
    "Situational Leader Agentic AI Agent CEO": {"desc": "Adapts to context.", "bias": {"S":-1,"O":-1,"D":+1}, "voice":"pragmatic"},
    "Visionary Leader Agentic AI Agent CEO": {"desc": "Long horizon.", "bias": {"S":+2,"O":+1,"D":-1}, "voice":"strategic"},
    "Bureaucratic Leader Agentic AI Agent CEO": {"desc": "Policy-first.", "bias": {"S":-1,"O":0,"D":+2}, "voice":"controlled"},
}

RISKY_KEYWORDS = {
    "merger": (2,1,-1), "acquisition": (2,1,-1), "layoff": (2,2,-1),
    "restructure": (1,1,-1), "pivot": (2,1,-1), "ai": (1,1,-1),
    "cloud": (1,0,0), "shutdown": (3,2,-2), "outsourcing": (1,1,0),
    "automation": (1,1,0), "cybersecurity": (2,1,1), "compliance": (1,0,2),
    "regulation": (1,0,2), "expansion": (1,1,-1)
}
URGENCY_HINTS = {"immediately":2, "urgent":2, "asap":2, "deadline":1, "shutdown":3, "breach":3}

# ---------------------------------
# Helpers
# ---------------------------------

def clamp(v:int, lo:int=1, hi:int=10) -> int:
    return max(lo, min(hi, int(round(v))))

@st.cache_data(show_spinner=False)
def extract_themes(problem: str, decision: str) -> Dict[str, float]:
    text = f"{problem} {decision}".lower()
    scores = defaultdict(int)
    for theme, cfg in THEMES.items():
        for kw in cfg["keywords"]:
            hits = text.count(kw)
            if hits:
                scores[theme] += hits
    total = sum(scores.values()) or 1
    return {k: v/total for k, v in scores.items()}

@st.cache_data(show_spinner=False)
def risky_shifts(problem: str, decision: str) -> Tuple[int,int,int,int]:
    text = f"{problem} {decision}".lower()
    dS=dO=dD=0; urgency=0
    for kw,(s,o,d) in RISKY_KEYWORDS.items():
        if kw in text:
            dS += s; dO += o; dD += d
    for u,w in URGENCY_HINTS.items():
        if u in text: urgency += w
    return dS,dO,dD,urgency

@st.cache_data(show_spinner=False)
def fmea_scores(problem: str, decision: str, leader: str):
    baseS, baseO, baseD = 5,5,5
    length_factor = min(len(problem + decision)//200, 3)
    baseS += length_factor; baseO += length_factor
    dS,dO,dD,urgency = risky_shifts(problem, decision)
    if any(x in decision.lower() for x in ["stop","cancel","pause","hold"]):
        dS += 1; baseO -= 1
    if any(x in decision.lower() for x in ["expand","launch","invest","go","scale"]):
        dO += 1
    bias = LEADERSHIP[leader]["bias"]
    S = clamp(baseS + dS + bias["S"]) 
    O = clamp(baseO + dO + bias["O"]) 
    D = clamp(baseD + dD + bias["D"]) 
    S = clamp(S + (urgency//2)); D = clamp(D - (urgency//2))
    RPN = S*O*D
    return S,O,D,RPN

# -------- Wikipedia & News (no API keys) --------

@st.cache_data(show_spinner=False)
def wiki_context(problem: str, decision: str, limit:int=3) -> List[Dict]:
    text = f"{problem} {decision}"[:300]
    try:
        q = wikipedia.search(text)[:limit]
    except Exception:
        q = []
    results = []
    for term in q:
        try:
            s = wikipedia.summary(term, sentences=2)
            results.append({"title": term, "summary": s})
        except Exception:
            continue
    return results

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
]

@st.cache_data(show_spinner=False)
def news_context(limit:int=5) -> List[Dict]:
    items = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:2]:
                items.append({"title": e.get("title",""), "link": e.get("link",""), "summary": e.get("summary","")[:280]})
        except Exception:
            continue
    # de-duplicate by title
    seen=set(); uniq=[]
    for it in items:
        t = it["title"].strip()
        if t and t not in seen:
            seen.add(t); uniq.append(it)
        if len(uniq)>=limit: break
    return uniq

# -------- Knowledge Graph build & plot --------

def build_graph(problem: str, decision: str, themes: Dict[str,float], wiki: List[Dict], news: List[Dict]):
    G = nx.Graph()
    G.add_node("Problem", type="input"); G.add_node("Decision", type="input")
    for t,w in themes.items():
        G.add_node(t, type="theme", weight=float(w))
        G.add_edge("Problem", t); G.add_edge("Decision", t)
    for w in wiki:
        node = f"WIKI: {w['title'][:18]}"; G.add_node(node, type="wiki")
        # connect to strongest theme
        if themes:
            top = max(themes.items(), key=lambda x: x[1])[0]
            G.add_edge(node, top)
    for i,n in enumerate(news[:5]):
        node = f"NEWS{i+1}: {n['title'][:22]}"; G.add_node(node, type="news")
        if themes:
            top = max(themes.items(), key=lambda x: x[1])[0]
            G.add_edge(node, top)
    return G

def graph_dataframe(G:nx.Graph):
    pos = nx.spring_layout(G, seed=42, k=0.9)
    nodes = []
    for n,d in G.nodes(data=True):
        x,y = pos[n]
        nodes.append({"id": n, "x": x, "y": y, "type": d.get("type","other")})
    edges = []
    for s,t in G.edges():
        sx,sy = pos[s]; tx,ty = pos[t]
        edges.append({"source": s, "target": t, "sx": sx, "sy": sy, "tx": tx, "ty": ty})
    return pd.DataFrame(nodes), pd.DataFrame(edges)

def draw_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    edge_chart = alt.Chart(edges_df).mark_line().encode(
        x="sx:Q", y="sy:Q", x2="tx:Q", y2="ty:Q"
    )
    node_chart = alt.Chart(nodes_df).mark_circle(size=200).encode(
        x="x:Q", y="y:Q", color="type:N", tooltip=["id","type"]
    )
    text_chart = alt.Chart(nodes_df).mark_text(dy=-12, fontSize=11).encode(
        x="x:Q", y="y:Q", text="id:N"
    )
    st.altair_chart((edge_chart + node_chart + text_chart).properties(height=420), use_container_width=True)

# -------- LLM & Mitigation Writer --------

def _llm_generate(prompt:str, max_new_tokens:int=180) -> str:
    if not LLM_AVAILABLE:
        return ""
    try:
        out = LLM(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9, top_p=0.95)[0]["generated_text"]
        return out[len(prompt):].strip()
    except Exception:
        return ""

def mitigation_llm(problem:str, decision:str, themes:Dict[str,float], wiki:List[Dict], news:List[Dict], leader:str) -> List[str]:
    top_themes = ", ".join([f"{k}({v:.2f})" for k,v in sorted(themes.items(), key=lambda x:x[1], reverse=True)[:3]])
    ctx_wiki = "; ".join([w["title"] for w in wiki[:2]])
    ctx_news = "; ".join([n["title"] for n in news[:3]])
    style = LEADERSHIP[leader]["voice"]
    prompt = (
        f"You are a Fortune-500-grade CEO operating in the style: {leader} ({style}).\n"
        f"Problem: {problem}\nDecision: {decision}\n"
        f"Top themes: {top_themes}.\n"
        f"Relevant context — Wikipedia: {ctx_wiki}. News: {ctx_news}.\n"
        f"Write a 10-20 line Mitigation Strategy with numbered actions, owners, KPIs, and 30/60/90 timelines.\n"
        f"Each line should be crisp and directly tied to the problem & decision.\n"
        f"Start now:\n1. "
    )
    text = _llm_generate(prompt)
    if not text:
        return []
    # split into lines, keep 10-20
    lines = [l.strip(" -\t") for l in re.split(r"\n|\r", text) if l.strip()]
    # keep only 10-20 numbered or bullet-like lines
    cleaned = []
    for l in lines:
        if len(cleaned)>=22: break
        if re.match(r"^\d+\.|^-", l):
            cleaned.append(l)
    if len(cleaned) < 10:
        # fallback: take first 15 non-empty sentences
        sents = re.split(r"(?<=[.!?])\s+", text)
        cleaned = [f"- {s}" for s in sents[:15] if s.strip()]
    return cleaned[:20]

def mitigation_rule_based(problem:str, decision:str, themes:Dict[str,float], leader:str) -> List[str]:
    # Use top themes and style to craft specific lines
    top = [k for k,_ in sorted(themes.items(), key=lambda x:x[1], reverse=True)[:3]] or ["Market","Product","Finance"]
    voice = LEADERSHIP[leader]["voice"]
    base = [
        f"30d | {top[0]} | Stand up tiger team to address: '{problem[:90]}'. Owner: VP of {top[0].split('/')[0]}. KPI: leading indicator moves +10%.",
        f"30d | {top[1]} | Translate decision '{decision[:90]}' into 30/60/90 deliverables. Owner: COO. KPI: milestones delivered.",
        f"30d | Risk | Establish risk register for {', '.join(top)}; weekly review. Owner: PMO. KPI: red risks → amber.",
        f"60d | {top[0]} | Run 3 controlled experiments aligned to decision; sunset losers. Owner: Analytics. KPI: lift vs control.",
        f"60d | {top[1]} | Close top 5 blockers from execution retros. Owner: Eng Leads. KPI: cycle time -15%.",
        f"60d | {top[2] if len(top)>2 else 'Finance'} | Re-forecast with decision impact. Owner: FP&A. KPI: variance < 5%.",
        f"90d | Brand/Comms | Publish external update on outcomes. Owner: Comms. KPI: sentiment ↑, investor Qs ↓.",
        f"90d | People | Targeted upskilling for impacted teams. Owner: L&D. KPI: certification rate > 80%.",
        f"90d | Governance | Post-mortem on decision efficacy; reset targets. Owner: Exec Staff. KPI: RPN ↓ 30%.",
    ]
    # Style seasoning
    if "Autocratic" in leader: base.insert(0, "0d | Command center with single-threaded owner. Daily standups. KPI: decisions <24h.")
    if "Democratic" in leader: base.append("Ongoing | Stakeholder forum for feedback; publish decisions log. KPI: adoption > 85%.")
    if "Transformational" in leader: base.append("Ongoing | Change story and vision cascade to teams. KPI: change-readiness index.")
    if "Transactional" in leader: base.append("Monthly | Align incentives to roadmap outcomes. KPI: OKR attainment.")
    if "Bureaucratic" in leader: base.append("Ongoing | Policy fast-track for experiments. KPI: lead time < 7d.")
    if "Situational" in leader: base.append("Quarterly | Reassess team maturity & adapt coaching. KPI: competency scores.")
    if "Charismatic" in leader: base.append("Weekly | Executive AMA to rally support; dispel rumors. KPI: eNPS ↑.")
    if "Visionary" in leader: base.append("Quarterly | Back-cast vision into 3 horizons. KPI: horizon milestones.")
    if "Laissez-Faire" in leader: base.append("Biweekly | Minimal check-ins, dashboards for visibility. KPI: SLA adherence.")
    if "Servant" in leader: base.append("Monthly | Psychological safety & performance balance review. KPI: attrition ↓.")
    return base[:20]

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.header("Settings")
    delay = st.slider("Thinking delay per agent (s)", 0.0, 0.2, 0.04, 0.01)
    use_llm = st.checkbox("Use tiny local LLM for mitigation (if available)", value=True and LLM_AVAILABLE, help="Uses distilgpt2 if installed; otherwise falls back.")
    st.caption("No API keys. Wikipedia & RSS are used for lightweight context. LLM is optional.")

# ---------------------------------
# Inputs
# ---------------------------------
st.title("🤖 Agentic AI CEO — 10 Leadership Agents, LLM Mitigations, Live Knowledge Graph")
st.caption("Enter a Problem and the CEO Decision. Each agent runs FMEA and writes a mitigation plan tied to your scenario.")

c1,c2 = st.columns([1,1])
problem = c1.text_area("Problem", height=120, placeholder="Describe the business problem …")
decision = c2.text_area("Decision taken by CEO", height=120, placeholder="Describe the decision …")
run = st.button("Run FMEA with 10 Leadership Agents")

# ---------------------------------
# Run
# ---------------------------------
if run:
    if not problem.strip() or not decision.strip():
        st.warning("Please provide both Problem and Decision.")
        st.stop()

    st.success("Collecting context …")
    themes = extract_themes(problem, decision)
    if not themes:
        themes = {"Market":0.34, "Product":0.33, "Finance":0.33}

    wiki = wiki_context(problem, decision, limit=3)
    news = news_context(limit=5)

    # Knowledge Graph
    st.subheader("Knowledge Graph")
    G = build_graph(problem, decision, themes, wiki, news)
    nodes_df, edges_df = graph_dataframe(G)
    draw_graph(nodes_df, edges_df)

    # Agents
    st.subheader("Leadership Agents — FMEA + Mitigations")
    per_agent = []
    combined_lines = []

    for leader, cfg in LEADERSHIP.items():
        with st.expander(leader, expanded=False):
            time.sleep(delay)
            S,O,D,RPN = fmea_scores(problem, decision, leader)
            per_agent.append({"Leader": leader, "S": S, "O": O, "D": D, "RPN": RPN})

            st.caption(cfg["desc"])            
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Severity", S); m2.metric("Occurrence", O); m3.metric("Detection", D); m4.metric("RPN", RPN)
            st.markdown(
                f"**Failure Mode:** Applying '{decision[:140]}' to '{problem[:140]}' may create execution gaps, depending on {leader} style."
            )

            # Write mitigations
            lines = []
            if use_llm and LLM_AVAILABLE:
                lines = mitigation_llm(problem, decision, themes, wiki, news, leader)
            if not lines:
                lines = mitigation_rule_based(problem, decision, themes, leader)

            # Show
            st.markdown("**Mitigation Strategy (10–20 lines):**")
            for ln in lines:
                st.markdown(f"- {ln}")
            combined_lines.extend([(leader, ln) for ln in lines])

    st.success("All agents completed.")

    # Summary & Downloads
    st.subheader("Summary & Downloads")
    scores_df = pd.DataFrame(per_agent).sort_values("RPN", ascending=False)
    st.dataframe(scores_df, use_container_width=True)
    chart = (
        alt.Chart(scores_df)
        .mark_bar()
        .encode(x=alt.X("Leader:N", sort="-y"), y="RPN:Q", tooltip=["Leader","RPN","S","O","D"])  
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Combined 30/60/90 roadmap from top themes
    st.subheader("Combined 30/60/90 Roadmap")
    top_t = [k for k,_ in sorted(themes.items(), key=lambda x:x[1], reverse=True)[:3]] or ["Market","Product","Finance"]
    roadmap = [
        {"Timeline":"0-30d","Action":f"Stand up tiger team for {top_t[0]} and translate decision into execution.","Owner":"COO","KPI":"Milestones on time"},
        {"Timeline":"0-30d","Action":f"Risk register + daily dashboard for {', '.join(top_t)}.","Owner":"PMO","KPI":"Critical risks down"},
        {"Timeline":"30-60d","Action":f"Run 3 experiments aligned to '{decision[:60]}'.","Owner":"Analytics","KPI":"Lift vs control"},
        {"Timeline":"60-90d","Action":"Publish outcomes; reset targets.","Owner":"Comms","KPI":"Sentiment ↑"},
    ]
    rmap_df = pd.DataFrame(roadmap)
    st.dataframe(rmap_df, use_container_width=True)

    st.download_button(
        "Download FMEA Scores (CSV)", data=scores_df.to_csv(index=False).encode(),
        file_name="fmea_scores.csv", mime="text/csv"
    )
    full = {
        "timestamp": datetime.utcnow().isoformat(),
        "problem": problem, "decision": decision,
        "themes": themes, "wiki": wiki, "news": news,
        "agents": per_agent, "combined_mitigations": combined_lines,
        "roadmap": roadmap,
    }
    st.download_button(
        "Download Full Results (JSON)", data=json.dumps(full, indent=2).encode(),
        file_name="agentic_ceo_full.json", mime="application/json"
    )
else:
    st.info("Enter a Problem and a Decision, then click **Run FMEA with 10 Leadership Agents**.")
