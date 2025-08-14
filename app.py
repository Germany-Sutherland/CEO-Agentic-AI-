import streamlit as st
import pandas as pd
import numpy as np
import wikipedia
import feedparser
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------
# APP CONFIG
# ------------------------------
st.set_page_config(page_title="AI CEO Agent", layout="wide")

# Leadership styles
leadership_styles = [
    "Transformational", "Transactional", "Servant", "Autocratic", "Democratic",
    "Laissez-Faire", "Charismatic", "Situational", "Visionary", "Coaching"
]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def run_fmea(style, problem):
    """Generate FMEA table for a leadership style and problem."""
    np.random.seed(len(style) + len(problem))
    data = []
    for failure_mode in ["Decision Delay", "Poor Communication", "Low Morale", "Resource Misallocation"]:
        sev = np.random.randint(1, 10)
        occ = np.random.randint(1, 10)
        det = np.random.randint(1, 10)
        rpn = sev * occ * det
        data.append([failure_mode, sev, occ, det, rpn])
    df = pd.DataFrame(data, columns=["Failure Mode", "Severity", "Occurrence", "Detection", "RPN"])
    return df.sort_values(by="RPN", ascending=False)

def mitigation_strategy(style, fmea_df):
    """Generate mitigation strategy text (simulated NLP)."""
    high_risk = fmea_df.iloc[0]["Failure Mode"]
    return f"""
**Mitigation Strategy for {style} Leadership**
- Primary Risk Identified: {high_risk}
- Implement cross-functional decision committees.
- Enhance transparent communication channels.
- Schedule regular leadership reflection and team feedback.
- Allocate resources based on strategic priorities.
- Encourage innovation while managing risk tolerance.
- Monitor KPIs closely to adapt quickly to changes.
- Train teams to handle {high_risk} scenarios effectively.
- Foster resilience through continuous learning.
- Ensure accountability at all leadership levels.
"""

def get_wiki_summary(topic):
    try:
        return wikipedia.summary(topic, sentences=2)
    except:
        return "No Wikipedia data found."

def get_latest_news(query="leadership"):
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}")
    return [entry.title for entry in feed.entries[:5]]

def create_knowledge_graph(topics):
    G = nx.Graph()
    for t in topics:
        G.add_node(t)
        G.add_edge("Leadership", t)
    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color='lightblue', font_size=8, node_size=2000, ax=ax)
    st.pyplot(fig)

# ------------------------------
# APP LAYOUT
# ------------------------------
st.title("🤖 AI CEO Agent – Fortune 500 Decision Maker")
st.write("This app simulates a CEO with 10 leadership agents, runs FMEA, and suggests mitigation strategies.")

problem_input = st.text_input("Enter a business problem:")
if problem_input:
    # Knowledge graph
    st.subheader("📚 Knowledge Graph from Wikipedia & News")
    wiki_info = [get_wiki_summary(style) for style in leadership_styles[:5]]
    news_info = get_latest_news(problem_input)
    create_knowledge_graph(leadership_styles[:5] + news_info)

    # Loop through agents
    for style in leadership_styles:
        st.markdown(f"### 🧠 {style} Leadership Agent")
        fmea_df = run_fmea(style, problem_input)
        st.dataframe(fmea_df)
        st.markdown(mitigation_strategy(style, fmea_df))
