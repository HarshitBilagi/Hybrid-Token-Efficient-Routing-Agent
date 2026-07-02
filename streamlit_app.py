"""
Hybrid Token-Efficient Routing Agent — Streamlit Demo UI

Thin visual wrapper around the FastAPI /route endpoint. Run the API server
first (uvicorn src.api.main:app --port 8000), then launch this:

    streamlit run streamlit_app.py

Portfolio note: this is intentionally a demo shell, not the engineering
deliverable. The routing logic, classifiers, and eval harness are the
substance of the project — this just makes it visible and screenshot-able.
"""

import time
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Hybrid Routing Agent",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling — minimal dark theme, NPU-purple / remote-teal accent colors
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at top, #1a1c2e 0%, #0d0e1a 60%);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #5eead4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #8b8ea3;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    .hw-chip {
        display: inline-block;
        background: rgba(167, 139, 250, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.25);
        color: #c4b5fd;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
    }

    .route-card {
        border-radius: 14px;
        padding: 18px 22px;
        margin-top: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        animation: fadeIn 0.4s ease-out;
    }
    .route-local {
        background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(167,139,250,0.03));
        border-color: rgba(167,139,250,0.3);
    }
    .route-remote {
        background: linear-gradient(135deg, rgba(94,234,212,0.12), rgba(94,234,212,0.03));
        border-color: rgba(94,234,212,0.3);
    }

    .route-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .route-local .route-label { color: #c4b5fd; }
    .route-remote .route-label { color: #5eead4; }

    .response-text {
        color: #e5e7eb;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 10px;
    }

    .metric-row {
        display: flex;
        gap: 24px;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .metric-item { font-family: 'JetBrains Mono', monospace; }
    .metric-value { color: #f1f5f9; font-size: 1.1rem; font-weight: 600; }
    .metric-label { color: #6b7280; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }

    .stTextInput input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
    }
    .stTextInput input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 1px #a78bfa !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #a78bfa, #5eead4) !important;
        color: #0d0e1a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 28px !important;
        transition: transform 0.15s ease !important;
    }
    .stButton button:hover { transform: translateY(-1px); }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .footer-note {
        text-align: center;
        color: #4b5563;
        font-size: 0.75rem;
        margin-top: 3rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Hybrid Token-Efficient Routing Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    '<span class="hw-chip">⚡ Intel AI Boost NPU</span>'
    '<span class="hw-chip">☁️ Groq · Llama-3.3-70B</span>'
    '<span class="hw-chip">🎯 Output-Length-Aware Routing</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.json()
    except requests.exceptions.RequestException:
        return None

health = check_health()

if health is None:
    st.error("⚠️ API server not reachable. Run `uvicorn src.api.main:app --port 8000` first.")
    st.stop()
elif not health.get("npu_ready"):
    st.warning("⏳ NPU pipeline still loading — first startup takes ~30s.")

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "query",
        placeholder="Ask anything — try a simple fact, some code, or a current-events question…",
        label_visibility="collapsed",
    )
with col2:
    classifier = st.selectbox(
        "classifier",
        ["rule_based", "llm_judged"],
        label_visibility="collapsed",
    )

run = st.button("Route query →", use_container_width=False)

example_queries = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Who is the current president of the United States?",
    "Explain the difference between TCP and UDP.",
]

st.markdown(
    '<div style="margin-top: -8px; margin-bottom: 8px;">'
    + " · ".join(f'<span style="color:#6b7280; font-size:0.8rem;">{q}</span>' for q in example_queries)
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Route + display
# ---------------------------------------------------------------------------
if run and query.strip():
    with st.spinner("Classifying and routing…"):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_URL}/route",
                json={
                    "query": query,
                    "context": [],
                    "constraints": {"prefer_local": False},
                    "metadata": {},
                    "classifier": classifier,
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()
        wall_ms = round((time.perf_counter() - t0) * 1000)

    route = result["route_taken"]
    card_class = "route-local" if route == "local" else "route-remote"
    icon = "⚡" if route == "local" else "☁️"
    route_display = "LOCAL · Intel AI Boost NPU" if route == "local" else "REMOTE · Groq Llama-3.3-70B"

    fallback_note = ""
    if result.get("fallback_triggered"):
        fallback_note = f'<div style="color:#f59e0b; font-size:0.75rem; margin-top:6px;">↳ fallback triggered: {result.get("fallback_reason", "unknown")}</div>'

    st.markdown(f"""
    <div class="route-card {card_class}">
        <div class="route-label">{icon} {route_display}</div>
        <div class="response-text">{result['response']}</div>
        {fallback_note}
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value">{result['latency_ms']}ms</div>
                <div class="metric-label">Inference Latency</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{result['tokens'].get('completion', 0)}</div>
                <div class="metric-label">Tokens Generated</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{result['classifier_used']}</div>
                <div class="metric-label">Classifier</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{wall_ms}ms</div>
                <div class="metric-label">Total Wall Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Classifier signals"):
        st.json(result.get("classifier_signals", {}))

# ---------------------------------------------------------------------------
# Benchmark snapshot — static, from eval/results
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Benchmark results (19-query eval set)"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy %",
        x=["rule_based", "llm_judged"],
        y=[89.5, 100.0],
        marker_color="#a78bfa",
        text=["89.5%", "100.0%"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Token Savings %",
        x=["rule_based", "llm_judged"],
        y=[19.1, 14.3],
        marker_color="#5eead4",
        text=["19.1%", "14.3%"],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)"),
        font=dict(family="Inter", color="#e5e7eb"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "LLM-judged classifier trades 4.8% token savings for a 10.5pp accuracy gain "
        "by correctly identifying which simple queries the INT4 local model handles unreliably."
    )

st.markdown(
    '<div class="footer-note">Phi-3.5-mini INT4 on Intel AI Boost NPU · '
    "OpenVINO GenAI · FastAPI</div>",
    unsafe_allow_html=True,
)