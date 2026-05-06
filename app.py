"""
Streamlit Dashboard — Multi-Horizon AOP Early Warning System
=============================================================
Predicts apnea probability at 30s, 60s, and 120s horizons.
Shows a 120-second vital-signs trend chart with reference bands.

Run with:  streamlit run app.py
"""

from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from clinical_logic import (EventTracker, classify_risk, detect_event,
                            get_recommendation, risk_color)
from setup import setup_if_needed

# -------------------------------
# Auto-setup: generate data + train models on first launch
# (needed for cloud deployment where we can't run scripts manually)
# -------------------------------
setup_if_needed()

# -------------------------------
# Configuration
# -------------------------------
HORIZONS = [30, 60, 120]
WINDOW_SIZE = 120  # seconds — long enough for PB detection AND the trend chart

st.set_page_config(
    page_title="AOP Early Warning System",
    page_icon="🫁",
    layout="wide",
)


# -------------------------------
# Load models bundle (cached)
# -------------------------------
@st.cache_resource
def load_models():
    return joblib.load("aop_models.pkl")


models = load_models()


# -------------------------------
# Session state initialization
# -------------------------------
if "hr_history" not in st.session_state:
    st.session_state.hr_history = deque(maxlen=WINDOW_SIZE)
    st.session_state.spo2_history = deque(maxlen=WINDOW_SIZE)
    st.session_state.rr_history = deque(maxlen=WINDOW_SIZE)
    # One probability buffer per horizon
    st.session_state.prob_history = {h: deque(maxlen=WINDOW_SIZE) for h in HORIZONS}
    st.session_state.event_tracker = EventTracker()
    # Pre-fill with baseline so charts/detection are sensible from t=0
    for _ in range(WINDOW_SIZE):
        st.session_state.hr_history.append(150)
        st.session_state.spo2_history.append(97)
        st.session_state.rr_history.append(45)
        for h in HORIZONS:
            st.session_state.prob_history[h].append(0.05)


# -------------------------------
# Header
# -------------------------------
st.title("🫁 AOP Early Warning & Decision Support System")
st.caption("Multi-horizon apnea prediction (30 s • 60 s • 120 s) — MVP")


# -------------------------------
# Sidebar — vital sign inputs
# -------------------------------
with st.sidebar:
    st.header("📊 Patient Vital Signs")
    st.caption("Adjust the sliders to simulate live bedside readings.")

    hr = st.slider("Heart Rate (HR) — bpm", 40, 220, 150, step=1)
    spo2 = st.slider("Oxygen Saturation (SpO₂) — %", 60, 100, 97, step=1)
    rr = st.slider("Respiratory Rate (RR) — breaths/min", 0, 90, 45, step=1)

    st.divider()
    st.subheader("🎬 Quick Scenarios")

    if st.button("✅ Normal Vitals"):
        for _ in range(WINDOW_SIZE):
            st.session_state.hr_history.append(150)
            st.session_state.spo2_history.append(97)
            st.session_state.rr_history.append(45)
        st.rerun()

    if st.button("📉 Gradual Deterioration"):
        # Slow downward trend over 120s — shows 120s horizon firing first
        seq_hr, seq_spo2, seq_rr = [], [], []
        for i in range(WINDOW_SIZE):
            p = i / WINDOW_SIZE  # 0 -> 1
            seq_hr.append(150 - 25 * p)
            seq_spo2.append(97 - 6 * p)
            seq_rr.append(45 - 20 * p)
        st.session_state.hr_history = deque(seq_hr, maxlen=WINDOW_SIZE)
        st.session_state.spo2_history = deque(seq_spo2, maxlen=WINDOW_SIZE)
        st.session_state.rr_history = deque(seq_rr, maxlen=WINDOW_SIZE)
        st.rerun()

    if st.button("🚨 Simulate AOP Episode"):
        for _ in range(95):
            st.session_state.hr_history.append(150)
            st.session_state.spo2_history.append(97)
            st.session_state.rr_history.append(45)
        for _ in range(25):
            st.session_state.hr_history.append(85)
            st.session_state.spo2_history.append(82)
            st.session_state.rr_history.append(2)
        st.rerun()

    if st.button("💨 Simulate Periodic Breathing"):
        seq_hr, seq_spo2, seq_rr = [], [], []
        for _ in range(5):
            seq_hr.extend([150] * 15);  seq_spo2.extend([97] * 15);  seq_rr.extend([45] * 15)
            seq_hr.extend([150] * 7);   seq_spo2.extend([97] * 7);   seq_rr.extend([2]  * 7)
        pad = WINDOW_SIZE - len(seq_hr)
        seq_hr = [150]*pad + seq_hr
        seq_spo2 = [97]*pad + seq_spo2
        seq_rr = [45]*pad + seq_rr
        st.session_state.hr_history = deque(seq_hr, maxlen=WINDOW_SIZE)
        st.session_state.spo2_history = deque(seq_spo2, maxlen=WINDOW_SIZE)
        st.session_state.rr_history = deque(seq_rr, maxlen=WINDOW_SIZE)
        st.rerun()


# -------------------------------
# Append the current slider reading
# -------------------------------
st.session_state.hr_history.append(hr)
st.session_state.spo2_history.append(spo2)
st.session_state.rr_history.append(rr)


# -------------------------------
# Run all THREE model predictions
# -------------------------------
features = pd.DataFrame([[hr, spo2, rr]], columns=["HR", "SpO2", "RR"])
probabilities = {}
risk_levels = {}
for h in HORIZONS:
    p = float(models[h].predict_proba(features)[0, 1])
    probabilities[h] = p
    risk_levels[h] = classify_risk(p)
    st.session_state.prob_history[h].append(p)

# Use the 60s horizon as the "primary" risk level driving recommendations
# (clinically most actionable — long enough to react, short enough to be reliable)
primary_horizon = 60
primary_probability = probabilities[primary_horizon]
primary_risk = risk_levels[primary_horizon]


# -------------------------------
# Detection on the 120s window
# -------------------------------
hr_w = list(st.session_state.hr_history)
spo2_w = list(st.session_state.spo2_history)
rr_w = list(st.session_state.rr_history)

current_event = detect_event(hr_w, spo2_w, rr_w)
st.session_state.event_tracker.log_event(current_event)
recommendation = get_recommendation(primary_risk, current_event)


# =================================================================
# MAIN LAYOUT — Multi-Horizon Risk Panel
# =================================================================
st.subheader("⚡ Multi-Horizon Apnea Prediction")

cols = st.columns(3)
horizon_labels = {30: "30 seconds", 60: "60 seconds", 120: "120 seconds"}
horizon_descriptions = {
    30: "Imminent — react NOW",
    60: "Short-term warning",
    120: "Early heads-up",
}

for col, h in zip(cols, HORIZONS):
    with col:
        risk = risk_levels[h]
        prob = probabilities[h]
        color = risk_color(risk)
        st.markdown(
            f"""
            <div style="background:{color}; padding:18px; border-radius:10px;
                        text-align:center; color:white; margin-bottom:10px;">
                <div style="font-size:13px; opacity:0.85;">PREDICTION HORIZON</div>
                <div style="font-size:24px; font-weight:bold;">{horizon_labels[h]}</div>
                <div style="font-size:11px; opacity:0.85; margin-top:4px;">
                    {horizon_descriptions[h]}
                </div>
                <hr style="border-color:rgba(255,255,255,0.3); margin:10px 0;">
                <div style="font-size:36px; font-weight:bold;">{prob*100:.1f}%</div>
                <div style="font-size:18px;">Risk: {risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob)

st.divider()

# -------------------------------
# Current Event Display
# -------------------------------
event_color = {"AOP": "#dc3545", "PB": "#ffc107", "None": "#28a745"}[current_event]
event_label = {"AOP": "🚨 AOP DETECTED",
               "PB": "💨 Periodic Breathing",
               "None": "✅ No Active Event"}[current_event]

st.markdown(
    f"""
    <div style="background:{event_color}; padding:15px; border-radius:10px;
                text-align:center; color:white;">
        <div style="font-size:13px; opacity:0.9;">CURRENT DETECTED EVENT</div>
        <div style="font-size:28px; font-weight:bold;">{event_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -------------------------------
# Clinical Recommendation
# -------------------------------
st.subheader("🩺 Clinical Recommendation")
st.caption(f"Based on the {primary_horizon}-second horizon (primary clinical action window)")

urgency_color = {
    "EMERGENCY": "🔴",
    "URGENT": "🟠",
    "CAUTION": "🟡",
    "INFORMATIONAL": "🔵",
    "ROUTINE": "🟢",
}.get(recommendation["urgency"], "⚪")

st.markdown(f"### {urgency_color} {recommendation['title']}")
st.markdown(f"**Urgency: `{recommendation['urgency']}`**")
st.markdown("**Recommended Actions:**")
for action in recommendation["actions"]:
    st.markdown(f"- {action}")

st.divider()

# -------------------------------
# Event Tracking
# -------------------------------
st.subheader("📈 Event Tracking")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("AOP events — last 60 minutes",
              st.session_state.event_tracker.aop_count_last_hour())
with c2:
    st.metric("AOP events — last 24 hours",
              st.session_state.event_tracker.aop_count_last_24h())
with c3:
    last_ts, last_type = st.session_state.event_tracker.get_last_event()
    if last_ts:
        elapsed = (datetime.now() - last_ts).total_seconds()
        if elapsed < 60:
            elapsed_str = f"{int(elapsed)} seconds ago"
        elif elapsed < 3600:
            elapsed_str = f"{int(elapsed // 60)} minutes ago"
        else:
            elapsed_str = f"{int(elapsed // 3600)} hours ago"
        st.metric("Last detected event", last_type,
                  delta=elapsed_str, delta_color="off")
    else:
        st.metric("Last detected event", "None")

st.divider()

# =================================================================
# ENHANCED VITAL SIGNS TREND — 120 SECOND WINDOW
# =================================================================
st.subheader("📊 Vital Signs — Last 120 Seconds")
st.caption("Solid line = patient value · Dashed line = clinical alert threshold")

# Time axis: -120s ... 0s ("now" on the right)
time_axis = np.arange(-WINDOW_SIZE + 1, 1)

hr_arr = np.array(st.session_state.hr_history)
spo2_arr = np.array(st.session_state.spo2_history)
rr_arr = np.array(st.session_state.rr_history)

# Build three side-by-side charts so each vital has its own scale + reference line
chart_cols = st.columns(3)

with chart_cols[0]:
    st.markdown("**❤️ Heart Rate (bpm)**")
    hr_df = pd.DataFrame({
        "HR": hr_arr,
        "Bradycardia threshold (<100)": [100] * WINDOW_SIZE,
    }, index=time_axis)
    hr_df.index.name = "Seconds (relative to now)"
    st.line_chart(hr_df, height=240, color=["#e63946", "#888888"])
    st.caption(
        f"Current: **{hr:.0f} bpm**  •  "
        f"Min(120s): {hr_arr.min():.0f}  •  "
        f"Mean: {hr_arr.mean():.0f}"
    )

with chart_cols[1]:
    st.markdown("**🫁 SpO₂ (%)**")
    spo2_df = pd.DataFrame({
        "SpO2": spo2_arr,
        "Desaturation threshold (<90)": [90] * WINDOW_SIZE,
    }, index=time_axis)
    spo2_df.index.name = "Seconds (relative to now)"
    st.line_chart(spo2_df, height=240, color=["#2a9d8f", "#888888"])
    st.caption(
        f"Current: **{spo2:.0f}%**  •  "
        f"Min(120s): {spo2_arr.min():.0f}%  •  "
        f"Mean: {spo2_arr.mean():.0f}%"
    )

with chart_cols[2]:
    st.markdown("**💨 Respiratory Rate (bpm)**")
    rr_df = pd.DataFrame({
        "RR": rr_arr,
        "Apnea threshold (<5)": [5] * WINDOW_SIZE,
    }, index=time_axis)
    rr_df.index.name = "Seconds (relative to now)"
    st.line_chart(rr_df, height=240, color=["#264653", "#888888"])
    st.caption(
        f"Current: **{rr:.0f} bpm**  •  "
        f"Min(120s): {rr_arr.min():.0f}  •  "
        f"Mean: {rr_arr.mean():.0f}"
    )

st.divider()

# -------------------------------
# Probability trend — all 3 horizons together
# -------------------------------
st.subheader("🤖 Apnea Probability Trend (all 3 horizons)")

prob_df = pd.DataFrame({
    "30s horizon": [p * 100 for p in st.session_state.prob_history[30]],
    "60s horizon": [p * 100 for p in st.session_state.prob_history[60]],
    "120s horizon": [p * 100 for p in st.session_state.prob_history[120]],
}, index=time_axis)
prob_df.index.name = "Seconds (relative to now)"
st.line_chart(prob_df, height=280, color=["#dc3545", "#fd7e14", "#0d6efd"])
st.caption("Probability (%) over the last 120 seconds. The 30s line typically reacts last but most sharply.")

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(
    "⚠️ **Disclaimer:** MVP for demonstration only. "
    "Not validated for clinical use. Always defer to qualified medical judgment."
)
