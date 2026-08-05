"""
app.py
------
Sign Language Recognition + Analytics — Streamlit application.

Run with:
    streamlit run app.py

This is the only file that imports Streamlit; all detection, rule
logic, storage and analytics live in core/ and are independently
tested (see core/*.py). This separation keeps the UI layer thin so a
bug in one place can't take down the whole app.
"""
from __future__ import annotations

import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.analytics import (
    confidence_distribution_chart,
    detections_over_time_chart,
    letter_frequency_chart,
    session_summary_table,
    summary_stats,
)
from core.classifier import SUPPORTED_LETTERS, classify
from core.database import RecognitionDB
from core.hand_tracker import HandTracker

# ---------------------------------------------------------------- config
st.set_page_config(
    page_title="SignSense · ASL Recognition & Analytics",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0F1220;
    --card: #171B2E;
    --primary: #6C5CE7;
    --accent: #00CEC9;
    --text: #EAEAF4;
    --muted: #9AA0C3;
}
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b2040 0%, #0F1220 55%); }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: 0.2px; }
p, span, label, li { color: var(--muted); }

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(108,92,231,0.25), rgba(0,206,201,0.12));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.hero h1 { font-size: 2.1rem; margin-bottom: 0.2rem; }
.hero p { font-size: 1.0rem; color: var(--muted); margin: 0; }

.metric-card {
    background: var(--card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
.metric-card .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

.pred-badge {
    display: inline-block;
    font-size: 3.2rem;
    font-weight: 800;
    padding: 0.4rem 1.2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
    box-shadow: 0 8px 24px rgba(108,92,231,0.35);
}
.section-card {
    background: var(--card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] { background: #12142a; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- state
@st.cache_resource(show_spinner=False)
def get_tracker() -> HandTracker:
    return HandTracker(static_image_mode=True, max_num_hands=1)


@st.cache_resource(show_spinner=False)
def get_db() -> RecognitionDB:
    return RecognitionDB()


if "session_id" not in st.session_state:
    st.session_state.session_id = RecognitionDB.new_session_id()
if "history" not in st.session_state:
    st.session_state.history = []  # in-memory quick view for this run

db = get_db()
tracker = get_tracker()


# ---------------------------------------------------------------- helpers
def run_recognition(pil_image: Image.Image):
    """Runs hand detection + classification on a PIL image. Never raises;
    returns (annotated_rgb_array | None, prediction | None, error | None)."""
    try:
        rgb = np.array(pil_image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        hands = tracker.process(bgr)
        if not hands:
            return rgb, None, None
        hand = hands[0]
        pred = classify(hand.landmarks, hand.handedness)
        annotated_bgr = tracker.draw(bgr, hands)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        return annotated_rgb, pred, None
    except Exception as exc:  # defensive: UI should never hard-crash
        return None, None, f"{type(exc).__name__}: {exc}"


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🤟 SignSense")
    st.caption("ASL recognition + real-time analytics")
    page = st.radio(
        "Navigate",
        ["📷 Live Recognition", "📊 Analytics Dashboard", "🗂️ History & Export", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.caption("Supported letters:")
    st.code(" ".join(SUPPORTED_LETTERS), language=None)


# ---------------------------------------------------------------- pages
if page == "📷 Live Recognition":
    st.markdown(
        '<div class="hero"><h1>Live Sign Recognition</h1>'
        '<p>Show a static ASL letter to your camera (or upload a photo) and SignSense '
        'will identify it and log it to your analytics dashboard.</p></div>',
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        mode = st.radio("Input source", ["Camera snapshot", "Upload image"], horizontal=True)
        image = None
        if mode == "Camera snapshot":
            cam_img = st.camera_input("Take a photo of your hand sign")
            if cam_img is not None:
                image = Image.open(cam_img)
        else:
            up = st.file_uploader("Upload a hand sign image", type=["jpg", "jpeg", "png"])
            if up is not None:
                image = Image.open(up)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if image is None:
            st.info("Waiting for an image — take a snapshot or upload one to begin.")
        else:
            with st.spinner("Analyzing hand landmarks..."):
                annotated, pred, err = run_recognition(image)

            if err:
                st.error(f"Recognition failed: {err}")
            elif annotated is None:
                st.warning("Could not read the image. Please try another one.")
            else:
                st.image(annotated, caption="Detected landmarks", use_container_width=True)
                if pred is None:
                    st.warning("No hand detected. Make sure your hand is clearly visible.")
                elif pred.letter == "?":
                    st.warning("Hand detected, but the sign didn't match a known letter confidently.")
                else:
                    st.markdown(
                        f'<div class="pred-badge">{pred.letter}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(pred.confidence, 0.0), 1.0))
                    st.caption(f"Confidence: {pred.confidence:.0%}")

                    try:
                        db.log_event(st.session_state.session_id, pred.letter, pred.confidence)
                        st.session_state.history.append(
                            {"time": datetime.now().strftime("%H:%M:%S"),
                             "letter": pred.letter, "confidence": pred.confidence}
                        )
                        st.success("Logged to analytics ✅")
                    except Exception as exc:
                        st.error(f"Could not save to database: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("#### This session")
        st.dataframe(
            pd.DataFrame(st.session_state.history[::-1]),
            use_container_width=True, hide_index=True,
        )

elif page == "📊 Analytics Dashboard":
    st.markdown(
        '<div class="hero"><h1>Analytics Dashboard</h1>'
        '<p>Insights across every recognition session recorded on this device.</p></div>',
        unsafe_allow_html=True,
    )

    try:
        df = db.fetch_all()
    except Exception as exc:
        st.error(f"Could not load analytics data: {exc}")
        df = pd.DataFrame()

    stats = summary_stats(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Detections", str(stats["total_detections"]))
    with c2: metric_card("Unique Letters", str(stats["unique_letters"]))
    with c3: metric_card("Avg. Confidence", f'{stats["avg_confidence"]:.0%}')
    with c4: metric_card("Most Common", stats["top_letter"])
    with c5: metric_card("Sessions", str(stats["sessions"]))

    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.pyplot(letter_frequency_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.pyplot(confidence_distribution_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.pyplot(detections_over_time_chart(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Session Summary")
    st.dataframe(session_summary_table(df), use_container_width=True, hide_index=True)

elif page == "🗂️ History & Export":
    st.markdown(
        '<div class="hero"><h1>History & Export</h1>'
        '<p>Browse the full recognition log and export it as CSV.</p></div>',
        unsafe_allow_html=True,
    )
    try:
        df = db.fetch_all()
    except Exception as exc:
        st.error(f"Could not load history: {exc}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No recognition events logged yet. Head to Live Recognition to get started.")
    else:
        st.dataframe(df.sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                            file_name="sign_language_history.csv", mime="text/csv")

        st.divider()
        if st.button("🗑️ Clear all history", type="secondary"):
            try:
                db.clear_all()
                st.session_state.history = []
                st.success("History cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not clear history: {exc}")

else:  # About
    st.markdown(
        '<div class="hero"><h1>About SignSense</h1>'
        '<p>How the recognition pipeline works.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-card">
        <b>Pipeline</b><br>
        1. <b>MediaPipe Hands</b> extracts 21 3D landmarks per hand from an image.<br>
        2. A transparent, <b>rule-based geometric classifier</b> (no black-box model,
        no training data required) maps finger-extension patterns and distances
        to one of the static ASL letters.<br>
        3. Every prediction is logged to a local <b>SQLite</b> database.<br>
        4. The <b>Analytics Dashboard</b> summarizes that log with pandas + matplotlib.
        </div>
        <div class="section-card">
        <b>Supported letters</b><br>
        This build recognizes the static letters that don't require motion:
        A, B, C, D, E, F, I, L, O, R, S, U, V, W, Y.<br><br>
        <b>J</b> and <b>X/Z/etc.</b> that require motion or fine finger-crossing are
        out of scope for a single-frame classifier and are intentionally omitted
        rather than guessed unreliably.
        </div>
        """,
        unsafe_allow_html=True,
    )
