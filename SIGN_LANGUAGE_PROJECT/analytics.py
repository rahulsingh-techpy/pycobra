"""
analytics.py
-------------
Turns the raw recognition-event log (pandas DataFrame) into summary
statistics and matplotlib figures for the dashboard. Every function is
defensive against empty data so the dashboard never crashes on a fresh
install with no history yet.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # safe for headless / server rendering
import matplotlib.pyplot as plt
import pandas as pd

# ---- palette -----------------------------------------------------------
PRIMARY = "#6C5CE7"
ACCENT = "#00CEC9"
WARN = "#FDCB6E"
BG = "#0F1220"
GRID = "#2A2E45"
TEXT = "#EAEAF4"


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)


def summary_stats(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "total_detections": 0,
            "unique_letters": 0,
            "avg_confidence": 0.0,
            "top_letter": "-",
            "sessions": 0,
        }
    return {
        "total_detections": int(len(df)),
        "unique_letters": int(df["letter"].nunique()),
        "avg_confidence": float(df["confidence"].mean()),
        "top_letter": str(df["letter"].mode().iloc[0]) if not df["letter"].mode().empty else "-",
        "sessions": int(df["session_id"].nunique()),
    }


def letter_frequency_chart(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=140)
    _style_axes(ax)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color=TEXT)
        return fig
    counts = df["letter"].value_counts().sort_values(ascending=False)
    ax.bar(counts.index, counts.values, color=PRIMARY, edgecolor=ACCENT, linewidth=0.5)
    ax.set_title("Detections per Letter")
    ax.set_xlabel("Letter")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def confidence_distribution_chart(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=140)
    _style_axes(ax)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color=TEXT)
        return fig
    ax.hist(df["confidence"], bins=12, color=ACCENT, edgecolor=BG, alpha=0.9)
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def detections_over_time_chart(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=140)
    _style_axes(ax)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color=TEXT)
        return fig
    ts = df.set_index("timestamp").resample("1min").size()
    ax.plot(ts.index, ts.values, color=WARN, linewidth=2, marker="o", markersize=3)
    ax.set_title("Detections Over Time (per minute)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Detections")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def session_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["session_id", "detections", "avg_confidence", "start", "end"])
    g = df.groupby("session_id").agg(
        detections=("letter", "count"),
        avg_confidence=("confidence", "mean"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    ).reset_index().sort_values("start", ascending=False)
    g["avg_confidence"] = g["avg_confidence"].round(3)
    return g
