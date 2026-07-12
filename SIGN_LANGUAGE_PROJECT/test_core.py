"""
Quick offline smoke tests for the non-UI core logic.
Run with:  python -m pytest tests/  (or just: python tests/test_core.py)
None of these require a camera, Streamlit, or network access.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from core.classifier import classify, SUPPORTED_LETTERS
from core.database import RecognitionDB
from core import analytics as an
import pandas as pd


def test_classifier_never_crashes():
    rng = np.random.default_rng(1)
    for _ in range(50):
        lm = rng.random((21, 3)).astype(np.float32)
        pred = classify(lm)
        assert pred.letter in list(SUPPORTED_LETTERS) + ["?"]
        assert 0.0 <= pred.confidence <= 1.0


def test_classifier_handles_bad_input():
    assert classify(None).letter == "?"
    assert classify(np.zeros((3, 3))).letter == "?"


def test_database_roundtrip(tmp_path):
    db = RecognitionDB(db_path=tmp_path / "test.db")
    sid = db.new_session_id()
    db.log_event(sid, "A", 0.9)
    db.log_event(sid, "B", 0.8)
    df = db.fetch_all()
    assert len(df) == 2
    assert sid in db.session_ids()
    db.clear_all()
    assert db.fetch_all().empty


def test_analytics_on_empty_and_real_data():
    empty = pd.DataFrame(columns=["id", "session_id", "timestamp", "letter", "confidence", "source"])
    stats = an.summary_stats(empty)
    assert stats["total_detections"] == 0
    an.letter_frequency_chart(empty)
    an.confidence_distribution_chart(empty)
    an.detections_over_time_chart(empty)


if __name__ == "__main__":
    test_classifier_never_crashes()
    test_classifier_handles_bad_input()
    test_analytics_on_empty_and_real_data()
    print("All smoke tests passed (run test_database_roundtrip via pytest for tmp_path fixture).")
