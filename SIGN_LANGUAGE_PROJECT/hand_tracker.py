"""
hand_tracker.py
----------------
Thin, defensive wrapper around MediaPipe Hands for extracting 21 hand
landmarks (x, y, z) from a single BGR image (as returned by OpenCV).

Kept deliberately dependency-light and side-effect-free so it can be
unit tested without a real camera or Streamlit running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandResult:
    """Container for a single detected hand."""
    landmarks: np.ndarray          # shape (21, 3) -> x, y, z (normalized 0-1)
    handedness: str                # "Left" or "Right"
    score: float                   # detection confidence 0-1
    bbox: tuple                    # (x_min, y_min, x_max, y_max) in pixels


class HandTracker:
    """Detects hands and returns normalized landmarks + drawing helpers."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        """Run detection on a BGR frame. Returns a list of HandResult."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        hands_out: List[HandResult] = []
        if not results.multi_hand_landmarks:
            return hands_out

        h, w = frame_bgr.shape[:2]
        handedness_list = results.multi_handedness or []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            bbox = (
                float(np.min(xs)), float(np.min(ys)),
                float(np.max(xs)), float(np.max(ys)),
            )

            label = "Right"
            score = 1.0
            if i < len(handedness_list) and handedness_list[i].classification:
                cls = handedness_list[i].classification[0]
                label = cls.label
                score = cls.score

            hands_out.append(HandResult(landmarks=pts, handedness=label,
                                         score=score, bbox=bbox))
        return hands_out

    def draw(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe hand landmarks onto a copy of the frame for display."""
        annotated = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = self._hands.process(rgb)
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return annotated

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
