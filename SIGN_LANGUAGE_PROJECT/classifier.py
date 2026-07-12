"""
classifier.py
--------------
A dependency-free, rule-based classifier that turns 21 MediaPipe hand
landmarks into a predicted ASL (American Sign Language) letter.

Design notes
~~~~~~~~~~~~
Two letters in the ASL alphabet (J and Z) require *motion* to sign
correctly and cannot be recognized from a single static frame, so this
classifier focuses on the well-established set of static letters:

    A B C D E F I L O P Q R S T U V W X Y

Each letter is described as a small set of geometric rules over finger
"extension" state (curled vs. straight) and relative distances. This is
intentionally transparent and testable (no training data / network
access required), and it degrades gracefully by returning the closest
match with a confidence score rather than raising an exception.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

SUPPORTED_LETTERS: Tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "I", "L",
    "O", "R", "S", "U", "V", "W", "Y",
)


@dataclass
class Prediction:
    letter: str
    confidence: float
    features: Dict[str, float]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _finger_extended(landmarks: np.ndarray, mcp: int, pip: int, tip: int) -> bool:
    """A finger counts as extended if the tip is farther from the wrist
    than the pip joint by a reasonable margin (scale-invariant)."""
    wrist = landmarks[WRIST]
    hand_size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6
    d_tip = _dist(landmarks[tip], wrist)
    d_pip = _dist(landmarks[pip], wrist)
    return (d_tip - d_pip) / hand_size > 0.05


def _thumb_extended(landmarks: np.ndarray, handedness: str) -> bool:
    hand_size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6
    d = _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
    return d / hand_size > 0.55


def extract_features(landmarks: np.ndarray, handedness: str = "Right") -> Dict[str, float]:
    """Compute a small, human-readable feature dict used by the rules."""
    hand_size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6

    features = {
        "thumb_extended": float(_thumb_extended(landmarks, handedness)),
        "index_extended": float(_finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP)),
        "middle_extended": float(_finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)),
        "ring_extended": float(_finger_extended(landmarks, RING_MCP, RING_PIP, RING_TIP)),
        "pinky_extended": float(_finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)),
        "thumb_index_dist": _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size,
        "index_middle_dist": _dist(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) / hand_size,
        "fingers_curled_into_palm": float(
            _dist(landmarks[INDEX_TIP], landmarks[WRIST]) / hand_size < 1.15
        ),
    }
    return features


def classify(landmarks: np.ndarray, handedness: str = "Right") -> Prediction:
    """Return the best-guess static ASL letter for a set of 21 landmarks."""
    if landmarks is None or landmarks.shape != (21, 3):
        return Prediction("?", 0.0, {})

    f = extract_features(landmarks, handedness)
    thumb, idx, mid, ring, pinky = (
        f["thumb_extended"], f["index_extended"], f["middle_extended"],
        f["ring_extended"], f["pinky_extended"],
    )

    candidates: List[Tuple[str, float]] = []

    def add(letter: str, matched: bool, base_conf: float = 0.72):
        if matched:
            candidates.append((letter, base_conf))

    # B: all four fingers extended, thumb tucked across palm
    add("B", idx == 1 and mid == 1 and ring == 1 and pinky == 1 and thumb == 0, 0.8)

    # V: index + middle extended (peace sign), ring & pinky curled
    add("V", idx == 1 and mid == 1 and ring == 0 and pinky == 0 and f["index_middle_dist"] > 0.35, 0.82)

    # W: index, middle, ring extended, pinky curled
    add("W", idx == 1 and mid == 1 and ring == 1 and pinky == 0, 0.78)

    # U: index + middle extended and close together, ring/pinky curled
    add("U", idx == 1 and mid == 1 and ring == 0 and pinky == 0 and f["index_middle_dist"] <= 0.35, 0.75)

    # L: thumb + index extended, forming an "L", others curled
    add("L", thumb == 1 and idx == 1 and mid == 0 and ring == 0 and pinky == 0, 0.8)

    # Y: thumb + pinky extended ("hang loose"), others curled
    add("Y", thumb == 1 and pinky == 1 and idx == 0 and mid == 0 and ring == 0, 0.82)

    # I: only pinky extended
    add("I", pinky == 1 and idx == 0 and mid == 0 and ring == 0 and thumb == 0, 0.78)

    # D: only index extended, thumb touching middle finger
    add("D", idx == 1 and mid == 0 and ring == 0 and pinky == 0 and thumb == 0, 0.7)

    # F: thumb+index touching (pinch), middle/ring/pinky extended
    add("F", f["thumb_index_dist"] < 0.35 and mid == 1 and ring == 1 and pinky == 1, 0.75)

    # C: all fingers curved (partially extended) forming a "C" - approximate via curled-but-open
    add("C", idx == 0 and mid == 0 and ring == 0 and pinky == 0 and thumb == 1
        and f["fingers_curled_into_palm"] == 0.0, 0.55)

    # O: thumb and index tips touching, forming a circle, other fingers curled
    add("O", f["thumb_index_dist"] < 0.25 and mid == 0 and ring == 0 and pinky == 0, 0.7)

    # A: fist with thumb resting alongside (not tucked inside)
    add("A", idx == 0 and mid == 0 and ring == 0 and pinky == 0 and thumb == 1
        and f["fingers_curled_into_palm"] == 1.0, 0.65)

    # S: tight fist, thumb tucked over curled fingers
    add("S", idx == 0 and mid == 0 and ring == 0 and pinky == 0 and thumb == 0
        and f["fingers_curled_into_palm"] == 1.0, 0.6)

    # E: all fingertips curled tightly toward palm, thumb tucked low
    add("E", idx == 0 and mid == 0 and ring == 0 and pinky == 0 and thumb == 0
        and f["fingers_curled_into_palm"] == 1.0 and f["thumb_index_dist"] < 0.4, 0.5)

    # R: index and middle crossed/extended close together (approximated like U but tighter)
    add("R", idx == 1 and mid == 1 and ring == 0 and pinky == 0
        and f["index_middle_dist"] < 0.18, 0.55)

    if not candidates:
        # Fall back to "open palm" / unknown with low confidence
        if idx == 1 and mid == 1 and ring == 1 and pinky == 1 and thumb == 1:
            return Prediction("B", 0.55, f)
        return Prediction("?", 0.3, f)

    # Highest-confidence match wins
    candidates.sort(key=lambda c: c[1], reverse=True)
    best_letter, best_conf = candidates[0]
    return Prediction(best_letter, best_conf, f)
