# 🤟 SignSense — ASL Sign Language Recognition & Analytics

A Python application that recognizes static American Sign Language (ASL)
alphabet letters from a photo/webcam snapshot using **MediaPipe hand
tracking** + a transparent **rule-based geometric classifier**, logs every
prediction to a local **SQLite** database, and visualizes the results on a
dark, modern **analytics dashboard** built with Streamlit, pandas and
matplotlib.

## ✨ Features

- 📷 **Live recognition** — take a camera snapshot or upload a photo
- ✋ **21-point hand landmark detection** (MediaPipe Hands)
- 🔤 Recognizes static ASL letters: `A B C D E F I L O R S U V W Y`
- 📊 **Analytics dashboard** — detections per letter, confidence
  distribution, detections over time, per-session summary
- 🗂️ **History & CSV export**, clear-history option
- 🎨 Polished dark UI with custom styling
- 🛡️ Defensive error handling throughout — a bad frame or missing hand
  never crashes the app, it just shows a friendly message

## 📁 Project structure

```
sign_language_app/
├── app.py                  # Streamlit UI (the only file that imports streamlit)
├── core/
│   ├── hand_tracker.py     # MediaPipe wrapper -> 21 landmarks per hand
│   ├── classifier.py       # Rule-based landmark -> ASL letter classifier
│   ├── database.py         # SQLite logging (zero extra dependencies)
│   └── analytics.py        # Summary stats + matplotlib charts
├── tests/
│   └── test_core.py        # Offline smoke tests (no camera/network needed)
├── data/                   # Created automatically; holds recognition_log.db
├── requirements.txt
└── README.md
```

## 🚀 Setup

Requires **Python 3.9–3.12**.

```bash
cd sign_language_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Grant camera permission when prompted if you want to use live snapshots
— uploading an image works too, with no camera required.

## ✅ Run the offline tests (optional but recommended)

```bash
pip install pytest
python -m pytest tests/ -v
```

These tests exercise the classifier, database, and analytics modules
directly (no Streamlit, no camera) and should always pass on a clean
install.

## 🧠 How recognition works

1. **`core/hand_tracker.py`** runs MediaPipe Hands on the input image and
   returns 21 normalized `(x, y, z)` landmarks per detected hand.
2. **`core/classifier.py`** computes simple, human-readable geometric
   features from those landmarks — which fingers are extended, thumb
   position, fingertip distances — and matches them against known ASL
   hand shapes. This is fully transparent (no training data or internet
   access required) and returns a confidence score with each guess.
3. **`core/database.py`** logs every prediction (letter, confidence,
   timestamp, session) to a local SQLite file at `data/recognition_log.db`.
4. **`core/analytics.py`** turns that log into the dashboard charts using
   pandas + matplotlib.

### Why only some letters?

`J` and `Z` require **motion** (tracing a shape in the air) and a few
other letters depend on subtle finger crossings that are unreliable to
detect from a single static photo. Rather than guess unreliably, this
build focuses on the 15 letters that have a stable, purely static hand
shape: `A B C D E F I L O R S U V W Y`. Swap in a trained ML model in
`core/classifier.py` if you want full 26-letter + motion (J/Z) coverage.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` inside your active virtual environment |
| Camera not detected | Use the **Upload image** option in the sidebar instead |
| Low accuracy | Ensure good lighting and that your whole hand is visible against a plain background |
| App won't start | Confirm Python 3.9+ with `python --version`, and that you're in the project's virtual environment |

## 📄 License

Provided as-is for learning and prototyping purposes.
