# NeuroKey — AI-Powered Parkinson's Early Detection via Keystroke Dynamics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Accuracy-99.5%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/AUC--ROC-1.0-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

> A non-invasive, real-time Parkinson's disease screening tool that analyzes keystroke dynamics — the way you type — to detect early motor control irregularities associated with Parkinson's disease. Built with Flask, scikit-learn, and a research-backed 10-feature ML pipeline.

---

## The Problem It Solves

Parkinson's disease affects over **10 million people worldwide**. The average time between symptom onset and clinical diagnosis is **3–7 years** — because early motor symptoms are subtle and often dismissed. By the time a patient receives a formal diagnosis, significant neurological damage has already occurred.

Traditional screening requires:
- Clinical visits and specialized neurologists
- Expensive DaTscan dopamine imaging
- Multiple motor function tests over several sessions

**NeuroKey eliminates these barriers.** A person types one sentence on any keyboard. The system silently measures their keystroke patterns, computes 10 neurological biomarkers, and delivers a clinical-grade screening report in under 90 seconds — for free, from anywhere.

---

## Research Foundation

NeuroKey's feature set and normal value ranges are grounded in four peer-reviewed publications:

| Paper | Key Finding Used |
|---|---|
| Giancardo et al., *Nature Scientific Reports* (2016) — neuroQWERTY | Hold time variance is the strongest PD discriminator |
| Arroyo-Gallego et al., *Science Translational Medicine* (2017) | IKI coefficient of variation separates PD from healthy controls |
| Diagnostic accuracy of keystroke dynamics, *Nature* (2022) | Flight time std and rhythm score correlate with motor decline |
| Imbalanced ensemble learning in PD using keystroke data, *ScienceDirect* (2023) | Ensemble models outperform single classifiers for PD keystroke classification |

---

## Dataset

### Generation Approach

The dataset was synthetically generated from published research values using `generate_and_train.py`. It contains **1,050 subjects** (600 healthy controls, 450 PD patients) across age groups 30–82, with realistic age-stratified distributions.

### Why Synthetic?

The original archived keystroke data contained corrupted values (negative hold times, hold times exceeding 2.8 million ms) making it unsuitable for training. The synthetic dataset was built from scratch using published normal ranges, validated against research findings, and generates realistic inter-subject variability.

### Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Healthy (0) | 600 | 57.1% |
| Parkinson's (1) | 450 | 42.9% |

---

## Feature Engineering

The model uses **10 features** extracted from raw keystroke events. Each feature has a specific neurological basis:

### Feature 1 — `mean_hold_time` (ms)

**What it is:** The average duration a key is physically held down before release.

**Neurological basis:** Bradykinesia (slowness of movement) is one of the cardinal symptoms of Parkinson's disease. PD patients hold keys for longer durations due to impaired motor release mechanisms. Healthy range: 80–160ms. PD patients: 130–380ms.

---

### Feature 2 — `std_hold_time` (ms)

**What it is:** The standard deviation of hold times across all keystrokes.

**Neurological basis:** This is the **strongest single discriminator** in PD research. Healthy typists press keys with consistent duration (low variance). PD patients exhibit arrhythmokinesia — irregular, inconsistent motor timing — causing high variance in hold durations. Even a PD patient who types at normal speed will show elevated std_hold. Normal range: 20–55ms. PD: 55–320ms.

---

### Feature 3 — `mean_flight_time` (ms)

**What it is:** The average time between consecutive key releases (inter-keystroke interval).

**Neurological basis:** PD patients exhibit longer inter-key delays due to impaired motor initiation (akinesia). They struggle to transition quickly from one finger movement to the next. Normal range: 150–250ms. PD: 200–600ms.

---

### Feature 4 — `std_flight_time` (ms)

**What it is:** Standard deviation of flight times — rhythm variability.

**Neurological basis:** The **second strongest discriminator**. Healthy typists have rhythmic, consistent typing. PD patients alternate between freezing episodes (very long gaps) and hastening (very short gaps), creating high flight time variance. Normal: 40–80ms. PD: 100–480ms.

---

### Feature 5 — `hold_to_flight_ratio`

**What it is:** Ratio of mean hold time to mean flight time.

**Neurological basis:** In healthy typists this ratio is stable (0.4–0.7). PD disrupts the coordination between key press duration and inter-key timing, causing the ratio to drift outside normal bounds. Computed as:

```python
hold_to_flight_ratio = mean_hold / mean_flight
```

---

### Feature 6 — `iki_coefficient_variation` (IKI-CV)

**What it is:** Coefficient of variation of inter-keystroke intervals = `std_flight / mean_flight`.

**Neurological basis:** IKI-CV is a dimensionless rhythm irregularity measure used in the neuroQWERTY study. A value close to 0 means highly rhythmic typing. PD patients show IKI-CV values 2–3x higher than healthy controls. Used in published clinical research as a standalone diagnostic indicator.

```python
iki_cv = std_flight / mean_flight if mean_flight > 0 else 0.3
```

---

### Feature 7 — `rhythm_consistency_score`

**What it is:** Inverse of normalized variance. Score of 1.0 = perfectly consistent rhythm. Score of 0.0 = completely erratic.

**Neurological basis:** Captures the overall rhythmic regularity of typing. PD patients show freezing of gait which manifests as typing freezes — sudden long pauses followed by bursts. This score captures that pattern holistically.

```python
cv = np.std(flight) / np.mean(flight)
rhythm_score = float(np.clip(1.0 - cv, 0.1, 1.0))
```

Normal range: 0.65–0.95. PD range: 0.20–0.70.

---

### Feature 8 — `backspace_rate`

**What it is:** Number of backspace presses divided by total keypresses.

**Neurological basis:** PD patients make more typing errors due to impaired fine motor control and reduced finger dexterity. Higher backspace rates indicate degraded motor precision. Normal: < 4%. PD: > 8%.

---

### Feature 9 — `age`

**What it is:** Patient's age in years (provided in the patient form).

**Neurological basis:** Age is a confounding variable in typing speed research. A 70-year-old typing slowly is expected; the same speed in a 30-year-old is suspicious. The model was trained with age-stratified data so it correctly normalizes other features against age. Clipped to range 18–100.

---

### Feature 10 — `gender`

**What it is:** Binary encoded — Male = 1, Female = 0.

**Neurological basis:** Parkinson's disease is approximately 1.5x more common in men than women (58% male in clinical datasets). Gender is included as a weak prior that shifts the model's decision boundary slightly. Its feature importance is low (< 1%) but it contributes meaningful signal at population scale.

---

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `std_flight_time` | 30.7% |
| 2 | `std_hold_time` | 23.5% |
| 3 | `iki_coefficient_variation` | 18.2% |
| 4 | `rhythm_consistency_score` | 12.1% |
| 5 | `mean_flight_time` | 7.4% |
| 6 | `hold_to_flight_ratio` | 4.1% |
| 7 | `mean_hold_time` | 2.3% |
| 8 | `backspace_rate` | 1.4% |
| 9 | `age` | 0.2% |
| 10 | `gender` | < 0.1% |

---

## Data Pipeline

```
Browser (User Types)
        │
        ▼
JavaScript captures raw events
  ├── keydown timestamp per key
  ├── keyup timestamp per key
  ├── backspace count
  └── typed text string
        │
        ▼
Client-side filtering
  ├── Hold times: 20ms ≤ t ≤ 800ms
  └── Flight times: 20ms ≤ t ≤ 500ms
  (removes noise + inter-word pauses)
        │
        ▼
POST /predict
  Body: hold_times[], flight_times[],
        typed_text, patient_name,
        patient_age, patient_gender,
        patient_mobile, backspace_count
        │
        ▼
Flask app.py — compute_features()
  ├── mean_hold_time
  ├── std_hold_time
  ├── mean_flight_time
  ├── std_flight_time
  ├── hold_to_flight_ratio
  ├── iki_coefficient_variation
  ├── rhythm_consistency_score
  ├── backspace_rate
  ├── age (from form)
  └── gender (from form)
        │
        ▼
StandardScaler.transform(features)
        │
        ▼
VotingClassifier.predict_proba()
  ├── RandomForest (weight: 3)
  ├── GradientBoosting (weight: 3)
  └── LogisticRegression (weight: 1)
        │
        ▼
prediction + confidence score
        │
        ▼
Flask session stores result
        │
        ├──▶ render index.html (results page)
        └──▶ GET /report → render report.html (PDF)
```

---
## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 99.5% |
| Precision | 100.0% |
| Recall | 98.9% |
| F1 Score | 99.4% |
| AUC-ROC | 1.0 |
| Cross-Val Accuracy (5-fold) | 99.9% ± 0.1% |

---

## Project Structure

```
parkinsons-keystroke-detector/
│
├── app.py                          # Flask backend — routes, feature computation, prediction
├── generate_and_train.py           # Dataset generation + model training script
├── parkinsons_synthetic_dataset.csv # 1,050-subject research-backed dataset
│
├── model/
│   ├── best_model.pkl              # Trained VotingClassifier
│   ├── scaler.pkl                  # StandardScaler fitted on training data
│   ├── metrics.pkl                 # Model metrics + feature list
│   └── metrics.json                # Human-readable metrics
│
├── templates/
│   ├── index.html                  # Main SPA (landing, form, typing test, results)
│   └── report.html                 # Print-optimized medical report page
│
├── static/
│   └── style.css                   # Supplementary styles
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Backend API

### `POST /predict`

Accepts raw keystroke data, computes all 10 features server-side, returns prediction.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `hold_times` | JSON array | Raw hold durations in ms |
| `flight_times` | JSON array | Raw flight durations in ms |
| `typed_text` | string | What the user actually typed |
| `patient_name` | string | For report generation |
| `patient_age` | integer | Used as model feature |
| `patient_gender` | string | Male/Female/Other |
| `patient_mobile` | string | For report |
| `backspace_count` | integer | Number of backspace presses |

### `GET /report`

Reads prediction from Flask session, renders a clean print-optimized HTML report. User saves as PDF.

### `POST /api/predict`

JSON endpoint for testing via curl or Postman:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_hold_time": 180,
    "std_hold_time": 95,
    "mean_flight_time": 280,
    "std_flight_time": 180,
    "hold_to_flight_ratio": 0.64,
    "iki_coefficient_variation": 0.64,
    "rhythm_consistency_score": 0.45,
    "backspace_rate": 0.12,
    "age": 68,
    "gender": 1
  }'
```

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/samruddhi042/parkinsons-keystroke-detector.git
cd parkinsons-keystroke-detector

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Docker

```bash
# Build image
docker build -t neurokey .

# Run container
docker run -p 5000:5000 neurokey
```

### Retrain the Model

```bash
python generate_and_train.py
# Outputs: model/best_model.pkl, model/scaler.pkl, model/metrics.pkl
```

---

## How to Use

1. **Open the app** at `http://127.0.0.1:5000`
2. **Enter patient details** — name, age, gender, mobile
3. **Type the sentence** — naturally at your normal pace
4. **View results** — confidence score, keystroke vs normal range comparison, clinical recommendations
5. **Download the report** — opens a print-optimized page, → Save as PDF

---

## Gap Filled

| Existing Solutions | NeuroKey |
|---|---|
| Requires clinical visit | Works from any browser |
| Expensive hardware (wrist sensors, DaTscan) | Standard keyboard only |
| Specialist neurologist needed | Automated AI screening |
| Results in days/weeks | Results in 90 seconds |
| No personalized report | Full PDF report with patient data and doctor recommendations |
| No age/gender normalization | Model trained with age-stratified data |
| Single-feature analysis | 10-feature research-backed pipeline |

---

## Built By

**Samruddhi & Gargi** — B.Tech CSE (Data Science), VIT Pune, 2025
