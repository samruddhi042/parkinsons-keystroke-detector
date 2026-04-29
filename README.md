# Parkinson's Keystroke Detector

A machine learning based web application that detects early signs of Parkinson's disease using keystroke dynamics analysis.

## Features
- Keystroke feature extraction (hold time, flight time, ratios)
- Ensemble ML model with PSO-based feature selection
- Flask REST backend with HTML frontend
- Real-time prediction with confidence score

## Stack
- **Backend**: Python, Flask, scikit-learn, NumPy, pandas
- **ML**: Ensemble classifier, PSO feature selection
- **Frontend**: HTML, CSS (to be upgraded)

## Setup
pip install flask scikit-learn numpy pandas
python app.py

## Dataset
Keystroke dynamics data from archived user sessions. Features extracted: mean/std hold time, mean/std flight time, hold-to-flight ratio.
