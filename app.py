from flask import Flask, render_template, request, session, jsonify
import numpy as np
import pickle
import os
import json
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Load model artifacts ──
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

model_accuracy  = metrics['accuracy']
model_precision = metrics['precision']
model_recall    = metrics.get('recall', 0)
model_f1        = metrics.get('f1_score', 0)
model_auc       = metrics.get('auc_roc', 0)

FEATURES = metrics.get('features', [
    'mean_hold_time', 'std_hold_time', 'mean_flight_time', 'std_flight_time',
    'hold_to_flight_ratio', 'iki_coefficient_variation',
    'rhythm_consistency_score', 'backspace_rate', 'age', 'gender'
])

TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank"


def compute_features(hold_times, flight_times, typed_text, age, gender):
    """
    Compute all 10 features from raw keystroke data.
    All values grounded in published research ranges.
    """
    hold  = np.array([h for h in hold_times  if 20 <= h <= 800], dtype=float)
    flight = np.array([f for f in flight_times if 20 <= f <= 500], dtype=float)

    # Guard: need minimum keystrokes
    if len(hold) < 8:
        return None, "Not enough keystrokes recorded. Please type the full sentence."

    mean_hold   = float(np.mean(hold))
    std_hold    = float(np.std(hold))
    mean_flight = float(np.mean(flight)) if len(flight) >= 2 else mean_hold * 1.6
    std_flight  = float(np.std(flight))  if len(flight) >= 2 else 0.0

    hold_to_flight_ratio = mean_hold / mean_flight if mean_flight > 0 else 0.5

    # IKI coefficient of variation: std_flight / mean_flight
    # High = arrhythmic (PD pattern)
    iki_cv = std_flight / mean_flight if mean_flight > 0 else 0.3

    # Rhythm consistency: inverse of normalized variance
    # 1.0 = perfectly consistent, 0.0 = erratic
    if len(flight) >= 4:
        cv = np.std(flight) / np.mean(flight)
        rhythm_score = float(np.clip(1.0 - cv, 0.1, 1.0))
    else:
        rhythm_score = 0.7

    # Backspace rate: backspaces / total keystrokes
    typed_len   = len(typed_text)
    n_backspace = max(0, len(hold_times) - typed_len)
    backspace_rate = min(n_backspace / max(len(hold_times), 1), 0.4)

    # Age and gender (passed from patient form)
    age_val    = float(np.clip(age, 18, 100))
    gender_val = 1.0 if str(gender).upper().startswith('M') else 0.0

    features = {
        'mean_hold_time':           round(mean_hold, 3),
        'std_hold_time':            round(std_hold, 3),
        'mean_flight_time':         round(mean_flight, 3),
        'std_flight_time':          round(std_flight, 3),
        'hold_to_flight_ratio':     round(hold_to_flight_ratio, 4),
        'iki_coefficient_variation':round(iki_cv, 4),
        'rhythm_consistency_score': round(rhythm_score, 4),
        'backspace_rate':           round(backspace_rate, 4),
        'age':                      age_val,
        'gender':                   gender_val
    }
    return features, None


def compute_typing_accuracy(typed_text):
    target_words = TARGET_SENTENCE.lower().strip().split()
    typed_words  = typed_text.lower().strip().split()
    correct = sum(1 for a, b in zip(target_words, typed_words) if a == b)
    return round((correct / len(target_words)) * 100, 1) if target_words else 0


@app.route('/')
def home():
    return render_template('index.html',
        prediction=None,
        accuracy=model_accuracy,
        precision=model_precision,
        target_sentence=TARGET_SENTENCE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── Parse inputs ──
        hold_times   = json.loads(request.form.get('hold_times',   '[]'))
        flight_times = json.loads(request.form.get('flight_times', '[]'))
        typed_text   = request.form.get('typed_text', '')

        patient_name   = request.form.get('patient_name',   'Patient')
        patient_age    = int(request.form.get('patient_age',   60))
        patient_gender = request.form.get('patient_gender', 'Unknown')
        patient_mobile = request.form.get('patient_mobile', '—')

        print(f"\nDEBUG ── Patient: {patient_name}, Age: {patient_age}, Gender: {patient_gender}")
        print(f"DEBUG ── hold_times: {len(hold_times)} samples, flight_times: {len(flight_times)} samples")
        print(f"DEBUG ── typed_text: {typed_text[:60]}")

        # ── Compute features ──
        features, error = compute_features(
            hold_times, flight_times, typed_text, patient_age, patient_gender
        )

        if error:
            return render_template('index.html',
                prediction=f"Error: {error}",
                confidence=0, accuracy=model_accuracy, precision=model_precision,
                target_sentence=TARGET_SENTENCE,
                mean_hold=0, std_hold=0, std_flight=0,
                typing_accuracy=0, key_count=0)

        typing_accuracy = compute_typing_accuracy(typed_text)
        key_count       = len(hold_times)

        print(f"DEBUG ── Features: {features}")

        # ── Predict ──
        input_array  = np.array([[features[f] for f in FEATURES]])
        scaled_input = scaler.transform(input_array)

        prediction_val = model.predict(scaled_input)[0]
        probability    = model.predict_proba(scaled_input)[0]
        confidence     = round(float(max(probability)) * 100, 1)

        result = "Likely Parkinson's" if prediction_val == 1 else "Unlikely Parkinson's"

        print(f"DEBUG ── Prediction: {result}, Confidence: {confidence}%")
        print(f"DEBUG ── Typing accuracy: {typing_accuracy}%")

        # ── Store in session for report ──
        session['report_data'] = {
            'patient_name':   patient_name,
            'patient_age':    patient_age,
            'patient_gender': patient_gender,
            'patient_mobile': patient_mobile,
            'prediction':     result,
            'confidence':     confidence,
            'model_accuracy': float(model_accuracy),
            'model_precision':float(model_precision),
            'model_recall':   float(model_recall),
            'model_f1':       float(model_f1),
            'model_auc':      float(model_auc),
            'mean_hold':      features['mean_hold_time'],
            'std_hold':       features['std_hold_time'],
            'mean_flight':    features['mean_flight_time'],
            'std_flight':     features['std_flight_time'],
            'iki_cv':         features['iki_coefficient_variation'],
            'rhythm_score':   features['rhythm_consistency_score'],
            'backspace_rate': features['backspace_rate'],
            'key_count':      key_count,
            'typing_accuracy':typing_accuracy
        }

        return render_template('index.html',
            prediction=result,
            confidence=confidence,
            accuracy=model_accuracy,
            precision=model_precision,
            target_sentence=TARGET_SENTENCE,
            mean_hold=features['mean_hold_time'],
            std_hold=features['std_hold_time'],
            std_flight=features['std_flight_time'],
            typing_accuracy=typing_accuracy,
            key_count=key_count)

    except Exception as e:
        import traceback
        print(f"DEBUG ── ERROR: {e}")
        traceback.print_exc()
        return render_template('index.html',
            prediction=f"Error: {str(e)}",
            confidence=0, accuracy=model_accuracy, precision=model_precision,
            target_sentence=TARGET_SENTENCE,
            mean_hold=0, std_hold=0, std_flight=0,
            typing_accuracy=0, key_count=0)


@app.route('/report')
def report():
    data = session.get('report_data')
    if not data:
        return "No report data. Please complete the test first.", 400

    is_positive = data['prediction'].lower().startswith('likely')

    return render_template('report.html',
        patient_name   = data['patient_name'],
        patient_age    = data['patient_age'],
        patient_gender = data['patient_gender'],
        patient_mobile = data['patient_mobile'],
        report_date    = datetime.now().strftime('%d %B %Y'),
        report_id      = 'NK-' + str(uuid.uuid4())[:8].upper(),
        is_positive    = is_positive,
        prediction     = data['prediction'],
        confidence     = data['confidence'],
        model_accuracy = data['model_accuracy'],
        model_precision= data['model_precision'],
        model_recall   = data['model_recall'],
        model_f1       = data['model_f1'],
        model_auc      = data['model_auc'],
        mean_hold      = data['mean_hold'],
        std_hold       = data['std_hold'],
        mean_flight    = data['mean_flight'],
        std_flight     = data['std_flight'],
        iki_cv         = data['iki_cv'],
        rhythm_score   = data['rhythm_score'],
        backspace_rate = data['backspace_rate'],
        key_count      = data['key_count'],
        typing_accuracy= data['typing_accuracy']
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON endpoint for testing via curl/Postman"""
    try:
        data = request.get_json()
        features_input = np.array([[
            data['mean_hold_time'], data['std_hold_time'],
            data['mean_flight_time'], data['std_flight_time'],
            data['hold_to_flight_ratio'], data['iki_coefficient_variation'],
            data['rhythm_consistency_score'], data['backspace_rate'],
            data['age'], data['gender']
        ]])
        scaled = scaler.transform(features_input)
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0]
        return jsonify({
            'prediction': "Likely Parkinson's" if pred == 1 else "Unlikely Parkinson's",
            'confidence': round(float(max(prob)) * 100, 1),
            'model_accuracy': round(model_accuracy * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)