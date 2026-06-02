from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model, scaler, metrics
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

model_accuracy = metrics['accuracy']
model_precision = metrics['precision']

# The fixed sentence the user must type
TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank"

def compute_features(hold_times, flight_times):
    """Compute mean hold, std hold, std flight from raw keystroke arrays"""
    hold = np.array(hold_times)
    flight = np.array(flight_times)

    mean_hold = float(np.mean(hold)) if len(hold) > 0 else 0
    std_hold = float(np.std(hold)) if len(hold) > 1 else 0
    std_flight = float(np.std(flight)) if len(flight) > 1 else 0

    return mean_hold, std_hold, std_flight

def compute_typing_accuracy(typed_text):
    """Compare typed text against target sentence"""
    target = TARGET_SENTENCE.lower().strip()
    typed = typed_text.lower().strip()

    target_words = target.split()
    typed_words = typed.split()

    correct = sum(1 for a, b in zip(target_words, typed_words) if a == b)
    total = len(target_words)
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0
    return accuracy

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
        # Get raw keystroke arrays from frontend
        import json

        hold_times = json.loads(request.form.get('hold_times', '[]'))
        flight_times = json.loads(request.form.get('flight_times', '[]'))
        typed_text = request.form.get('typed_text', '')

        print(f"DEBUG — hold_times count: {len(hold_times)}, flight_times count: {len(flight_times)}")
        print(f"DEBUG — typed_text: {typed_text[:50]}")

        # Validate enough keystrokes
        if len(hold_times) < 10:
            return render_template('index.html',
                prediction="Error: Not enough keystrokes recorded. Please type the full sentence.",
                confidence=0,
                accuracy=model_accuracy,
                precision=model_precision,
                target_sentence=TARGET_SENTENCE,
                mean_hold=0, std_hold=0, std_flight=0,
                typing_accuracy=0, key_count=0)

        # Compute features from raw data
        mean_hold, std_hold, std_flight = compute_features(hold_times, flight_times)
        typing_accuracy = compute_typing_accuracy(typed_text)
        key_count = len(hold_times)

        print(f"DEBUG — mean_hold={mean_hold:.1f}, std_hold={std_hold:.1f}, std_flight={std_flight:.1f}")
        print(f"DEBUG — typing_accuracy={typing_accuracy}%")

        # Scale and predict
        input_array = np.array([[mean_hold, std_hold, std_flight]])
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        confidence = round(float(max(probability)) * 100, 2)

        print(f"DEBUG — prediction={prediction}, confidence={confidence}%")

        result = "Likely Parkinson's" if prediction == 1 else "Unlikely Parkinson's"

        return render_template('index.html',
            prediction=result,
            confidence=confidence,
            accuracy=model_accuracy,
            precision=model_precision,
            target_sentence=TARGET_SENTENCE,
            mean_hold=round(mean_hold, 2),
            std_hold=round(std_hold, 2),
            std_flight=round(std_flight, 2),
            typing_accuracy=typing_accuracy,
            key_count=key_count)

    except Exception as e:
        print(f"DEBUG — error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
            prediction=f"Error: {str(e)}",
            confidence=0,
            accuracy=model_accuracy,
            precision=model_precision,
            target_sentence=TARGET_SENTENCE,
            mean_hold=0, std_hold=0, std_flight=0,
            typing_accuracy=0, key_count=0)

if __name__ == '__main__':
    app.run(debug=True)