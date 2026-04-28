from flask import Flask, render_template, request
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

@app.route('/')
def home():
    return render_template('index.html', prediction=None, accuracy=model_accuracy, precision=model_precision)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['mean_hold_time']),
            float(request.form['std_hold_time']),
            float(request.form['std_flight_time']),
        ]
        input_array = np.array([features])
        scaled_input = scaler.transform(input_array)

        # Make prediction and get probability
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]  # probability of class 1 (Parkinson's)
        confidence = round(probability * 100, 2)

        if prediction == 1:
            result = f"Prediction: Likely Parkinson’s ({confidence}%)"
        else:
            result = f"Prediction: Unlikely Parkinson’s ({100 - confidence}%)"
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', prediction=result, accuracy=model_accuracy, precision=model_precision)

if __name__ == '__main__':
    app.run(debug=True)
