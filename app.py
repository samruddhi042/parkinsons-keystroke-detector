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
        mean_hold = float(request.form['mean_hold_time'])
        std_hold = float(request.form['std_hold_time'])
        std_flight = float(request.form['std_flight_time'])

        # Validation
        if mean_hold <= 0 or std_hold < 0 or std_flight < 0:
            return render_template('index.html', prediction="Error: Values must be positive.",
                confidence=0, accuracy=model_accuracy, precision=model_precision,
                mean_hold=mean_hold, std_hold=std_hold, std_flight=std_flight)

        input_array = np.array([[mean_hold, std_hold, std_flight]])
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        confidence = round(float(max(probability)) * 100, 2)

        result = "Likely Parkinson's" if prediction == 1 else "Unlikely Parkinson's"

        return render_template('index.html',
            prediction=result,
            confidence=confidence,
            accuracy=model_accuracy,
            precision=model_precision,
            mean_hold=mean_hold,
            std_hold=std_hold,
            std_flight=std_flight
        )
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}",
            confidence=0, accuracy=model_accuracy, precision=model_precision,
            mean_hold=0, std_hold=0, std_flight=0)

if __name__ == '__main__':
    app.run(debug=True)
