import pickle
import numpy as np

# Load model and scaler
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example normal human-like input
sample_input = np.array([[103, 31, 234, 296, 0.49]])  # mean/std hold/flight, ratio

# Scale the input
scaled_input = scaler.transform(sample_input)

# Predict
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0]

# Output
print("Prediction:", "Likely Parkinson’s" if prediction == 1 else "Unlikely Parkinson’s")
print("Probability:", proba)
