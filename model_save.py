import joblib

# Load the model
model = joblib.load("best_model.pkl")

# Use it to predict
sample_input = [[0.1, 0.02, 0.15, 0.01, 0.75]]  # Example features
prediction = model.predict(sample_input)

print("Prediction:", prediction)
