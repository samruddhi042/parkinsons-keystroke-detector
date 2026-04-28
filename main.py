import os
import pandas as pd
import numpy as np

# Feature extraction function
def extract_features_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    hold_times = []
    flight_times = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        try:
            hold_time = float(parts[4])
            flight_time = float(parts[7])
            hold_times.append(hold_time)
            flight_times.append(flight_time)
        except ValueError:
            continue

    if not hold_times or not flight_times:
        return None

    features = {
        'mean_hold_time': np.mean(hold_times),
        'std_hold_time': np.std(hold_times),
        'mean_flight_time': np.mean(flight_times),
        'std_flight_time': np.std(flight_times),
        'hold_to_flight_ratio': np.mean(hold_times) / np.mean(flight_times) if np.mean(flight_times) > 0 else 0,
    }

    # Classification logic (simple threshold based)
    if features['mean_hold_time'] > 150 or features['std_hold_time'] > 50:
        features['parkinson_prediction'] = 1  # Parkinson's detected
    else:
        features['parkinson_prediction'] = 0  # Healthy

    return features

# Path to folder containing keystroke data
data_folder = 'archived-Data'

# Process all files
data = []
for filename in os.listdir(data_folder):
    filepath = os.path.join(data_folder, filename)
    features = extract_features_from_file(filepath)
    if features:
        features['file'] = filename
        data.append(features)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('keystroke_features_and_prediction.csv', index=False)

print("✅ Feature extraction complete. Saved to 'keystroke_features_and_prediction.csv'")
