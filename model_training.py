import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score

# Load data
df = pd.read_csv("processed_data.csv")
X = df[['mean_hold_time', 'std_hold_time', 'std_flight_time']]  # PSO selected features
y = df['parkinson_prediction']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensemble - Voting Classifier
model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
], voting='soft')

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Voting Classifier Accuracy: {accuracy:.3f}")
print(f"Voting Classifier Precision: {precision:.3f}")
print(classification_report(y_test, y_pred))

# Save model & scaler
with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metrics
with open('model/metrics.pkl', 'wb') as f:
    pickle.dump({'accuracy': accuracy, 'precision': precision}, f)
