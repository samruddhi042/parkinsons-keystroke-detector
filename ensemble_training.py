import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load your cleaned dataset
df = pd.read_csv('processed_data.csv')

# Prepare features and labels
X = df.drop(['parkinson_prediction', 'file'], axis=1)
y = df['parkinson_prediction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base models
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)
svm = SVC(kernel='linear', probability=True)

# -----------------------
# 1️⃣ Voting Classifier
# -----------------------
voting = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('svm', svm)
], voting='soft')

voting.fit(X_train_scaled, y_train)
voting_preds = voting.predict(X_test_scaled)

print("🔸 Voting Classifier Results:")
print(classification_report(y_test, voting_preds))
print("Accuracy:", accuracy_score(y_test, voting_preds))

# -----------------------
# 2️⃣ Stacking Classifier
# -----------------------
stacking = StackingClassifier(
    estimators=[('rf', rf), ('svm', svm)],
    final_estimator=LogisticRegression(),
    passthrough=True
)

stacking.fit(X_train_scaled, y_train)
stacking_preds = stacking.predict(X_test_scaled)

print("\n🔸 Stacking Classifier Results:")
print(classification_report(y_test, stacking_preds))
print("Accuracy:", accuracy_score(y_test, stacking_preds))

# Save best performing model (replace with 'voting' or 'stacking' based on performance)
# Save best performing model
best_model = stacking  # <-- Update here

with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

