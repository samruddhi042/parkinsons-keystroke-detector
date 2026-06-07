"""
NeuroKey Dataset Generator + Model Trainer
Based on published research:
- Giancardo et al. (Nature Scientific Reports, 2016) - neuroQWERTY
- Diagnostic accuracy of keystroke dynamics (Nature, 2022)
- Imbalanced ensemble learning in PD using KD (ScienceDirect, 2023)

Key research findings used:
- Healthy hold time: ~100ms, range 60-160ms
- PD hold time: elevated and MORE VARIABLE (higher std), ~130-250ms
- Healthy flight time std: low variability ~40-80ms
- PD flight time: longer inter-key delay, higher variance ~100-400ms
- PD characterized by arrhythmokinesia: irregular timing rhythms
- Age affects both groups: older = slightly slower but PD effect is VARIANCE not just mean
- Gender: minimal effect on hold time (research confirms HT largely gender-independent)
- Hold-to-flight ratio: PD patients show disrupted ratio patterns
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score,
                              precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 1: GENERATE REALISTIC SYNTHETIC DATASET
# ─────────────────────────────────────────────

def generate_healthy_subject(age, gender):
    """
    Generate keystroke features for a healthy control.
    Based on neuroQWERTY research: healthy HT ~100ms, low variance.
    Age effect: ~0.3ms increase per year over 30.
    """
    age_factor = (age - 30) * 0.3  # slight slowdown with age

    # Mean hold time: 80-140ms for healthy, slightly higher with age
    mean_hold = np.random.normal(105 + age_factor * 0.4, 12)
    mean_hold = np.clip(mean_hold, 65, 175)

    # Std hold time: LOW for healthy (consistent pressing) ~20-55ms
    # This is the KEY discriminator per research
    std_hold = np.random.normal(35 + age_factor * 0.1, 8)
    std_hold = np.clip(std_hold, 12, 65)

    # Mean flight time: ~150-250ms for healthy
    mean_flight = np.random.normal(190 + age_factor * 0.5, 25)
    mean_flight = np.clip(mean_flight, 100, 320)

    # Std flight time: LOW for healthy (rhythmic typing) ~40-80ms
    std_flight = np.random.normal(58 + age_factor * 0.2, 12)
    std_flight = np.clip(std_flight, 25, 95)

    # Hold-to-flight ratio: consistent for healthy ~0.4-0.7
    hold_to_flight = mean_hold / mean_flight
    hold_to_flight = np.clip(hold_to_flight + np.random.normal(0, 0.03), 0.3, 0.75)

    # IKI (Inter-keystroke interval) coefficient of variation: low for healthy
    iki_cv = std_flight / mean_flight
    iki_cv = np.clip(iki_cv + np.random.normal(0, 0.02), 0.15, 0.45)

    # Rhythm consistency score (1 = perfectly consistent, 0 = erratic)
    rhythm_score = np.random.normal(0.78, 0.08)
    rhythm_score = np.clip(rhythm_score, 0.55, 0.95)

    # Backspace rate: healthy typists correct errors normally
    backspace_rate = np.random.normal(0.04, 0.02)
    backspace_rate = np.clip(backspace_rate, 0, 0.12)

    return {
        'mean_hold_time': round(mean_hold, 3),
        'std_hold_time': round(std_hold, 3),
        'mean_flight_time': round(mean_flight, 3),
        'std_flight_time': round(std_flight, 3),
        'hold_to_flight_ratio': round(hold_to_flight, 4),
        'iki_coefficient_variation': round(iki_cv, 4),
        'rhythm_consistency_score': round(rhythm_score, 4),
        'backspace_rate': round(backspace_rate, 4),
        'age': age,
        'gender': 1 if gender == 'M' else 0,
        'label': 0  # healthy
    }


def generate_pd_subject(age, gender, severity='early'):
    """
    Generate keystroke features for a PD patient.
    Key PD characteristics from research:
    - Bradykinesia: slower key release -> HIGHER mean hold time
    - Arrhythmokinesia: irregular timing -> MUCH HIGHER std (variance)
    - Longer inter-key delays (flight time)
    - Disrupted hold-to-flight ratio
    - Early PD: mild changes; moderate: clear changes
    """
    age_factor = (age - 30) * 0.3

    if severity == 'early':
        hold_elevation = np.random.uniform(15, 45)
        std_elevation = np.random.uniform(25, 70)
        flight_elevation = np.random.uniform(20, 80)
        std_flight_elevation = np.random.uniform(40, 120)
        rhythm_drop = np.random.uniform(0.08, 0.18)
    else:  # moderate
        hold_elevation = np.random.uniform(45, 120)
        std_elevation = np.random.uniform(70, 180)
        flight_elevation = np.random.uniform(80, 200)
        std_flight_elevation = np.random.uniform(120, 280)
        rhythm_drop = np.random.uniform(0.18, 0.35)

    # Mean hold time: elevated due to bradykinesia
    mean_hold = np.random.normal(115 + age_factor * 0.4 + hold_elevation, 18)
    mean_hold = np.clip(mean_hold, 85, 380)

    # Std hold time: KEY DISCRIMINATOR - much higher variance in PD
    # PD patients have inconsistent key press durations
    std_hold = np.random.normal(55 + age_factor * 0.15 + std_elevation, 20)
    std_hold = np.clip(std_hold, 35, 320)

    # Mean flight time: longer inter-key delay
    mean_flight = np.random.normal(210 + age_factor * 0.5 + flight_elevation, 40)
    mean_flight = np.clip(mean_flight, 130, 600)

    # Std flight time: HIGH variance (arrhythmokinesia - hastening/freezing)
    std_flight = np.random.normal(90 + age_factor * 0.3 + std_flight_elevation, 30)
    std_flight = np.clip(std_flight, 55, 480)

    # Hold-to-flight ratio: disrupted
    hold_to_flight = mean_hold / mean_flight
    hold_to_flight = np.clip(hold_to_flight + np.random.normal(0, 0.07), 0.25, 0.95)

    # IKI coefficient of variation: HIGH for PD
    iki_cv = std_flight / mean_flight
    iki_cv = np.clip(iki_cv + np.random.normal(0, 0.05), 0.3, 0.9)

    # Rhythm consistency: LOW for PD
    rhythm_score = np.random.normal(0.55, 0.10)
    rhythm_score = np.clip(rhythm_score, 0.20, 0.75)

    # Backspace rate: PD patients make more errors
    backspace_rate = np.random.normal(0.09, 0.04)
    backspace_rate = np.clip(backspace_rate, 0.01, 0.25)

    return {
        'mean_hold_time': round(mean_hold, 3),
        'std_hold_time': round(std_hold, 3),
        'mean_flight_time': round(mean_flight, 3),
        'std_flight_time': round(std_flight, 3),
        'hold_to_flight_ratio': round(hold_to_flight, 4),
        'iki_coefficient_variation': round(iki_cv, 4),
        'rhythm_consistency_score': round(rhythm_score, 4),
        'backspace_rate': round(backspace_rate, 4),
        'age': age,
        'gender': 1 if gender == 'M' else 0,
        'label': 1  # PD
    }


# Generate dataset
print("Generating research-backed synthetic dataset...")
records = []

# Healthy controls - age range 30-80, realistic distribution
# More subjects in 50-75 range (PD screening demographic)
healthy_ages = (
    list(np.random.randint(30, 50, 150)) +  # younger healthy
    list(np.random.randint(50, 65, 250)) +  # middle age
    list(np.random.randint(65, 80, 200))    # older healthy
)

for age in healthy_ages:
    gender = np.random.choice(['M', 'F'], p=[0.5, 0.5])
    records.append(generate_healthy_subject(age, gender))

# PD patients - mostly 50-80 (PD onset typically 60+)
pd_ages_early = (
    list(np.random.randint(45, 60, 150)) +   # early onset
    list(np.random.randint(60, 75, 200)) +   # typical onset
    list(np.random.randint(75, 82, 100))     # late
)

for age in pd_ages_early:
    gender = np.random.choice(['M', 'F'], p=[0.58, 0.42])  # PD slightly more common in men
    severity = 'early' if age < 68 else 'moderate'
    records.append(generate_pd_subject(age, gender, severity))

df = pd.DataFrame(records)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nFeature statistics:")
print(df.drop(['label','gender'], axis=1).describe().round(2))

# Save dataset
df.to_csv('/home/claude/parkinsons_dataset.csv', index=False)
print("\nDataset saved.")

# ─────────────────────────────────────────────
# STEP 2: TRAIN THE MODEL
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# Features - 10 rich features
FEATURES = [
    'mean_hold_time',
    'std_hold_time',
    'mean_flight_time',
    'std_flight_time',
    'hold_to_flight_ratio',
    'iki_coefficient_variation',
    'rhythm_consistency_score',
    'backspace_rate',
    'age',
    'gender'
]

X = df[FEATURES]
y = df['label']

# Train/test split - stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train class distribution: {dict(y_train.value_counts())}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensemble: Random Forest + Gradient Boosting + Logistic Regression
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)

lr = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft',
    weights=[3, 3, 1]  # RF and GB weighted higher
)

ensemble.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# STEP 3: EVALUATE
# ─────────────────────────────────────────────

y_pred = ensemble.predict(X_test_scaled)
y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*40}")
print("MODEL EVALUATION RESULTS")
print(f"{'='*40}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negative  (Healthy→Healthy): {cm[0][0]}")
print(f"  False Positive (Healthy→PD):      {cm[0][1]}")
print(f"  False Negative (PD→Healthy):      {cm[1][0]}")
print(f"  True Positive  (PD→PD):           {cm[1][1]}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))

# 5-Fold Cross Validation
print("Running 5-Fold Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all_scaled = scaler.transform(X)
cv_scores = cross_val_score(ensemble, X_all_scaled, y, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(ensemble, X_all_scaled, y, cv=cv, scoring='roc_auc')

print(f"\nCross-Val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Cross-Val AUC-ROC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Feature importance from RF
rf_trained = ensemble.estimators_[0]
importances = dict(zip(FEATURES, rf_trained.feature_importances_))
print(f"\nFeature Importances (Random Forest):")
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<35} {imp:.4f} ({imp*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 4: SAVE
# ─────────────────────────────────────────────

os.makedirs('/home/claude/model_new', exist_ok=True)

with open('/home/claude/model_new/best_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

with open('/home/claude/model_new/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc,
    'cv_accuracy_mean': float(cv_scores.mean()),
    'cv_accuracy_std': float(cv_scores.std()),
    'cv_auc_mean': float(cv_auc.mean()),
    'features': FEATURES
}

with open('/home/claude/model_new/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

import json
with open('/home/claude/model_new/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*40}")
print("All files saved to model_new/")
print(f"  best_model.pkl")
print(f"  scaler.pkl")
print(f"  metrics.pkl")
print(f"  metrics.json")
print(f"\nDataset saved: parkinsons_dataset.csv")
print(f"{'='*40}")