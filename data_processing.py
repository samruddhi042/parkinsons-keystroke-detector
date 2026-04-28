# ------------------------------------
# 1. Import Libraries
# ------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------
# 2. Load Dataset
# ------------------------------------
df = pd.read_csv('keystroke_features_and_prediction.csv')  # change file name if needed
print(df.columns)
# ------------------------------------
# 3. Basic Cleaning (optional tweaks based on EDA)
# ------------------------------------
# Replace any obvious outliers or fix negatives if needed (example only)
# df['mean_hold_time'] = df['mean_hold_time'].apply(lambda x: abs(x))  # remove negatives
# You can also clip:
# df['mean_hold_time'] = df['mean_hold_time'].clip(lower=0)

# Drop any rows with NaNs (optional, or use imputation)
df.dropna(inplace=True)

# ------------------------------------
# 4. Feature-Target Split
# ------------------------------------
X = df.drop(['parkinson_prediction', 'file'], axis=1)   # all features
y = df['parkinson_prediction']                # target variable (0/1)

# ------------------------------------
# 5. Train-Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------
# 6. Feature Scaling
# ------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------
# 7. Convert Back to DataFrames (Optional, useful for analysis)
# ------------------------------------
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# ------------------------------------
# 8. Quick Sanity Check
# ------------------------------------
print("Training shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("Target distribution:\n", y.value_counts())

# Plot distribution of scaled features (optional)
# sns.histplot(X_train_scaled['mean_hold_time'], kde=True)
# plt.title("Scaled mean_hold_time Distribution")
# plt.show()
# Save the processed data to CSV
df.to_csv("processed_data.csv", index=False)
