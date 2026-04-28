# EDA for Parkinson’s Detection via Keystroke Dynamics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('keystroke_features_and_prediction.csv')  # 🔁 Replace with your actual filename
print("🔹 First 5 rows:\n", df.head())
print("\n🔹 Dataset Info:")
print(df.info())
print("\n🔹 Statistical Summary:")
print(df.describe())

# Check for nulls
print("\n🔹 Missing values:\n", df.isnull().sum())

# Check for duplicates
print(f"\n🔹 Duplicate rows: {df.duplicated().sum()}")

# Class distribution
plt.figure(figsize=(5,4))
sns.countplot(x='parkinson_prediction', data=df)
plt.title("🧠 Parkinson's Class Distribution")
plt.xlabel("Parkinson Prediction (0=Healthy, 1=Parkinson's)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Pair Plot to compare feature distributions by class
sns.pairplot(df, hue='parkinson_prediction',
             vars=['mean_hold_time', 'std_hold_time', 'mean_flight_time',
                   'std_flight_time', 'hold_to_flight_ratio'])
plt.suptitle("Pairplot by Parkinson Prediction", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.drop(columns=['file']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("🔗 Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplots to check outliers
features = ['mean_hold_time', 'std_hold_time', 'mean_flight_time',
            'std_flight_time', 'hold_to_flight_ratio']

for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='parkinson_prediction', y=feature, data=df)
    plt.title(f"📦 Boxplot of {feature} by Class")
    plt.tight_layout()
    plt.show()

# Distribution plots
for feature in features:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df, x=feature, hue='parkinson_prediction', kde=True)
    plt.title(f"📊 Distribution of {feature} by Class")
    plt.tight_layout()
    plt.show()

# Skewness and Kurtosis
print("\n🔹 Skewness:")
print(df[features].skew())
print("\n🔹 Kurtosis:")
print(df[features].kurt())
