import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('keystroke_features_and_prediction.csv')

# Preview data
print(df.head())
print(df.columns)

# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values if needed
# df.dropna(inplace=True)
# df.fillna(df.mean(), inplace=True)

# Check data types
print(df.dtypes)

# Convert prediction column to integer
df['parkinson_prediction'] = df['parkinson_prediction'].astype('int')
