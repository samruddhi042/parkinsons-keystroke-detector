import pandas as pd

# Load your processed dataset
df = pd.read_csv('processed_data.csv')

# Filter normal users (no Parkinson's)
normal_df = df[df['parkinson_prediction'] == 0]

# Print descriptive stats
print("Normal user averages:\n")
print(normal_df.describe())
