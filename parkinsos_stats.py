import pandas as pd

# Load the dataset
df = pd.read_csv('processed_data.csv')

# Filter only Parkinson's users
parkinsons_users = df[df['parkinson_prediction'] == 1]

# Show average values
print("\nParkinson’s user averages:\n")
print(parkinsons_users.describe())
