import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pyswarms as ps

# Load your dataset
df = pd.read_csv("processed_data.csv")

# Prepare data
X = df.drop(['parkinson_prediction', 'file'], axis=1)
y = df['parkinson_prediction']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Define objective function for PSO
def fitness_function(particle):
    n_particles = particle.shape[0]
    scores = np.zeros(n_particles)

    for i in range(n_particles):
        mask = particle[i] > 0.5  # binary mask
        if np.sum(mask) == 0:
            scores[i] = 0
        else:
            clf = RandomForestClassifier()
            clf.fit(X_train[:, mask], y_train)
            preds = clf.predict(X_test[:, mask])
            scores[i] = accuracy_score(y_test, preds)
    
    return 1 - scores  # since PSO minimizes

# PSO Setup
options = {
    'c1': 0.5,     # cognitive coefficient
    'c2': 0.3,     # social coefficient
    'w': 0.9,      # inertia weight
    'k': 5,        # number of neighbors
    'p': 2         # Minkowski p-norm
}

dimensions = X.shape[1]
optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)

# Perform optimization
best_cost, best_pos = optimizer.optimize(fitness_function, iters=30)
print("Best Features Mask:", best_pos)

# Train final model with best features
selected_features = X.columns[best_pos > 0.5]
print("Selected Features:", list(selected_features))
