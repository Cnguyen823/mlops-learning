from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Define model
clf = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))
