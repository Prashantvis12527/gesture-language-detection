import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump((model, encoder), f)

print("✅ Model saved as gesture_model.pkl")