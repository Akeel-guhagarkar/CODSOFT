import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (limit rows for speed + memory)
data = pd.read_csv("fraudTrain.csv", nrows=100000)

# Select ONLY numeric columns (VERY IMPORTANT)
numeric_cols = [
    "amt", "lat", "long", "city_pop", "unix_time", "is_fraud"
]

data = data[numeric_cols]

# Features & target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

print("Model training completed.\n")

# Validation
y_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_val, y_pred))

# Sample prediction
sample = X_val.iloc[0:1]
prediction = model.predict(sample)

print("\nSample Transaction Prediction:")
print("Fraud" if prediction[0] == 1 else "Not Fraud")
