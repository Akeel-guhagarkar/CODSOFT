import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset (limit rows for speed)
train_data = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["genre", "plot"],
    nrows=8000
)

# Split data
X = train_data["plot"]
y = train_data["genre"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF (fast)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    ngram_range=(1, 1)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Logistic Regression (MULTICLASS + FAST)
model = LogisticRegression(
    max_iter=300,
    solver="saga"
)

model.fit(X_train_vec, y_train)

# Accuracy
y_val_pred = model.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", accuracy)

# Custom prediction
sample_plot = "A police officer fights dangerous criminals and saves the city from chaos"
sample_vec = vectorizer.transform([sample_plot])
prediction = model.predict(sample_vec)

print("Predicted Genre:", prediction[0])
