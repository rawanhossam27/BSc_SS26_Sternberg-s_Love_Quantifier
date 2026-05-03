import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load
df = pd.read_csv("dataset/initial_dataset.csv")

print(df.head())
print(df.columns.tolist())
print(df.shape)

# Keep only needed columns
df = df[["text", "label"]].copy()

# Clean
df["text"] = df["text"].astype(str).str.strip().str.strip('"').str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Drop bad rows
df = df.dropna(subset=["text", "label"])
df = df[df["text"] != ""]
df = df[df["label"].isin(["passion", "intimacy", "commitment"])]

print("Dataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "pred_label": y_pred
})

wrong = results[results["true_label"] != results["pred_label"]]
print("\nMisclassified examples:")
print(wrong.to_string(index=False))

samples = [
    "I want you more than anything in this world",
    "I trust you with everything I am",
    "I will stay with you no matter what happens"
]

for s in samples:
    print(s, "->", model.predict([s])[0])