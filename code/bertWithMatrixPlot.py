from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 1

# =========================
# Paths
# =========================
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "300dataset.csv"
output_dir = Path(__file__).resolve().parent.parent / "models" / "bert_test_output"

# =========================
# Load and clean data
# =========================
df = pd.read_csv(csv_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.loc[:, ["text", "label"]].copy()
df["text"] = df["text"].astype(str).str.strip().str.strip('"').str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

df = df.dropna(subset=["text", "label"])
df = df[df["text"] != ""]
df = df[df["label"].isin(["passion", "intimacy", "commitment"])]

print("Clean shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

# =========================
# Label mapping
# =========================
label2id = {
    "commitment": 0,
    "intimacy": 1,
    "passion": 2,
}
id2label = {v: k for k, v in label2id.items()}

df["label_id"] = df["label"].map(label2id)

# =========================
# Train / test split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label_id"],
)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================
# Convert to Hugging Face Dataset
# =========================
train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
    preserve_index=False,
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
    preserve_index=False,
)

# =========================
# Tokenizer and model
# =========================
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir=str(output_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Evaluate
# =========================
pred_output = trainer.predict(test_dataset)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=-1)

y_true_names = [id2label[i] for i in y_true]
y_pred_names = [id2label[i] for i in y_pred]

print("\nClassification Report:")
report = classification_report(y_true_names, y_pred_names, zero_division=0)
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(
    y_true_names,
    y_pred_names,
    labels=["commitment", "intimacy", "passion"]
)
print(cm)

# =========================
# Better Confusion Matrix Plot (Thesis-ready)
# =========================
import seaborn as sns
import matplotlib.pyplot as plt

labels = ["commitment", "intimacy", "passion"]

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 12}
)

plt.title("Confusion Matrix for BERT Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.tight_layout()

# Save for LaTeX
cm_path = output_dir / "confusion_matrix_bert.png"
plt.savefig(cm_path, dpi=300)

plt.show()

# =========================
# Normalized Confusion Matrix
# =========================
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix - BERT")

plt.tight_layout()

cm_norm_path = output_dir / "confusion_matrix_bert_normalized.png"
plt.savefig(cm_norm_path, dpi=300)

plt.show()


# Optional: save report to file
report_path = output_dir / "classification_report.txt"
output_dir.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n")
    f.write(report)
    f.write("\n\nConfusion Matrix\n")
    f.write(np.array2string(cm))

# =========================
# Manual sample predictions
# =========================
samples = [
    "I want to explore every secret corner of your body tonight",
    "I’ve never felt more at home than I do when I’m just sitting in silence with you",
    "I choose you as my family, my partner, and my spouse, through every high and every low",
]

print("\nManual sample predictions:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

for text in samples:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(outputs.logits.argmax(dim=-1).item())

    print(f'"{text}" -> {id2label[pred_id]}')