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
# =========================
# 3D Emotion Plot from Sample Sentences / Dialogue Utterances
# =========================
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

SEED = 42

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

# =========================
# Save RAW model outputs 
# =========================

# Get probabilities from logits
probs = torch.nn.functional.softmax(
    torch.tensor(pred_output.predictions), dim=1
).numpy()

# Create DataFrame
raw_df = pd.DataFrame({
    "text": test_df["text"].values,
    "true_label": [id2label[i] for i in y_true],
    "raw_intimacy": probs[:, label2id["intimacy"]],
    "raw_passion": probs[:, label2id["passion"]],
    "raw_commitment": probs[:, label2id["commitment"]],
    "raw_prediction": [id2label[i] for i in y_pred],
})

# Save CSV
raw_csv_path = output_dir / "bert_raw_outputs.csv"
output_dir.mkdir(parents=True, exist_ok=True)
raw_df.to_csv(raw_csv_path, index=False)

print("\nRaw outputs saved to:", raw_csv_path)

# =========================
#  Evidence marker detector
# =========================

schema_markers = {
    "intimacy": [
        "trust", "understand", "understands me", "understood", 
        "share", "open up", "safe", "feel safe", "close", "close to you",
        "know me", "you know me", "i know you", "listen",
        "talk to you", "tell you", "care about you",
        "feel understood", "honest with you", "comfort", "happy with you", "happy together", 
        "home", "not alone", "feel alone", "i don't feel alone",
        "miss you", "important to me", "most important thing",
        "you matter", "makes me happy", "make me happy",
        "you see me", "be myself", "vulnerable",
        "afraid", "scared", "my heart", "my best friend",
        "best friend", "i feel better", "feel better",
        "when i'm with you", "when i am with you", "happy to be with you" ,
        "being with you", "look at you", "heard me", "i'm here" , "i'm here for you" ,
        "you make me feel", "you make me really happy"
    ],

    "passion": [
        "want you", "i want you", "need you", "kiss", "kiss me",
        "kiss you", "touch", "touch me", "touch you",
        "desire", "crave", "fire", "burn", "burning",
        "ache", "aching", "hungry for you", "can't stay away",
        "cant stay away", "body", "your body", "skin",
        "your skin", "lips", "your lips", "breath",
        "breath against", "neck", "pull me closer",
        "hold me", "hold you", "chemistry", "attracted",
        "attraction", "breathless", "right now",
        "every inch", "trace every inch", "explore every inch",
        "thinking about you", "stop thinking about",
        "heart skip", "make my heart skip", "thought about you" , "thought of you" , "thinking of you" ,
        "smell of you", "mouth", "the way you kiss",
        "begging me", "tonight", "wild", "desperately",
        "obsessed", "can't stop", "cant stop"
    ],

    "commitment": [
        "promise", "promised", "forever with you", "forever with me", "always", "always with you", "always with me",
        "stay", "stay with me", "never leave", "never let you go", "never leave you", "forever together" , "together forever" ,
        "choose you", "i choose you", "i will choose you",
        "no matter what", "no matter what happens",
        "be there", "i'll be there", "i will be there",
        "loyal", "future", "marry", "marriage", "wife", "home", "home with you",
        "husband", "grow old", "grow old with you", "devoted", "devoted to you",
        "rest of my life", "for the rest of my life",
        "rest of your life", "stand by you",
        "through anything", "through everything",
        "i'm not going anywhere", "im not going anywhere",
        "i will be here", "i'll be here", "we will",
        "together", "life with you", "living after you",
        "survive", "wait for you", "come back",
        "came back", "i love you", "i have always loved you",
        "always loved you", "i'll love you", "i will love you",
        "tomorrow i'll love", "hundred lifetimes",
        "hundred worlds", "any version of reality",
        "you are my everything", "my everything",
        "happiest with you", "full and happy life"
    ],
}

def detect_evidence(text):
    text_lower = str(text).lower()

    evidence = {
        "intimacy_evidence": [],
        "passion_evidence": [],
        "commitment_evidence": [],
    }

    for label, markers in schema_markers.items():
        for marker in markers:
            if marker in text_lower:
                evidence[f"{label}_evidence"].append(marker)

    return evidence

# Test the detector quickly
example_text = "I promise I will always stay with you."
print("\nEvidence detector test:")
print(example_text)
print(detect_evidence(example_text))

# =========================
# Step 5: Add evidence columns to raw outputs
# =========================

evidence_rows = raw_df["text"].apply(detect_evidence)

raw_df["intimacy_evidence"] = evidence_rows.apply(lambda x: ", ".join(x["intimacy_evidence"]))
raw_df["passion_evidence"] = evidence_rows.apply(lambda x: ", ".join(x["passion_evidence"]))
raw_df["commitment_evidence"] = evidence_rows.apply(lambda x: ", ".join(x["commitment_evidence"]))

raw_with_evidence_path = output_dir / "bert_raw_outputs_with_evidence.csv"
raw_df.to_csv(raw_with_evidence_path, index=False)

print("\nRaw outputs with evidence saved to:", raw_with_evidence_path)
print(raw_df[[
    "text",
    "true_label",
    "raw_prediction",
    "intimacy_evidence",
    "passion_evidence",
    "commitment_evidence"
]].head())

# =========================
# Step 6: Convert evidence into numeric scores
# =========================

raw_df["intimacy_evidence_score"] = raw_df["intimacy_evidence"].apply(
    lambda x: 0 if x == "" else len(x.split(", "))
)

raw_df["passion_evidence_score"] = raw_df["passion_evidence"].apply(
    lambda x: 0 if x == "" else len(x.split(", "))
)

raw_df["commitment_evidence_score"] = raw_df["commitment_evidence"].apply(
    lambda x: 0 if x == "" else len(x.split(", "))
)

raw_with_scores_path = output_dir / "bert_raw_outputs_with_evidence_scores.csv"
raw_df.to_csv(raw_with_scores_path, index=False)

print("\nRaw outputs with evidence scores saved to:", raw_with_scores_path)

print(raw_df[[
    "text",
    "raw_prediction",
    "intimacy_evidence_score",
    "passion_evidence_score",
    "commitment_evidence_score"
]].head())

# =========================
# Step 7: Auditor (correction function)
# =========================

def apply_audit(row):
    # Copy raw scores
    i = row["raw_intimacy"]
    p = row["raw_passion"]
    c = row["raw_commitment"]

    # Evidence scores
    i_e = row["intimacy_evidence_score"]
    p_e = row["passion_evidence_score"]
    c_e = row["commitment_evidence_score"]

    # --- Rule 1: Boost based on evidence ---
    i += 0.1 * i_e
    p += 0.1 * p_e
    c += 0.1 * c_e

    # --- Rule 2: Penalize if no evidence but high score ---
    if i > 0.5 and i_e == 0:
        i -= 0.2
    if p > 0.5 and p_e == 0:
        p -= 0.2
    if c > 0.5 and c_e == 0:
        c -= 0.2

    # --- Rule 3: Resolve conflicts (intimacy vs commitment) ---
    if i_e < c_e:
        i -= 0.1
        c += 0.1
    elif c_e < i_e:
        c -= 0.1
        i += 0.1

    # --- Keep values in range [0,1] ---
    i = max(0, min(1, i))
    p = max(0, min(1, p))
    c = max(0, min(1, c))

    return pd.Series({
        "verified_intimacy": i,
        "verified_passion": p,
        "verified_commitment": c,
    })

# Apply auditor
verified_scores = raw_df.apply(apply_audit, axis=1)

raw_df = pd.concat([raw_df, verified_scores], axis=1)

# Get final prediction
raw_df["verified_prediction"] = raw_df[[
    "verified_commitment",
    "verified_intimacy",
    "verified_passion"
]].idxmax(axis=1).str.replace("verified_", "")

# =========================
# Audit Status (Supported / Corrected / Uncertain)
# =========================

def audit_status(row):
    final_pred = row["verified_prediction"]

    evidence_scores = {
        "intimacy": row["intimacy_evidence_score"],
        "passion": row["passion_evidence_score"],
        "commitment": row["commitment_evidence_score"],
    }

    best_evidence_label = max(evidence_scores, key=evidence_scores.get)
    best_evidence_score = evidence_scores[best_evidence_label]

    if best_evidence_score == 0:
        return "uncertain"

    if final_pred == best_evidence_label:
        return "supported"

    return "corrected"

raw_df["audit_status"] = raw_df.apply(audit_status, axis=1)

# =========================
# Evidence Alignment Before vs After Audit
# =========================

def best_evidence_label(row):
    evidence_scores = {
        "intimacy": row["intimacy_evidence_score"],
        "passion": row["passion_evidence_score"],
        "commitment": row["commitment_evidence_score"],
    }

    best_label = max(evidence_scores, key=evidence_scores.get)
    best_score = evidence_scores[best_label]

    if best_score == 0:
        return "none"

    return best_label


raw_df["best_evidence_label"] = raw_df.apply(best_evidence_label, axis=1)

evidence_df = raw_df[raw_df["best_evidence_label"] != "none"].copy()

raw_alignment = (
    evidence_df["raw_prediction"] == evidence_df["best_evidence_label"]
).mean()

verified_alignment = (
    evidence_df["verified_prediction"] == evidence_df["best_evidence_label"]
).mean()

print("\nEvidence Alignment Before vs After Audit:")
print(f"Raw prediction evidence alignment: {raw_alignment:.4f}")
print(f"Verified prediction evidence alignment: {verified_alignment:.4f}")
print(f"Evidence-supported samples: {len(evidence_df)} out of {len(raw_df)}")

# Save file
verified_path = output_dir / "bert_verified_outputs.csv"
raw_df.to_csv(verified_path, index=False)

print("\nVerified outputs saved to:", verified_path)

print(raw_df[[
    "text",
    "raw_prediction",
    "verified_prediction",
    "verified_intimacy",
    "verified_passion",
    "verified_commitment"
]].head())

verified_path = output_dir / "bert_verified_outputs.csv"
raw_df.to_csv(verified_path, index=False)
print("\nAudit Status Distribution:")
print(raw_df["audit_status"].value_counts(normalize=True) * 100)

print("\nAudit Status Distribution:")

counts = raw_df["audit_status"].value_counts()
total = len(raw_df)

for status in ["supported", "corrected", "uncertain"]:
    if status in counts:
        percentage = (counts[status] / total) * 100
        print(f"{status}: {counts[status]} ({percentage:.2f}%)")
# =========================
# Step 8: Compare raw vs verified performance
# =========================

raw_accuracy = accuracy_score(raw_df["true_label"], raw_df["raw_prediction"])
raw_f1 = f1_score(raw_df["true_label"], raw_df["raw_prediction"], average="macro")

verified_accuracy = accuracy_score(raw_df["true_label"], raw_df["verified_prediction"])
verified_f1 = f1_score(raw_df["true_label"], raw_df["verified_prediction"], average="macro")

print("\nRaw vs Verified Performance:")
print(f"Raw Accuracy: {raw_accuracy:.4f}")
print(f"Raw Macro F1: {raw_f1:.4f}")
print(f"Verified Accuracy: {verified_accuracy:.4f}")
print(f"Verified Macro F1: {verified_f1:.4f}")

comparison_df = pd.DataFrame({
    "version": ["Raw BERT", "BERT + Audit"],
    "accuracy": [raw_accuracy, verified_accuracy],
    "macro_f1": [raw_f1, verified_f1],
})

comparison_path = output_dir / "raw_vs_verified_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\nComparison saved to:", comparison_path)
print(comparison_df)

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

labels = ["commitment", "intimacy", "passion"]
_cm_annot, _cm_title, _cm_axis, _cm_tick = 20, 18, 16, 14

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": _cm_annot},
)

plt.tick_params(axis="both", labelsize=_cm_tick)
plt.title("Confusion Matrix for BERT Model", fontsize=_cm_title)
plt.xlabel("Predicted Label", fontsize=_cm_axis)
plt.ylabel("True Label", fontsize=_cm_axis)

plt.tight_layout()

cm_path = output_dir / "confusion_matrix_bert.png"
plt.savefig(cm_path, dpi=300)

plt.show()

# =========================
# Normalized Confusion Matrix
# =========================
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": _cm_annot},
)

plt.tick_params(axis="both", labelsize=_cm_tick)
plt.xlabel("Predicted Label", fontsize=_cm_axis)
plt.ylabel("True Label", fontsize=_cm_axis)
plt.title("Normalized Confusion Matrix - BERT", fontsize=_cm_title)

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
sample_texts = test_df["text"].sample(30, random_state=SEED).tolist()

print("\n3D plot sample predictions:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

plot_data = []

for i, text in enumerate(sample_texts, start=1):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    commitment_score = float(probs[label2id["commitment"]])
    intimacy_score = float(probs[label2id["intimacy"]])
    passion_score = float(probs[label2id["passion"]])

    plot_data.append({
        "text": text,
        "commitment": commitment_score,
        "intimacy": intimacy_score,
        "passion": passion_score,
        "predicted_label": id2label[pred_id],
        "index": i,
    })

    print(f'{i}. "{text}"')
    print(
        f'   commitment={commitment_score:.3f}, '
        f'intimacy={intimacy_score:.3f}, '
        f'passion={passion_score:.3f} '
        f'-> predicted: {id2label[pred_id]}'
    )

# Convert to DataFrame for easier saving if needed
plot_df = pd.DataFrame(plot_data)
print("\nPlot DataFrame:")
print(plot_df)

# Save the probabilities to CSV
plot_csv_path = output_dir / "3d_plot_probabilities.csv"
output_dir.mkdir(parents=True, exist_ok=True)
plot_df.to_csv(plot_csv_path, index=False)

# Create 3D scatter plot
_3d_title, _3d_axis, _3d_tick, _3d_legend, _3d_annot = 18, 16, 14, 14, 14

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

x = plot_df["intimacy"]
y = plot_df["passion"]
z = plot_df["commitment"]

ax.scatter(x, y, z, s=80)

# Annotate each point with sentence number
for _, row in plot_df.iterrows():
    ax.text(
        row["intimacy"],
        row["passion"],
        row["commitment"],
        str(row["index"]),
        fontsize=_3d_annot,
    )
color_map = {
    "passion": "red",
    "intimacy": "blue",
    "commitment": "green"
}

colors = [color_map[label] for label in plot_df["predicted_label"]]

ax.scatter(x, y, z, c=colors, s=80)

legend_handles = [
    mpatches.Patch(color="red", label="Passion"),
    mpatches.Patch(color="blue", label="Intimacy"),
    mpatches.Patch(color="green", label="Commitment"),
]

ax.legend(handles=legend_handles, fontsize=_3d_legend)

ax.set_xlabel("Intimacy", fontsize=_3d_axis)
ax.set_ylabel("Passion", fontsize=_3d_axis)
ax.set_zlabel("Commitment", fontsize=_3d_axis)
ax.set_title("3D Distribution of BERT Emotion Probabilities", fontsize=_3d_title)
for _axis in "xyz":
    ax.tick_params(axis=_axis, labelsize=_3d_tick)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)

ax.scatter([1,0,0], [0,1,0], [0,0,1], s=200)
ax.text(1, 0, 0, "Intimacy", fontsize=_3d_annot)
ax.text(0, 1, 0, "Passion", fontsize=_3d_annot)
ax.text(0, 0, 1, "Commitment", fontsize=_3d_annot)
plt.tight_layout()

ax.view_init(elev=20, azim=45)

plot_path = output_dir / "3d_emotion_plot.png"
plt.savefig(plot_path, dpi=300)
plt.show()

# =========================
# 3D Emotional Trajectory Plot for One Scene
# =========================
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

trajectory_samples = [
    "I missed you.",
    "I was scared something would happen to you.",
    "Being with you makes me feel safe.",
    "I want you right now.",
    "Please stay with me tonight.",
    "I will always choose you.",
]

trajectory_data = []

for i, text in enumerate(trajectory_samples, start=1):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    trajectory_data.append({
        "step": i,
        "text": text,
        "commitment": float(probs[label2id["commitment"]]),
        "intimacy": float(probs[label2id["intimacy"]]),
        "passion": float(probs[label2id["passion"]]),
        "predicted_label": id2label[pred_id],
    })

trajectory_df = pd.DataFrame(trajectory_data)

print("\nTrajectory DataFrame:")
print(trajectory_df)

# Save CSV
trajectory_csv_path = output_dir / "3d_trajectory_probabilities.csv"
trajectory_df.to_csv(trajectory_csv_path, index=False)

# Color map
color_map = {
    "passion": "red",
    "intimacy": "blue",
    "commitment": "green"
}
colors = [color_map[label] for label in trajectory_df["predicted_label"]]

# Coordinates
x = trajectory_df["intimacy"]
y = trajectory_df["passion"]
z = trajectory_df["commitment"]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter points
ax.scatter(x, y, z, c=colors, s=90)

# Connect points in order
ax.plot(x, y, z, linewidth=2)

# Annotate each point with step number
for _, row in trajectory_df.iterrows():
    ax.text(
        row["intimacy"],
        row["passion"],
        row["commitment"],
        str(row["step"]),
        fontsize=_3d_annot,
    )

# Axis labels
ax.set_xlabel("Intimacy", fontsize=_3d_axis)
ax.set_ylabel("Passion", fontsize=_3d_axis)
ax.set_zlabel("Commitment", fontsize=_3d_axis)
ax.set_title("3D Emotional Trajectory for a Dialogue Scene", fontsize=_3d_title)
for _axis in "xyz":
    ax.tick_params(axis=_axis, labelsize=_3d_tick)

# Keep same scale
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Better viewing angle
ax.view_init(elev=20, azim=45)

# Legend
legend_handles = [
    mpatches.Patch(color="red", label="Passion"),
    mpatches.Patch(color="blue", label="Intimacy"),
    mpatches.Patch(color="green", label="Commitment"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=_3d_legend)

plt.tight_layout()

trajectory_plot_path = output_dir / "3d_emotional_trajectory_scene.png"
plt.savefig(trajectory_plot_path, dpi=300)
plt.show()