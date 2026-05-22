from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# Paths
# =========================
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "300dataset.csv"
output_dir = Path(__file__).resolve().parent.parent / "models" / "bilstm_output"
output_dir.mkdir(parents=True, exist_ok=True)

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
# Simple tokenizer
# =========================
def simple_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

# =========================
# Build vocabulary from train set only
# =========================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
pad_idx = 0
unk_idx = 1

counter = Counter()
for text in train_df["text"]:
    counter.update(simple_tokenize(text))

min_freq = 1
vocab = {
    PAD_TOKEN: pad_idx,
    UNK_TOKEN: unk_idx,
}

for token, freq in counter.items():
    if freq >= min_freq:
        vocab[token] = len(vocab)

print("\nVocab size:", len(vocab))

# =========================
# Encode text
# =========================
max_len = 40

def encode_text(text: str, vocab: dict, max_len: int):
    tokens = simple_tokenize(text)
    ids = [vocab.get(token, unk_idx) for token in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]

    length = len(ids)

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))

    return ids, length

# =========================
# Dataset
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids, length = encode_text(text, self.vocab, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }

train_dataset = TextDataset(
    train_df["text"],
    train_df["label_id"],
    vocab,
    max_len,
)

test_dataset = TextDataset(
    test_df["text"],
    test_df["label_id"],
    vocab,
    max_len,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =========================
# BiLSTM Model
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_classes=3,
        pad_idx=0,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)

        # hidden shape: (num_directions, batch, hidden_dim)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    num_classes=3,
    pad_idx=pad_idx,
    dropout=0.3,
).to(device)

# =========================
# Training setup
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

# =========================
# Training loop
# =========================
best_f1 = 0.0
best_model_path = output_dir / "best_bilstm.pt"

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Evaluate each epoch
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {np.mean(train_losses):.4f} | "
        f"Val Acc: {epoch_acc:.4f} | "
        f"Val F1 Macro: {epoch_f1:.4f}"
    )

    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        torch.save(model.state_dict(), best_model_path)

# =========================
# Load best model
# =========================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# =========================
# Final evaluation
# =========================
all_preds = []
all_labels = []
all_probs = []
all_texts = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_texts.extend(batch["text"])

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# =========================
# Save RAW BiLSTM outputs
# =========================

probs = np.array(all_probs)

raw_df = pd.DataFrame({
    "text": all_texts,
    "true_label": [id2label[i] for i in y_true],
    "raw_intimacy": probs[:, label2id["intimacy"]],
    "raw_passion": probs[:, label2id["passion"]],
    "raw_commitment": probs[:, label2id["commitment"]],
    "raw_prediction": [id2label[i] for i in y_pred],
})

raw_csv_path = output_dir / "bilstm_raw_outputs.csv"
raw_df.to_csv(raw_csv_path, index=False)

print("\nRaw BiLSTM outputs saved to:", raw_csv_path)

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

raw_with_evidence_path = output_dir / "bilstm_raw_outputs_with_evidence.csv"
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

raw_with_scores_path = output_dir / "bilstm_raw_outputs_with_evidence_scores.csv"
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
verified_path = output_dir / "bilstm_verified_outputs.csv"
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

verified_path = output_dir / "bilstm_verified_outputs.csv"
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
    "version": ["Raw BiLSTM", "BiLSTM + Audit"],
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
import seaborn as sns

# =========================
# Confusion Matrix Plot
# =========================
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
plt.title("Confusion Matrix for BiLSTM Model", fontsize=_cm_title)
plt.xlabel("Predicted Label", fontsize=_cm_axis)
plt.ylabel("True Label", fontsize=_cm_axis)

plt.tight_layout()

cm_path = output_dir / "confusion_matrix_bilstm.png"
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
plt.title("Normalized Confusion Matrix for BiLSTM Model", fontsize=_cm_title)
plt.xlabel("Predicted Label", fontsize=_cm_axis)
plt.ylabel("True Label", fontsize=_cm_axis)

plt.tight_layout()

cm_norm_path = output_dir / "confusion_matrix_bilstm_normalized.png"
plt.savefig(cm_norm_path, dpi=300)
plt.show()
# Save report
report_path = output_dir / "classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n")
    f.write(report)
    f.write("\n\nConfusion Matrix\n")
    f.write(np.array2string(cm))

# =========================
# Distribution plot data
# =========================
sample_plot_df = test_df.sample(30, random_state=SEED).copy().reset_index(drop=True)

plot_data = []

print("\n3D plot sample predictions:")

for i, row in sample_plot_df.iterrows():
    text = row["text"]
    input_ids, length = encode_text(text, vocab, max_len)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
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
        "index": i + 1,
    })

    print(f'{i + 1}. "{text}"')
    print(
        f'   commitment={commitment_score:.3f}, '
        f'intimacy={intimacy_score:.3f}, '
        f'passion={passion_score:.3f} '
        f'-> predicted: {id2label[pred_id]}'
    )

plot_df = pd.DataFrame(plot_data)

print("\nPlot DataFrame:")
print(plot_df)

plot_csv_path = output_dir / "3d_plot_probabilities.csv"
plot_df.to_csv(plot_csv_path, index=False)

# =========================
# Distribution Plot
# =========================
_3d_title, _3d_axis, _3d_tick, _3d_legend, _3d_annot = 18, 16, 14, 14, 14

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

x = plot_df["intimacy"]
y = plot_df["passion"]
z = plot_df["commitment"]

color_map = {
    "passion": "red",
    "intimacy": "blue",
    "commitment": "green"
}
colors = [color_map[label] for label in plot_df["predicted_label"]]

ax.scatter(x, y, z, c=colors, s=80)

for _, row in plot_df.iterrows():
    ax.text(
        row["intimacy"],
        row["passion"],
        row["commitment"],
        str(row["index"]),
        fontsize=_3d_annot,
    )

legend_handles = [
    mpatches.Patch(color="red", label="Passion"),
    mpatches.Patch(color="blue", label="Intimacy"),
    mpatches.Patch(color="green", label="Commitment"),
]
ax.legend(handles=legend_handles, fontsize=_3d_legend)

ax.set_xlabel("Intimacy", fontsize=_3d_axis)
ax.set_ylabel("Passion", fontsize=_3d_axis)
ax.set_zlabel("Commitment", fontsize=_3d_axis)
ax.set_title("3D Emotional Distribution Based on BiLSTM Probabilities", fontsize=_3d_title)
for _axis in "xyz":
    ax.tick_params(axis=_axis, labelsize=_3d_tick)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Sternberg pure-component anchors
ax.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1], s=200, c="orange")
ax.text(1, 0, 0, "Intimacy", fontsize=_3d_annot)
ax.text(0, 1, 0, "Passion", fontsize=_3d_annot)
ax.text(0, 0, 1, "Commitment", fontsize=_3d_annot)

ax.view_init(elev=20, azim=45)
plt.tight_layout()

plot_path = output_dir / "3d_emotion_plot_bilstm.png"
plt.savefig(plot_path, dpi=300)
plt.show()

# =========================
# Trajectory Plot
# =========================
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
    input_ids, length = encode_text(text, vocab, max_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
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

trajectory_csv_path = output_dir / "3d_trajectory_probabilities.csv"
trajectory_df.to_csv(trajectory_csv_path, index=False)

x = trajectory_df["intimacy"]
y = trajectory_df["passion"]
z = trajectory_df["commitment"]
colors = [color_map[label] for label in trajectory_df["predicted_label"]]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, c=colors, s=90)
ax.plot(x, y, z, linewidth=2)

for _, row in trajectory_df.iterrows():
    ax.text(
        row["intimacy"],
        row["passion"],
        row["commitment"],
        str(row["step"]),
        fontsize=_3d_annot,
    )

ax.set_xlabel("Intimacy", fontsize=_3d_axis)
ax.set_ylabel("Passion", fontsize=_3d_axis)
ax.set_zlabel("Commitment", fontsize=_3d_axis)
ax.set_title("3D Emotional Trajectory for a Dialogue Scene Based on BiLSTM", fontsize=_3d_title)
for _axis in "xyz":
    ax.tick_params(axis=_axis, labelsize=_3d_tick)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.view_init(elev=20, azim=45)
ax.legend(handles=legend_handles, loc="upper left", fontsize=_3d_legend)

plt.tight_layout()

trajectory_plot_path = output_dir / "3d_emotional_trajectory_scene_bilstm.png"
plt.savefig(trajectory_plot_path, dpi=300)
plt.show()