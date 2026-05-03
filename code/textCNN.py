from pathlib import Path
import re
from collections import Counter
import random

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
import seaborn as sns

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# Paths
# =========================
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "300dataset.csv"
output_dir = Path(__file__).resolve().parent.parent / "models" / "textcnn_output"
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
# Build vocabulary from train only
# =========================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
pad_idx = 0
unk_idx = 1

counter = Counter()
for text in train_df["text"]:
    counter.update(simple_tokenize(text))

vocab = {
    PAD_TOKEN: pad_idx,
    UNK_TOKEN: unk_idx,
}

min_freq = 1
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

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))

    return ids

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
        input_ids = encode_text(text, self.vocab, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
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
# TextCNN Model
# =========================
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        num_classes=3,
        pad_idx=0,
        num_filters=100,
        filter_sizes=(2, 3, 4),
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs,
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        embedded = self.embedding(input_ids)            # [batch, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)           # [batch, emb_dim, seq_len]

        conv_outputs = []
        for conv in self.convs:
            x = conv(embedded)                         # [batch, num_filters, L]
            x = F.relu(x)
            x = F.max_pool1d(x, kernel_size=x.shape[2])  # [batch, num_filters, 1]
            x = x.squeeze(2)                          # [batch, num_filters]
            conv_outputs.append(x)

        concatenated = torch.cat(conv_outputs, dim=1) # [batch, num_filters * len(filter_sizes)]
        dropped = self.dropout(concatenated)
        logits = self.fc(dropped)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=100,
    num_classes=3,
    pad_idx=pad_idx,
    num_filters=100,
    filter_sizes=(2, 3, 4),
    dropout=0.3,
).to(device)

# =========================
# Training setup
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
best_f1 = 0.0
best_model_path = output_dir / "best_textcnn.pt"

# =========================
# Training loop
# =========================
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

    # validation
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
# Confusion matrix plots
# =========================
labels = ["commitment", "intimacy", "passion"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 12}
)
plt.title("Confusion Matrix for TextCNN Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()

cm_path = output_dir / "confusion_matrix_textcnn.png"
plt.savefig(cm_path, dpi=300)
plt.show()

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 12}
)
plt.title("Normalized Confusion Matrix for TextCNN Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()

cm_norm_path = output_dir / "confusion_matrix_textcnn_normalized.png"
plt.savefig(cm_norm_path, dpi=300)
plt.show()

# =========================
# Save report
# =========================
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
    input_ids = encode_text(text, vocab, max_len)
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
# Distribution plot
# =========================
fig = plt.figure(figsize=(9, 7))
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
        fontsize=10
    )

legend_handles = [
    mpatches.Patch(color="red", label="Passion"),
    mpatches.Patch(color="blue", label="Intimacy"),
    mpatches.Patch(color="green", label="Commitment"),
]
ax.legend(handles=legend_handles)

ax.set_xlabel("Intimacy")
ax.set_ylabel("Passion")
ax.set_zlabel("Commitment")
ax.set_title("3D Emotional Distribution Based on TextCNN Probabilities")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1], s=200, c="orange")
ax.text(1, 0, 0, "Intimacy")
ax.text(0, 1, 0, "Passion")
ax.text(0, 0, 1, "Commitment")

ax.view_init(elev=20, azim=45)
plt.tight_layout()

plot_path = output_dir / "3d_emotion_plot_textcnn.png"
plt.savefig(plot_path, dpi=300)
plt.show()

# =========================
# Trajectory plot
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
    input_ids = encode_text(text, vocab, max_len)
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

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, c=colors, s=90)
ax.plot(x, y, z, linewidth=2)

for _, row in trajectory_df.iterrows():
    ax.text(
        row["intimacy"],
        row["passion"],
        row["commitment"],
        str(row["step"]),
        fontsize=10
    )

ax.set_xlabel("Intimacy")
ax.set_ylabel("Passion")
ax.set_zlabel("Commitment")
ax.set_title("3D Emotional Trajectory for a Dialogue Scene Based on TextCNN")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.view_init(elev=20, azim=45)
ax.legend(handles=legend_handles, loc="upper left")

plt.tight_layout()

trajectory_plot_path = output_dir / "3d_emotional_trajectory_scene_textcnn.png"
plt.savefig(trajectory_plot_path, dpi=300)
plt.show()