# Sternberg’s Love Quantifier

A computational framework for modeling, validating, and visualizing Sternberg’s Triangular Theory of Love using Natural Language Processing.

---

## Overview

This project applies machine learning to quantify emotional components of love in text based on Sternberg’s theory:

- Intimacy
- Passion
- Commitment

Beyond classification, the system introduces an auditing layer to evaluate the reliability of model predictions and improve interpretability.

---

## Key Contributions

- Fine-tuned BERT model for emotional classification
- Custom balanced dataset (300+ labeled samples)
- 3D visualization of emotional components
- Comparative analysis with BiLSTM and TextCNN
- Prediction auditing system for reliability and explainability

---

## Auditing System

Neural models like BERT can produce confident predictions that are not always fully supported by the input text.

To address this, an auditing layer was introduced that classifies predictions into:

- **Supported**
  The prediction is clearly justified by the text

- **Corrected**
  The model prediction is adjusted based on rule-based verification

- **Uncertain**
  The prediction lacks strong supporting evidence

This allows the system to move from a purely predictive model to a more **trustworthy and interpretable framework**, even when raw accuracy improvements are moderate.

---

## Project Structure

```text
.
├── code/              # Training, testing, visualization, and auditing scripts
├── dataset/           # Labeled datasets
├── .gitignore
└── README.md
```

---

## Models Used

- **BERT (Primary Model)**
  Transformer-based model with strong contextual understanding

- **BiLSTM**
  Sequential model capturing word dependencies

- **TextCNN**
  Pattern-based model for local feature extraction

---

## Methodology

1. Text is processed using trained models (BERT, BiLSTM, TextCNN)
2. Each sentence is classified into one of the three components
3. The model outputs probability scores for each class
4. These probabilities are mapped into a 3D space:
   - X → Intimacy
   - Y → Passion
   - Z → Commitment

5. An auditing layer evaluates prediction reliability
6. Visualizations are generated for analysis

---

## Visualization

- **Distribution Plot**
  Shows how emotional components are spread across the dataset

- **Trajectory Plot**
  Shows how emotional states evolve over time in a dialogue

---

## Dataset

The dataset consists of labeled textual samples divided into:

- Intimacy
- Passion
- Commitment

The dataset is balanced to ensure fair model training and evaluation.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/rawanhossam27/BSc_SS22_Sternberg-s_Love_Quantifier
cd BSc_SS22_Sternberg-s_Love_Quantifier
```

### 2. Install Python

Make sure you have:

- Python 3.10 or higher

Check version:

```bash
python --version
```

---

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
pip install torch transformers datasets pandas scikit-learn matplotlib numpy
```

---

## Running the Project

Example:

```bash
python code/bert3D.py
```

Other scripts:

- `bert_test.py` → testing BERT model
- `bertWithMatrixPlot.py` → Code without 3D visualization
- `codeWithAudit.py` → auditing system
- `nonBert-BiLSTM.py` → BiLSTM model
- `textCNN.py` → TextCNN model

---

## Notes

- Model checkpoints and outputs are not included (added in the models folder and changed with every run)
- Results can be reproduced by running the scripts
- Random seeds can be modified to test stability

---

## Future Work

- Improve separation between intimacy and commitment
- Expand dataset for better generalization
- Integrate explainable AI techniques
- Enhance visualization quality

---

## Author

Rawan Hossam
Bachelor Thesis – Sternberg’s Love Quantifier
