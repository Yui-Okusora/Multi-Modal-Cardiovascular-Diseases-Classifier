import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Core Framework Dependencies
from src.TJEPA import MultimodalDownstreamClassifier

print("--- Initializing PyTorch Native Evaluation Engine ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. RECONSTRUCT COHERENT ICD CODEBOOK
# =====================================================================
cdha_raw = pd.read_csv("master_cdha_cleaned.csv", dtype=str)
cdha_raw['maicd'] = cdha_raw['maicd'].fillna("UNKNOWN_CODE").astype(str).str.strip()

all_patients = list(cdha_raw['mabn'].unique())
np.random.seed(42)  
np.random.shuffle(all_patients)
split_idx = int(len(all_patients) * 0.8)
train_mabns = set(all_patients[:split_idx])

train_cdha = cdha_raw[cdha_raw['mabn'].isin(train_mabns)].copy()
cat_type = train_cdha['maicd'].astype('category')
id_to_icd_string = dict(enumerate(cat_type.cat.categories))
num_classes = len(id_to_icd_string)

# =====================================================================
# 2. LOAD VAL FEATURES & PYTORCH CHECKPOINT WEIGHTS
# =====================================================================
df_val = pd.read_csv("val_features.csv")
target_col_index = -1

# Unpack inputs and map cleanly to PyTorch CUDA tensors
X_vitals = torch.tensor(df_val.iloc[:, :128].values, dtype=torch.float32).to(device)
X_text = torch.tensor(df_val.iloc[:, 128:896].values, dtype=torch.float32).to(device)
y_true = df_val.iloc[:, target_col_index].values.astype(int)

print(f"Loading trained weights checkpoint directly from storage engine...")
checkpoint_path = os.path.join("./checkpoints", "vitals_context_encoder.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Instantiate the head and bind saved weights
clf_head = MultimodalDownstreamClassifier(num_classes=num_classes).to(device)
clf_head.load_state_dict(checkpoint['classifier_head'])
clf_head.eval()

# =====================================================================
# 3. NATIVE HARDWARE ACCELERATED INFERENCE PASS
# =====================================================================
print("Executing parallel evaluation pass across target channels...")
with torch.no_grad():
    logits = clf_head(X_vitals, X_text)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# =====================================================================
# 4. COMPUTE SYSTEM METRICS
# =====================================================================
top1_acc = accuracy_score(y_true, preds)

# Calculate Top-5 Accuracy natively using numpy sorting
top5_hits = 0
for i in range(len(y_true)):
    top5_predictions = np.argsort(probs[i])[-5:]
    if y_true[i] in top5_predictions:
        top5_hits += 1
top5_acc = top5_hits / len(y_true)

print("\n" + "="*50)
print("          PYTORCH NATIVE VALIDATION REPORT       ")
print("="*50)
print(f"Top-1 Accuracy (Exact ICD Prediction): {top1_acc*100:.2f}%")
print(f"Top-5 Accuracy (Differential Diagnosis List): {top5_acc*100:.2f}%")
print("="*50)

# =====================================================================
# 5. PREDICTIVE UNCERTAINTY (SHANNON ENTROPY)
# =====================================================================
eps = 1e-15
prediction_entropy = -np.sum(probs * np.log2(probs + eps), axis=1)

UNCERTAINTY_THRESHOLD = 1.5
high_uncertainty_mask = prediction_entropy > UNCERTAINTY_THRESHOLD

low_uncertainty_preds = preds[~high_uncertainty_mask]
low_uncertainty_true = y_true[~high_uncertainty_mask]
safe_accuracy = accuracy_score(low_uncertainty_true, low_uncertainty_preds) if len(low_uncertainty_true) > 0 else 0.0

print(f"\n" + "="*50)
print("            UNCERTAINTY ANALYSIS REPORT          ")
print("="*50)
print(f"Mean Cohort Entropy: {prediction_entropy.mean():.4f} bits")
print(f"Flagged for Human Review (Uncertain): {high_uncertainty_mask.sum()} / {len(y_true)} samples ({high_uncertainty_mask.mean()*100:.2f}%)")
print(f"Accuracy on Safe/Confident Sub-cohort: {safe_accuracy*100:.2f}%")
print("="*50)

# =====================================================================
# 6. ERROR GEOMETRY PROFILING
# =====================================================================
print("\nTop 5 Most Severe Diagnostic Confusions:")
print("-" * 65)
misclassifications = [(true, pred) for true, pred in zip(y_true, preds) if true != pred]
top_confused_pairs = Counter(misclassifications).most_common(5)

if top_confused_pairs:
    for (true_cls, pred_cls), occurrence in top_confused_pairs:
        true_lbl = id_to_icd_string[true_cls]
        pred_lbl = id_to_icd_string[pred_cls]
        print(f"True Condition [{true_lbl:<7}] was mistaken for Predicted Condition [{pred_lbl:<7}] | Occurrences: {occurrence}")
else:
    print("Zero confusion pairs found! Perfection achieved.")
print("-" * 65)