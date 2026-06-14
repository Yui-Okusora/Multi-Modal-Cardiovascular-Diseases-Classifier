import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

# Import your actual production model definition
from src.TJEPA import MultimodalDownstreamClassifier

print("--- Launching Zero-Imputation Multi-Modal Ablation Pass ---")
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
# 2. LOAD VAL FEATURES & INJECT ZERO TENSORS
# =====================================================================
print("Harvesting validation profiles...")
df_val = pd.read_csv("val_features.csv")

# Extract the genuine normalized 128-dimensional vitals features
X_vitals = torch.tensor(df_val.iloc[:, :128].values, dtype=torch.float32).to(device)
y_true = df_val.iloc[:, -1].values.astype(int)

# HACK: Generate a matching tensor of absolute zeros for the 768-d text space
X_text_zero = torch.zeros(X_vitals.shape[0], 768, dtype=torch.float32).to(device)

# =====================================================================
# 3. LOAD PRODUCTION WEIGHTS NATIVELY
# =====================================================================
print("Loading trained multi-modal weights from storage checkpoint...")
checkpoint_path = os.path.join("./checkpoints", "vitals_context_encoder.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Instantiate the standard head and bind your saved weights directly
clf_head = MultimodalDownstreamClassifier(num_classes=num_classes).to(device)
clf_head.load_state_dict(checkpoint['classifier_head'])
clf_head.eval()

# =====================================================================
# 4. EXECUTE BLIND INFERENCE PASS
# =====================================================================
print("Executing parallel evaluation with zeroed text embeddings...")
with torch.no_grad():
    # Pass the real vitals alongside the zeroed-out text block
    logits = clf_head(X_vitals, X_text_zero)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# =====================================================================
# 5. COMPUTE SYSTEM METRICS
# =====================================================================
top1_acc = accuracy_score(y_true, preds)

top5_hits = 0
for i in range(len(y_true)):
    top5_predictions = np.argsort(probs[i])[-5:]
    if y_true[i] in top5_predictions:
        top5_hits += 1
top5_acc = top5_hits / len(y_true)

print("\n" + "="*50)
print("     PYTORCH DEFAULT HEAD: BLIND TEXT REPORT     ")
print("="*50)
print(f"Top-1 Accuracy (Exact ICD Prediction): {top1_acc*100:.2f}%")
print(f"Top-5 Accuracy (Differential Diagnosis List): {top5_acc*100:.2f}%")
print("="*50)

# =====================================================================
# 6. PREDICTIVE UNCERTAINTY (SHANNON ENTROPY)
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
# 7. ERROR GEOMETRY PROFILING
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