import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

print("--- Initializing PyTorch Native Vitals-Only Evaluation Engine ---")
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
# 2. LOAD CACHED TRAINING AND VALIDATION MATRICES
# =====================================================================
print("Loading uncompromised feature matrices from storage engine...")
df_train = pd.read_csv("train_features.csv")
df_val = pd.read_csv("val_features.csv")

# Isolate the first 128 columns representing the Vitals Context space
X_train_vitals = torch.tensor(df_train.iloc[:, :128].values, dtype=torch.float32).to(device)
y_train = torch.tensor(df_train.iloc[:, -1].values, dtype=torch.long).to(device)

X_val_vitals = torch.tensor(df_val.iloc[:, :128].values, dtype=torch.float32).to(device)
y_val_true = df_val.iloc[:, -1].values.astype(int)

# =====================================================================
# 3. DEFINE NATIVE VITALS CLASSIFIER PROBE HEAD
# =====================================================================
class VitalsOnlyClassifierHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes) # Outputs raw logits
        )
    def forward(self, x):
        return self.network(x)

clf_head = VitalsOnlyClassifierHead().to(device)

# =====================================================================
# 4. OPTIMIZE PROBE HEAD NATIVELY (REPLACES SCIKIT-LEARN)
# =====================================================================
print("Tuning PyTorch vitals-only linear probe across scaled distributions...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(clf_head.parameters(), lr=1e-3, weight_decay=1e-4)

# Create mini-batches for the linear probe optimization
dataset = torch.utils.data.TensorDataset(X_train_vitals, y_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

clf_head.train()
for epoch in range(30): # Smooth, fast convergence curve over 30 steps
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = clf_head(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# =====================================================================
# 5. EXECUTE PARALLEL EVALUATION PASS
# =====================================================================
clf_head.eval()
print("Executing parallel validation inference pass...")
with torch.no_grad():
    logits = clf_head(X_val_vitals)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# =====================================================================
# 6. COMPUTE COMPONENT TELEMETRY REPORTS
# =====================================================================
top1_acc = accuracy_score(y_val_true, preds)

top5_hits = 0
for i in range(len(y_val_true)):
    top5_predictions = np.argsort(probs[i])[-5:]
    if y_val_true[i] in top5_predictions:
        top5_hits += 1
top5_acc = top5_hits / len(y_val_true)

print("\n" + "="*50)
print("         PYTORCH NATIVE VITALS-ONLY REPORT       ")
print("="*50)
print(f"Top-1 Accuracy (Exact ICD Prediction): {top1_acc*100:.2f}%")
print(f"Top-5 Accuracy (Differential Diagnosis List): {top5_acc*100:.2f}%")
print("="*50)

# =====================================================================
# 7. PREDICTIVE UNCERTAINTY ANALYSIS (SHANNON ENTROPY)
# =====================================================================
eps = 1e-15
prediction_entropy = -np.sum(probs * np.log2(probs + eps), axis=1)

UNCERTAINTY_THRESHOLD = 1.5
high_uncertainty_mask = prediction_entropy > UNCERTAINTY_THRESHOLD

low_uncertainty_preds = preds[~high_uncertainty_mask]
low_uncertainty_true = y_val_true[~high_uncertainty_mask]
safe_accuracy = accuracy_score(low_uncertainty_true, low_uncertainty_preds) if len(low_uncertainty_true) > 0 else 0.0

print(f"\n" + "="*50)
print("            UNCERTAINTY ANALYSIS REPORT          ")
print("="*50)
print(f"Mean Cohort Entropy: {prediction_entropy.mean():.4f} bits")
print(f"Flagged for Human Review (Uncertain): {high_uncertainty_mask.sum()} / {len(y_val_true)} samples ({high_uncertainty_mask.mean()*100:.2f}%)")
print(f"Accuracy on Safe/Confident Sub-cohort: {safe_accuracy*100:.2f}%")
print("="*50)

# =====================================================================
# 8. ERROR GEOMETRY PROFILING
# =====================================================================
print("\nTop 5 Most Severe Vitals-Only Diagnostic Confusions:")
print("-" * 65)
misclassifications = [(true, pred) for true, pred in zip(y_val_true, preds) if true != pred]
top_confused_pairs = Counter(misclassifications).most_common(5)

if top_confused_pairs:
    for (true_cls, pred_cls), occurrence in top_confused_pairs:
        true_lbl = id_to_icd_string[true_cls]
        pred_lbl = id_to_icd_string[pred_cls]
        print(f"True Condition [{true_lbl:<7}] was mistaken for Predicted Condition [{pred_lbl:<7}] | Occurrences: {occurrence}")
else:
    print("Zero confusion pairs found! Perfection achieved.")
print("-" * 65)