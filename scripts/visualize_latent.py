import os
import pandas as pd
import numpy as np

# Force headless rendering backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

print("--- Launching Latent Space t-SNE Visualizer ---")

# =====================================================================
# 1. RECONSTRUCT COHERENT ICD CODEBOOK
# =====================================================================
print("Parsing clinical target indices...")
cdha_raw = pd.read_csv("master_cdha_cleaned.csv", dtype=str)
cdha_raw['maicd'] = cdha_raw['maicd'].fillna("UNKNOWN_CODE").astype(str).str.strip()

# Emulate the exact patient shuffle path to sync categorical indices
all_patients = list(cdha_raw['mabn'].unique())
np.random.seed(42)  
np.random.shuffle(all_patients)
split_idx = int(len(all_patients) * 0.8)
train_mabns = set(all_patients[:split_idx])

train_cdha = cdha_raw[cdha_raw['mabn'].isin(train_mabns)].copy()
cat_type = train_cdha['maicd'].astype('category')
id_to_icd_string = dict(enumerate(cat_type.cat.categories))

# =====================================================================
# 2. LOAD COMPONENT COHORTS
# =====================================================================
print("Loading uncompromised validation feature matrix...")
df_val = pd.read_csv("val_features.csv")
target_col_index = -1

X_multimodal = df_val.iloc[:, :-1].values
y_true = df_val.iloc[:, target_col_index].values.astype(int)

# Filter down to the Top 8 most frequent classes for visual clarity
TOP_K_CLASSES = 8
frequent_class_counts = pd.Series(y_true).value_counts().head(TOP_K_CLASSES)
target_classes = frequent_class_counts.index.tolist()

filter_mask = np.isin(y_true, target_classes)
X_filtered = X_multimodal[filter_mask]
y_filtered = y_true[filter_mask]

# Translate raw integer IDs to clean alphanumeric diagnostic strings
y_strings = [id_to_icd_string[cid] for cid in y_filtered]

# =====================================================================
# 3. HIGH-DIMENSIONAL SPACE DECOMPOSITION (t-SNE)
# =====================================================================
print(f"Executing non-linear manifold decomposition on {X_filtered.shape[0]} profiles...")
print("Compressing 896 dimensions down to a 2D map. This may take a moment...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42,
    max_iter=1000
)

X_embedded = tsne.fit_transform(X_filtered)

# Assemble tidy visualization data frame
df_plot = pd.DataFrame({
    't-SNE Component 1': X_embedded[:, 0],
    't-SNE Component 2': X_embedded[:, 1],
    'Condition': y_strings
})

# =====================================================================
# 4. RENDERING HIGH-RESOLUTION PLOT MATRIX
# =====================================================================
print("Generating canvas plot elements...")
plt.figure(figsize=(12, 8), dpi=300)
sns.set_theme(style="whitegrid")

# Plot with a distinct, high-contrast palette
ax = sns.scatterplot(
    data=df_plot,
    x='t-SNE Component 1',
    y='t-SNE Component 2',
    hue='Condition',
    style='Condition',
    palette='Set1',
    alpha=0.8,
    s=55,
    edgecolor='w',
    linewidth=0.4
)

plt.title("T-JEPA Cross-Modal Latent Space Geometry\n(Validation Patient Cohort - Top 8 Active Conditions)", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("t-SNE Component 1", fontsize=11, fontweight='bold')
plt.ylabel("t-SNE Component 2", fontsize=11, fontweight='bold')

# Clean up legend layout placement
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="ICD Diagnosis", title_fontproperties={'weight':'bold'})
plt.tight_layout()

# Save image file directly to the workspace disk engine
output_filename = "latent_space_tsne.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"📊 Visualization successfully serialized! Asset saved to: '{os.getcwd()}/{output_filename}'")