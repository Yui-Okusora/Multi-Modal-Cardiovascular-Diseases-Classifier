# src/TimelineDataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BVTDFlattenedDataset(Dataset):
    """🎯 PRE-FLATTENED LONGITUDINAL TRAJECTORY DATASET LOADER (REFACTORED)"""
    def __init__(self, preprocessed_csv_path, max_seq_len=128, max_targets=10):
        self.df = pd.read_csv(preprocessed_csv_path).fillna("")
        self.max_seq_len = max_seq_len
        self.max_targets = max_targets
        
    def __len__(self):
        return len(self.df)
        
    def _parse_sequence(self, str_val, dtype, pad_value, max_len):
        tokens = str(str_val).strip().split() if str(str_val).strip() else []
        tokens = [dtype(t) for t in tokens][:max_len]
        actual_len = len(tokens)
        padded = tokens + [pad_value] * (max_len - actual_len)
        mask = [False] * actual_len + [True] * (max_len - actual_len)
        return torch.tensor(padded, dtype=torch.float32 if dtype==float else torch.long), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cutoff = int(row['cutoff_idx']) # Direct absolute position tracking
        
        # ─── EXTRACT STANDALONE STATIC FEATURES ───
        age_tensor = torch.tensor(float(row['age']), dtype=torch.float32)
        gender_tensor = torch.tensor(int(row['gender']), dtype=torch.long)
        
        # Parse dynamic context sequences
        times, base_mask = self._parse_sequence(row['timestamps'], float, 0.0, self.max_seq_len)
        f_ids, _         = self._parse_sequence(row['feature_ids'], int, 0, self.max_seq_len)
        v_nums, _        = self._parse_sequence(row['numeric_values'], float, 0.0, self.max_seq_len)
        c_ids, _         = self._parse_sequence(row['cat_result_ids'], int, 0, self.max_seq_len)
        
        # 🛡️ Clean Student Causal Mask (Removes legacy +2 index padding hacks)
        student_mask = base_mask.clone()
        if cutoff + 1 < self.max_seq_len:
            student_mask[cutoff + 1:] = True  # Censors everything occurring after the true target token
            
        # 🎯 Clean Teacher Target Mask (Removes legacy +2 index padding hacks)
        teacher_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        if cutoff < self.max_seq_len:
            teacher_mask[cutoff] = False      # Isolates exactly the current target token step
            
        icd_ids, tgt_mask = self._parse_sequence(row['icd_targets'], int, 0, self.max_targets)
        
        return {
            'patient_session_id': str(row['mabn']),         # Shape: Python string (Non-tensor session ID)
            'feature_ids': f_ids,                           # Shape: [L]        │ Batched: [B, L]       (LongTensor)
            'numeric_values': v_nums,                       # Shape: [L]        │ Batched: [B, L]       (FloatTensor)
            'cat_result_ids': c_ids,                        # Shape: [L]        │ Batched: [B, L]       (LongTensor)
            'timestamps': times,                            # Shape: [L]        │ Batched: [B, L]       (FloatTensor)
            'student_mask': student_mask,                   # Shape: [L]        │ Batched: [B, L]       (BoolTensor)
            'teacher_mask': teacher_mask,                   # Shape: [L]        │ Batched: [B, L]       (BoolTensor)
            'age': age_tensor,                              # Shape: [] Scalar  │ Batched: [B] -> [B,1] (FloatTensor)
            'gender': gender_tensor,                        # Shape: [] Scalar  │ Batched: [B]          (LongTensor)
            'icd_targets': icd_ids,                         # Shape: [T]        │ Batched: [B, T]       (LongTensor)
            'target_mask': tgt_mask                         # Shape: [T]        │ Batched: [B, T]       (BoolTensor)
        }
    
def compute_static_class_frequencies(csv_path, num_classes=456):
    df = pd.read_csv(csv_path).fillna("")
    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    for target_str in df['icd_targets']:
        tokens = str(target_str).strip().split()
        for t in tokens:
            class_id = int(t)
            if class_id < num_classes:
                class_counts[class_id] += 1
                
    frequencies = class_counts / len(df)
    frequencies = np.clip(frequencies, 1e-5, 1.0)
    return torch.tensor(frequencies, dtype=torch.float32)