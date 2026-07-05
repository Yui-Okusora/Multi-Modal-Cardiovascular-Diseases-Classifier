# src/TimelineDataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class BVTDFlattenedDataset(Dataset):
    """🎯 PRE-FLATTENED LONGITUDINAL TRAJECTORY DATASET LOADER"""
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
        cutoff = int(row['cutoff_idx']) # Identify the exact step limit for this row
        
        times, base_mask = self._parse_sequence(row['timestamps'], float, 0.0, self.max_seq_len)
        f_ids, _         = self._parse_sequence(row['feature_ids'], int, 0, self.max_seq_len)
        v_nums, _        = self._parse_sequence(row['numeric_values'], float, 0.0, self.max_seq_len)
        c_ids, _         = self._parse_sequence(row['cat_result_ids'], int, 0, self.max_seq_len)
        
        # 🛡️ Dynamic Student Causal Mask Construction
        # Mask out all future tokens occurring after the current cutoff index position
        student_mask = base_mask.clone()
        if cutoff + 2 < self.max_seq_len:
            student_mask[cutoff + 2:] = True  # Accounts for the 2 static baseline tokens
            
        # 🎯 Dynamic Teacher Target Mask Construction
        # Blinds every position across the sequence except for the isolated target token at cutoff + 2
        teacher_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        if cutoff + 2 < self.max_seq_len:
            teacher_mask[cutoff + 2] = False
            
        icd_ids, tgt_mask = self._parse_sequence(row['icd_targets'], int, 0, self.max_targets)
        
        return {
            'patient_session_id': str(row['mabn']),
            'feature_ids': f_ids,
            'numeric_values': v_nums,
            'cat_result_ids': c_ids,
            'timestamps': times,
            'student_mask': student_mask,   # Direct sequence-level causal mask
            'teacher_mask': teacher_mask,   # Direct sequence-level target isolation mask
            'icd_targets': icd_ids,
            'target_mask': tgt_mask
        }