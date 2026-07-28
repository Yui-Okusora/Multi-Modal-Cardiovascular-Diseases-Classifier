# src/TimelineDataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from typing import Dict, Any, Tuple


# ==================================================================================================
# 1. PRE-FLATTENED LONGITUDINAL TRAJECTORY DATASET LOADER
# ==================================================================================================
class BVTDFlattenedDataset(Dataset):
    """
    DESCRIPTION:
    ------------
    PyTorch Dataset loader for unrolled longitudinal clinical trajectories.
    Features sample-independent dynamic backward block masking for Phase 1 JEPA.
    """
    def __init__(
        self, 
        preprocessed_csv_path: str, 
        max_seq_len: int = 256, 
        max_targets: int = 10,
        is_train: bool = True,   # Toggle dynamic random masking for training vs eval
        k_min: int = 2,          # Minimum target block length
        k_max: int = 6           # Maximum target block length
    ):
        self.df = pd.read_csv(preprocessed_csv_path).fillna("")
        self.max_seq_len = max_seq_len
        self.max_targets = max_targets
        self.is_train = is_train
        self.k_min = k_min
        self.k_max = k_max

    def __len__(self) -> int:
        return len(self.df)

    def _parse_sequence(self, str_val: Any, dtype: type, pad_value: Any, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = str(str_val).strip().split() if str(str_val).strip() else []
        tokens = [dtype(t) for t in tokens][:max_len]
        actual_len = len(tokens)
        
        padded = tokens + [pad_value] * (max_len - actual_len)
        mask = [False] * actual_len + [True] * (max_len - actual_len)
        
        return (
            torch.tensor(padded, dtype=torch.float32 if dtype == float else torch.long),
            torch.tensor(mask, dtype=torch.bool)
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # ─── 1. METADATA & DEMOGRAPHICS ───
        patient_session_id = str(row['mabn']) if 'mabn' in row else f"patient_{idx}"
        raw_cutoff = int(row['cutoff_idx']) if ('cutoff_idx' in row and str(row['cutoff_idx']) != "") else self.max_seq_len - 1
        
        # Guardrail: Guarantee cutoff is strictly inside [0, max_seq_len - 1]
        cutoff = min(max(0, raw_cutoff), self.max_seq_len - 1)

        age_tensor = torch.tensor(float(row['age']) if 'age' in row and str(row['age']) != "" else 0.0, dtype=torch.float32)
        gender_tensor = torch.tensor(int(row['gender']) if 'gender' in row and str(row['gender']) != "" else 0, dtype=torch.long)

        # ─── 2. PARSE SEQUENCES ───
        times, base_mask = self._parse_sequence(row['timestamps'], float, 0.0, self.max_seq_len)
        f_ids, _         = self._parse_sequence(row['feature_ids'], int, 0, self.max_seq_len)
        v_nums, _        = self._parse_sequence(row['numeric_values'], float, 0.0, self.max_seq_len)
        c_ids, _         = self._parse_sequence(row['cat_result_ids'], int, 0, self.max_seq_len)

        # ─── 3. SAMPLE-INDEPENDENT BACKWARD BLOCK MASKING ───
        if self.is_train and self.k_max > 1:
            k_sample = random.randint(self.k_min, self.k_max)
        else:
            k_sample = 1

        # Calculate backward target block boundaries from cutoff_idx
        # Safety Guardrail: max(1, ...) guarantees Student retains at least index 0
        target_start = max(1, cutoff - k_sample + 1)
        target_end = cutoff + 1  # Exclusive upper index for Python slicing

        # Student Mask: Censor from target_start to the end of the context window
        student_mask = base_mask.clone()
        student_mask[target_start:] = True

        # Teacher Mask: Unmask ONLY the target block [target_start : target_end]
        teacher_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        teacher_mask[target_start:target_end] = False

        # ─── 4. TARGET ICD CODES ───
        icd_ids, tgt_mask = self._parse_sequence(row['icd_targets'], int, 0, self.max_targets)

        return {
            'patient_session_id': patient_session_id,   
            'feature_ids': f_ids,                     # Shape: [max_seq_len] (LongTensor)
            'numeric_values': v_nums,                 # Shape: [max_seq_len] (FloatTensor)
            'cat_result_ids': c_ids,                  # Shape: [max_seq_len] (LongTensor)
            'timestamps': times,                      # Shape: [max_seq_len] (FloatTensor)
            'student_mask': student_mask,             # Shape: [max_seq_len] (BoolTensor)
            'teacher_mask': teacher_mask,             # Shape: [max_seq_len] (BoolTensor)
            'age': age_tensor,                        # Shape: [] Scalar (FloatTensor)
            'gender': gender_tensor,                  # Shape: [] Scalar (LongTensor)
            'icd_targets': icd_ids,                   # Shape: [max_targets] (LongTensor)
            'target_mask': tgt_mask                   # Shape: [max_targets] (BoolTensor)
        }


# ==================================================================================================
# 2. STATIC COHORT CLASS FREQUENCY CALCULATOR
# ==================================================================================================
def compute_static_class_frequencies(csv_path: str, num_classes: int = 456) -> torch.Tensor:
    r"""
    DESCRIPTION:
    ------------
    Scans the training CSV dataset to calculate normalized empirical class prevalence frequencies 
    \pi_c \in (0, 1] across all `num_classes` ICD targets.
    """
    df = pd.read_csv(csv_path).fillna("")
    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    for target_str in df['icd_targets']:
        tokens = str(target_str).strip().split()
        for t in tokens:
            try:
                class_id = int(t)
                if class_id < num_classes:
                    class_counts[class_id] += 1.0
            except ValueError:
                pass
                
    frequencies = class_counts / max(1, len(df))
    frequencies = np.clip(frequencies, 1e-5, 1.0)
    return torch.tensor(frequencies, dtype=torch.float32)