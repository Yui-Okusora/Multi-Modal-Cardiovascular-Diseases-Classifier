# src/TimelineDataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from typing import Dict, Any, Tuple


# ==================================================================================================
# 1. PRE-FLATTENED LONGITUDINAL TRAJECTORY DATASET LOADER (3D TENSOR SUPPORT)
# ==================================================================================================
class BVTDFlattenedDataset(Dataset):
    """
    DESCRIPTION:
    ------------
    PyTorch Dataset loader for unrolled longitudinal clinical trajectories.
    Features worker-safe, sample-independent dynamic stochastic dual-masking 
    (50% Causal Tail Forecasting vs. 50% Random Span In-Filling) for Phase 1 JEPA.
    """
    def __init__(
        self, 
        preprocessed_csv_path: str, 
        max_seq_len: int = 256, 
        max_subwords: int = 16,
        max_targets: int = 10,
        is_train: bool = True,
        k_min: int = 2,
        k_max: int = 6
    ):
        self.df = pd.read_csv(preprocessed_csv_path).fillna("")
        self.max_seq_len = max_seq_len
        self.max_subwords = max_subwords
        self.max_targets = max_targets
        self.is_train = is_train
        self.k_min = k_min
        self.k_max = k_max

    def __len__(self) -> int:
        return len(self.df)

    def _parse_1d_sequence(self, str_val: Any, dtype: type, pad_value: Any, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = str(str_val).strip().split() if str(str_val).strip() else []
        tokens = [dtype(t) for t in tokens][:max_len]
        actual_len = len(tokens)
        
        padded = tokens + [pad_value] * (max_len - actual_len)
        mask = [False] * actual_len + [True] * (max_len - actual_len)
        
        return (
            torch.tensor(padded, dtype=torch.float32 if dtype == float else torch.long),
            torch.tensor(mask, dtype=torch.bool)
        )

    def _parse_3d_cat_sequence(self, str_val: Any, max_len: int, max_subwords: int) -> torch.Tensor:
        blocks = str(str_val).strip().split() if str(str_val).strip() else []
        blocks = blocks[:max_len]
        actual_len = len(blocks)
        
        parsed_2d = []
        for block in blocks:
            subwords = [int(s) for s in block.split(',') if str(s).strip() != ""][:max_subwords]
            if len(subwords) < max_subwords:
                subwords += [0] * (max_subwords - len(subwords))
            parsed_2d.append(subwords)
            
        for _ in range(max_len - actual_len):
            parsed_2d.append([0] * max_subwords)
            
        return torch.tensor(parsed_2d, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        patient_session_id = str(row['mabn']) if 'mabn' in row else f"patient_{idx}"
        raw_cutoff = int(row['cutoff_idx']) if ('cutoff_idx' in row and str(row['cutoff_idx']) != "") else self.max_seq_len - 1
        cutoff = min(max(0, raw_cutoff), self.max_seq_len - 1)
        active_len = cutoff + 1

        age_tensor = torch.tensor(float(row['age']) if 'age' in row and str(row['age']) != "" else 0.0, dtype=torch.float32)
        gender_tensor = torch.tensor(int(row['gender']) if 'gender' in row and str(row['gender']) != "" else 0, dtype=torch.long)

        # ─── PARSE 1D AND 3D SEQUENCES ───
        times, base_mask = self._parse_1d_sequence(row['timestamps'], float, 0.0, self.max_seq_len)
        f_ids, _         = self._parse_1d_sequence(row['feature_ids'], int, 0, self.max_seq_len)
        v_nums, _        = self._parse_1d_sequence(row['numeric_values'], float, 0.0, self.max_seq_len)
        c_ids_3d         = self._parse_3d_cat_sequence(row['cat_result_ids'], self.max_seq_len, self.max_subwords)

        # ─── 🚀 WORKER-SAFE DYNAMIC BLOCK LENGTH SAMPLING ───
        if self.is_train and self.k_max > self.k_min:
            k_sample = int(torch.randint(low=self.k_min, high=self.k_max + 1, size=(1,)).item())
        else:
            k_sample = self.k_min if self.is_train else 1

        # Fallback for single-event timelines (cutoff == 0)
        if cutoff == 0:
            student_mask = base_mask.clone()
            teacher_mask = base_mask.clone()
        else:
            # Strictly mask the target horizon block [target_start : active_len] at the tail
            target_start = max(1, cutoff - k_sample + 1)
            target_end = active_len

            # Student sees strictly past history BEFORE target_start
            student_mask = base_mask.clone()
            student_mask[target_start:] = True

            # Teacher sees strictly the future target horizon block
            teacher_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            teacher_mask[target_start:target_end] = base_mask[target_start:target_end]

        icd_ids, tgt_mask = self._parse_1d_sequence(row['icd_targets'], int, 0, self.max_targets)

        return {
            'patient_session_id': patient_session_id,   
            'feature_ids': f_ids,                     # Shape: [max_seq_len] (LongTensor)
            'numeric_values': v_nums,                 # Shape: [max_seq_len] (FloatTensor)
            'cat_result_ids': c_ids_3d,               # Shape: [max_seq_len, max_subwords] (LongTensor)
            'timestamps': times,                      # Shape: [max_seq_len] (FloatTensor)
            'base_mask': base_mask,                   # Shape: [max_seq_len] (BoolTensor)
            'student_mask': student_mask,             # Shape: [max_seq_len] (BoolTensor)
            'teacher_mask': teacher_mask,             # Shape: [max_seq_len] (BoolTensor)
            'age': age_tensor,                        # Shape: [] Scalar (FloatTensor)
            'gender': gender_tensor,                  # Shape: [] Scalar (LongTensor)
            'icd_targets': icd_ids,                   # Shape: [max_targets] (LongTensor)
            'target_mask': tgt_mask                   # Shape: [max_targets] (BoolTensor)
        }


# ==================================================================================================
# 2. DATALOADER WORKER SEEDING UTILITY
# ==================================================================================================
def seed_worker(worker_id: int) -> None:
    """
    Ensures every PyTorch DataLoader worker process initializes with a unique, 
    sample-independent random seed for true per-sample dynamic masking.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================================================================================================
# 3. STATIC COHORT CLASS FREQUENCY CALCULATOR
# ==================================================================================================
def compute_static_class_frequencies(csv_path: str, num_classes: int = 456) -> torch.Tensor:
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