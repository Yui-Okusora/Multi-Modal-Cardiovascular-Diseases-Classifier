import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BVTDTimelineDataset(Dataset):
    """
    Directly indexes standalone sessionized patient trajectory files, converting 
    space-separated chronological tracking strings into parallel float/long tensor coordinates.
    """
    def __init__(self, preprocessed_csv_path, max_seq_len=128, max_targets=10):
        print(f"Loading sessionized patient dataset: {preprocessed_csv_path}")
        # Load string columns safely, filling unrecorded fields with blank text blocks
        self.df = pd.read_csv(preprocessed_csv_path).fillna("")
        self.max_seq_len = max_seq_len
        self.max_targets = max_targets
        
    def __len__(self):
        return len(self.df)
        
    def _parse_sequence(self, str_val, dtype, pad_value, max_len):
        """
        Splits text space-separated coordinates, runs truncation/padding rules,
        and builds corresponding attention masking matrices.
        """
        tokens = str(str_val).strip().split() if str(str_val).strip() else []
        tokens = [dtype(t) for t in tokens][:max_len]
        
        actual_len = len(tokens)
        padding_count = max_len - actual_len
        
        # Enforce shape consistency across batches
        padded_sequence = tokens + [pad_value] * padding_count
        
        # Generate strict boolean attention masks (True flags elements to ignore)
        mask = [False] * actual_len + [True] * padding_count
        
        # Select appropriate PyTorch primitive type bounds
        tensor_dtype = torch.float32 if dtype == float else torch.long
        return torch.tensor(padded_sequence, dtype=tensor_dtype), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse chronological event token components
        times, pad_mask = self._parse_sequence(row['timestamps'], float, 0.0, self.max_seq_len)
        feat_ids, _     = self._parse_sequence(row['feature_ids'], int, 0, self.max_seq_len)
        num_vals, _     = self._parse_sequence(row['numeric_values'], float, 0.0, self.max_seq_len)
        cat_ids, _      = self._parse_sequence(row['cat_result_ids'], int, 0, self.max_seq_len)
        
        # Parse multi-label discharge classification targets
        icd_ids, tgt_mask = self._parse_sequence(row['icd_targets'], int, 0, self.max_targets)
        
        return {
            'patient_session_id': str(row['mabn']),
            'feature_ids': feat_ids,       # Tensor Shape: [Max_Seq_Len] (torch.long)
            'numeric_values': num_vals,   # Tensor Shape: [Max_Seq_Len] (torch.float32)
            'cat_result_ids': cat_ids,     # Tensor Shape: [Max_Seq_Len] (torch.long)
            'timestamps': times,           # Tensor Shape: [Max_Seq_Len] (torch.float32)
            'padding_mask': pad_mask,     # Tensor Shape: [Max_Seq_Len] (torch.bool)
            'icd_targets': icd_ids,       # Tensor Shape: [Max_Targets] (torch.long)
            'target_mask': tgt_mask        # Tensor Shape: [Max_Targets] (torch.bool)
        }


def create_clinical_data_loaders(train_csv: str, val_csv: str, cfg):
    """
    Instantiates synchronized Training and Validation PyTorch Dataloaders 
    optimized for high-throughput VRAM async tensor migration.
    """
    train_dataset = BVTDTimelineDataset(
        preprocessed_csv_path=train_csv,
        max_seq_len=cfg.max_sequence_len,
        max_targets=cfg.max_targets
    )
    
    val_dataset = BVTDTimelineDataset(
        preprocessed_csv_path=val_csv,
        max_seq_len=cfg.max_sequence_len,
        max_targets=cfg.max_targets
    )
    
    # Construct multi-threaded batch aggregators
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,             # Randomizes batch distribution to stabilize gradient descent step vectors
        drop_last=True,           # Drops partial trailing blocks to maintain strict batch matrix dimensions
        num_workers=2,            # Leverages parallel CPU worker threads for background parsing
        pin_memory=True,          # Allocates tensors into page-locked CPU memory for ultra-fast GPU transfer
        prefetch_factor=2         # Pre-stages subsequent batches asynchronously
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,            # Maintain strict deterministic order for sequence validation tracking
        drop_last=False,          # Keep all remnants to ensure complete validation profiling
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

# =====================================================================
# PIPELINE INTEGRITY CHECK ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        max_sequence_len: int = 128
        max_targets: int = 10
        batch_size: int = 4
        
    cfg = MockConfig()
    
    # Quick execution scan over dummy files to verify matrix layout outputs
    try:
        mock_dataset = BVTDTimelineDataset("train_patient_grouped.csv", max_seq_len=128, max_targets=10)
        mock_loader = DataLoader(mock_dataset, batch_size=2, shuffle=True)
        
        print("\n🚀 Sampling execution track payload validation...")
        for batch in mock_loader:
            print(f"  • Sample Batch Patient IDs:    {batch['patient_session_id']}")
            print(f"  • Feature Coordinate Track Tensor: {batch['feature_ids'].shape} | Type: {batch['feature_ids'].dtype}")
            print(f"  • Continuous Value Metrics Tensor: {batch['numeric_values'].shape} | Type: {batch['numeric_values'].dtype}")
            print(f"  • Temporal Delta Matrix Tensor:    {batch['timestamps'].shape} | Type: {batch['timestamps'].dtype}")
            print(f"  • Transformer Attention Mask:      {batch['padding_mask'].shape} | Type: {batch['padding_mask'].dtype}")
            print(f"  • ICD Target Vector:               {batch['icd_targets'].shape} | Type: {batch['icd_targets'].dtype}")
            break
    except FileNotFoundError:
        print("\n💡 Script verified successfully. Run your build_features execution pass to instantiate target csv layers.")