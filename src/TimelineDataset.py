import torch
from torch.utils.data import Dataset
from datetime import datetime
import pandas as pd
import numpy as np

class BVTDTimelineDataset(Dataset):
    def __init__(self, cdha_df, xn_df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []  # Stores tuple of (cdha_row_dict, xn_row_dict)

        # 1. Standardize patient IDs
        cdha_df = cdha_df.copy()
        xn_df = xn_df.copy()

        cdha_df['maicd'] = cdha_df['maicd'].fillna("UNKNOWN_CODE").astype(str).str.strip()
        if 'maicd_label_id' not in cdha_df.columns:
            cdha_df['maicd_label_id'] = cdha_df['maicd'].astype('category').cat.codes
        self.num_classes = cdha_df['maicd_label_id'].nunique()

        cdha_df['mabn'] = cdha_df['mabn'].astype(str).str.strip()
        xn_df['mabn'] = xn_df['mabn'].astype(str).str.strip()

        # Print out a sample of the raw column so you can see exactly how it looks
        print(f"DEBUG: Raw XN date sample from file: {xn_df['ddmmyyyy'].dropna().iloc[0] if 'ddmmyyyy' in xn_df.columns and not xn_df['ddmmyyyy'].dropna().empty else 'COLUMN MISSING OR EMPTY'}")

        # 2. Parse CDHA dates (Month-Year format, e.g., '0626')
        cdha_df['mmyy'] = cdha_df['mmyy'].astype(str).str.strip().str.zfill(4)
        cdha_df['parsed_date'] = pd.to_datetime(cdha_df['mmyy'], format='%m%y', errors='coerce')

        # 3. FIX: Let Pandas adaptively parse format variations ('YYYY-MM-DD', 'DD/MM/YYYY', etc.)
        xn_df['parsed_date'] = pd.to_datetime(xn_df['ddmmyyyy'], errors='coerce', format='mixed')

        # --- Diagnostic Check ---
        valid_mabn = set(cdha_df['mabn']).intersection(set(xn_df['mabn']))
        print(f"DEBUG: Common patients found: {len(valid_mabn)}")
        print(f"DEBUG: Valid CDHA dates parsed: {cdha_df['parsed_date'].notna().sum()}/{len(cdha_df)}")
        print(f"DEBUG: Valid XN dates parsed: {xn_df['parsed_date'].notna().sum()}/{len(xn_df)}")
        # ------------------------
        
        # 3. Pre-compute valid pairs within the 60-day proximity window
        for mabn in valid_mabn:
            p_cdha = cdha_df[cdha_df['mabn'] == mabn]
            p_xn = xn_df[xn_df['mabn'] == mabn]
            
            for _, c_row in p_cdha.iterrows():
                if pd.isna(c_row['parsed_date']): continue
                
                best_xn = None
                min_days = 61  # Strict upper bound threshold
                
                for _, x_row in p_xn.iterrows():
                    if pd.isna(x_row['parsed_date']): continue
                    
                    diff_days = abs((c_row['parsed_date'] - x_row['parsed_date']).days)
                    if diff_days < min_days:
                        min_days = diff_days
                        best_xn = x_row
                
                # Only keep the pair if it strictly satisfies our temporal constraint
                if best_xn is not None:
                    # Convert to dictionaries for lightning-fast memory access later
                    self.pairs.append((c_row.to_dict(), best_xn.to_dict()))

        print(f"Dataset initialized with {len(self.pairs)} valid, temporally aligned pairs.")

    def __len__(self):
        return len(self.pairs)
    
    def _parse_vitals(self, xn_row):
        vitals = [0.0] * 6
        mask = [0.0] * 6

        # Define explicit clinical maximum boundaries for stable [0, 1] scaling
        CLINICAL_BOUNDS = {
            'sbp': 200.0,       # Systolic Blood Pressure
            'dbp': 130.0,       # Diastolic Blood Pressure
            'mach': 160.0,      # Heart Rate / Pulse
            'nhietdo': 45.0,    # Body Temperature
            'cannang': 150.0,   # Weight (kg)
            'chieucao': 200.0   # Height (cm)
        }

        hp = str(xn_row.get('huyetap', ''))
        if '/' in hp:
            try:
                sbp, dbp = map(float, hp.split('/'))
                # Scale bounded metrics safely between 0.0 and 1.0
                vitals[0] = clamp(sbp / CLINICAL_BOUNDS['sbp'], 0.0, 1.0)
                vitals[1] = clamp(dbp / CLINICAL_BOUNDS['dbp'], 0.0, 1.0)
                mask[0], mask[1] = 1.0, 1.0
            except ValueError:
                pass

        # Helper to prevent extreme outliers from exceeding 1.0 bounds
        def clamp(n, minn, maxn):
            return max(min(n, maxn), minn)

        # Process remaining features with explicit scalar bounds
        fields = ['mach', 'nhietdo', 'cannang', 'chieucao']
        bounds_keys = ['mach', 'nhietdo', 'cannang', 'chieucao']

        for i, (field, key) in enumerate(zip(fields, bounds_keys), start=2):
            val = xn_row.get(field)
            if val is not None and not pd.isna(val):
                try:
                    raw_float = float(val)
                    # Scale cleanly relative to clinical limits
                    vitals[i] = clamp(raw_float / CLINICAL_BOUNDS[key], 0.0, 1.0)
                    mask[i] = 1.0
                except ValueError:
                    pass
        
        return torch.tensor(vitals + mask, dtype=torch.float32)

    def __getitem__(self, idx):
        # Instant lookup: No loops, no dataframes, no groupings!
        cdha_sel, xn_sel = self.pairs[idx]
        
        # Tokenize text reports
        text_input = f"Chẩn đoán: {cdha_sel['chandoan']}. Kết luận: {cdha_sel['ketluan']}"
        tokens = self.tokenizer(
            text_input, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Build features
        vitals_tensor = self._parse_vitals(xn_sel)
        
        # Extract target disease label (assuming maicd is integer-coded)
        # If maicd lives in xn_df instead, change cdha_sel to xn_sel
        icd_label = torch.tensor(int(cdha_sel['maicd_label_id']), dtype=torch.long)
        
        # Return exact naming conventions expected by the T-JEPA network pipeline
        return {
            'vitals_12d': vitals_tensor,                         # Input context feature
            'input_ids': tokens['input_ids'].squeeze(0),         # Target language tracking
            'attention_mask': tokens['attention_mask'].squeeze(0), # Target mask tracking
            'maicd_label_id': icd_label                            # Downstream evaluation label
        }