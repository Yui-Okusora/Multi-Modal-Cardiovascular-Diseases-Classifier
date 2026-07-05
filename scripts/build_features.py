import pandas as pd
import numpy as np
from collections import Counter
import re
import json

CLINICAL_BOUNDS = {
    'sbp': (70.0, 200.0),       
    'dbp': (40.0, 130.0),       
    'mach': (30.0, 160.0),      
    'nhietdo': (35.0, 42.0),    
    'cannang': (30.0, 150.0),   
    'chieucao': (100.0, 200.0)  
}

def clean_and_parse_numeric(val_str):
    if pd.isna(val_str):
        return None
    cleaned = str(val_str).strip().replace(',', '.')
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    if match:
        return float(match.group())
    return None

def build_systemic_vocabularies(xn_df):
    """
    Builds global structural feature indices and categorical result maps.
    Reserves slots 0-7 for static context markers and basic vital scales.
    """
    feature_codebook = {
        'sbp': 0, 'dbp': 1, 'mach': 2, 'nhietdo': 3, 'cannang': 4, 'chieucao': 5,
        'tuoi': 6, 'phai': 7
    }
    
    unique_labs = xn_df['tenxn'].dropna().unique()
    for idx, lab_name in enumerate(unique_labs, start=8):
        normalized_key = str(lab_name).strip().lower()
        if normalized_key not in feature_codebook:
            feature_codebook[normalized_key] = len(feature_codebook)
        
    cat_result_vocab = {
        '[NUMERIC_ONLY]': 0,
        'nam': 1,
        'nu': 2
    }
    cat_idx = 3
    
    for val in xn_df['ketqua'].dropna().unique():
        val_str = str(val).strip().lower()  # 🎯 FIX: Lowercase categorical definitions too
        if clean_and_parse_numeric(val_str) is None and val_str != "":
            if val_str not in cat_result_vocab:
                cat_result_vocab[val_str] = cat_idx
                cat_idx += 1
                
    return feature_codebook, cat_result_vocab

if __name__ == "__main__":
    print("=== Launching Anonymized Sessionized Ingestion Pipeline ===")
    
    cdha_df = pd.read_csv("master_cdha_cleaned.csv", dtype=str)
    xn_df = pd.read_csv("master_xn_cleaned.csv", dtype=str)
    
    cdha_df['parsed_date'] = pd.to_datetime(cdha_df['mmyy'].astype(str).str.zfill(4), format='%m%y', errors='coerce')
    xn_df['parsed_date'] = pd.to_datetime(xn_df['ddmmyyyy'], errors='coerce', format='mixed')
    
    # Primary & Secondary Key Spatial Sorting to force contiguous block arrays
    print(" Executing multi-key sorting matrices...")
    cdha_df = cdha_df.dropna(subset=['mabn', 'parsed_date']).sort_values(by=['mabn', 'parsed_date']).reset_index(drop=True)
    xn_df = xn_df.dropna(subset=['mabn', 'parsed_date']).sort_values(by=['mabn', 'parsed_date']).reset_index(drop=True)
    
    # Compile dictionaries
    feature_codebook, cat_result_vocab = build_systemic_vocabularies(xn_df)
    
    cdha_df['maicd'] = cdha_df['maicd'].fillna("UNKNOWN").astype(str).str.strip()
    all_icd_classes = sorted(cdha_df['maicd'].unique())
    icd_codebook = {code: idx for idx, code in enumerate(all_icd_classes)}
    
    # Isolate valid overlapping cohort
    all_patients = sorted(list(set(cdha_df['mabn']).intersection(set(xn_df['mabn']))))
    
    # THE CENSOR LAYER: MAP EXACT MABN STRINGS TO A 1-BASED INDEX ARRAY
    print(" Instantiating data masking dictionaries (1-based index mapping)...")
    mabn_anonymizer = {raw_id: idx for idx, raw_id in enumerate(all_patients, start=1)}
    
    # Stratify cross-validation boundaries at patient level using randomized shuffles
    np.random.seed(42)
    np.random.shuffle(all_patients)
    split = int(len(all_patients) * 0.8)
    train_mabns = set(all_patients[:split])
    
    MAX_SEQ_LEN = 128  
    train_rows, val_rows = [], []
    
    print(" Digesting patient tracking lines into anonymized sessionized sequences...")
    xn_grouped = xn_df.groupby('mabn', sort=False)
    
    for mabn, p_cdha in cdha_df.groupby('mabn', sort=False):
        if mabn not in xn_grouped.groups: continue
        p_xn = xn_grouped.get_group(mabn)
        is_train = mabn in train_mabns
        
        # Extract the dynamic censored index ID for the active patient
        censored_mabn_id = mabn_anonymizer[mabn]
        
        # GROUP BY DATE TO COMPRESS MULTI-ROW DAILY LOGS INTO A SINGLE TIMELINE
        for tgt_date, session_df in p_cdha.groupby('parsed_date', sort=False):
            
            # Extract historical labs occurring within a strict 30-day window leading up to THIS report
            valid_history = p_xn[(p_xn['parsed_date'] <= tgt_date) & 
                                 ((tgt_date - p_xn['parsed_date']).dt.days <= 90)]
            
            if valid_history.empty: continue
            
            # Establish localized Day-Zero baseline for THIS session using its earliest recorded row
            encounter_base_date = valid_history.iloc[0]['parsed_date']
            
            # Resolve Demographic Static Context Blocks from the first row of this day's session slice
            c_row = session_df.iloc[0]
            raw_age = clean_and_parse_numeric(c_row.get('tuoi', 0)) or 0.0
            normalized_age = max(min(raw_age / 100.0, 1.0), 0.0)
            
            # Aligned explicitly with your schema's 'phai' column header
            raw_gender = str(c_row.get('phai', '')).strip().lower()
            gender_key = 'nam' if raw_gender in ['nam', 'm', 'male', '1'] else 'nu'
            gender_cat_id = cat_result_vocab.get(gender_key, 0)
            
            session_events = [
                (0.0, feature_codebook['tuoi'], float(normalized_age), 0),
                (0.0, feature_codebook['phai'], 0.0, int(gender_cat_id))
            ]
            
            # Ingest active indicators sequentially
            for _, x_row in valid_history.iterrows():
                # Compute relative hours from session base date
                elapsed_hours = float((x_row['parsed_date'] - encounter_base_date).total_seconds() / 3600.0)
                
                # Parse blood pressure metrics with strict winsorization clipping bounds
                hp = str(x_row.get('huyetap', ''))
                if '/' in hp:
                    try:
                        s_str, d_str = hp.split('/')
                        s_num, d_num = clean_and_parse_numeric(s_str), clean_and_parse_numeric(d_str)
                        if s_num:
                            clipped_s = float(max(min(s_num, CLINICAL_BOUNDS['sbp'][1]), CLINICAL_BOUNDS['sbp'][0]))
                            session_events.append((elapsed_hours, feature_codebook['sbp'], clipped_s, 0))
                        if d_num:
                            clipped_d = float(max(min(d_num, CLINICAL_BOUNDS['dbp'][1]), CLINICAL_BOUNDS['dbp'][0]))
                            session_events.append((elapsed_hours, feature_codebook['dbp'], clipped_d, 0))
                    except ValueError: pass

                # Preserving vital signals through bounded value mapping
                for field in ['mach', 'nhietdo', 'cannang', 'chieucao']:
                    v_num = clean_and_parse_numeric(x_row.get(field))
                    if v_num:
                        clipped_v = float(max(min(v_num, CLINICAL_BOUNDS[field][1]), CLINICAL_BOUNDS[field][0]))
                        session_events.append((elapsed_hours, feature_codebook[field], clipped_v, 0))
                        
                # Parse Laboratory parameters
                lab_name = str(x_row.get('tenxn', '')).strip().lower()
                if lab_name in feature_codebook:
                    f_id = feature_codebook[lab_name]
                    res_str = str(x_row.get('ketqua', '')).strip().lower()
                    num_parsed = clean_and_parse_numeric(res_str)
                    
                    if num_parsed is not None:
                        session_events.append((elapsed_hours, f_id, float(num_parsed), 0))
                    elif res_str in cat_result_vocab:
                        session_events.append((elapsed_hours, f_id, 0.0, cat_result_vocab[res_str]))
                        
            if len(session_events) <= 2: continue
            
            # Slice trailing items to fill remaining sequence budget
            static_block = session_events[:2]
            latest_dynamic_block = session_events[2:][-(MAX_SEQ_LEN - 2):]
            final_timeline = static_block + latest_dynamic_block
            
            # Gather all unique target codes compiled across this day's session_df records
            encounter_codes = session_df['maicd'].unique()
            
            # Expanded to allow the full target landscape to flow unhindered into downstream metrics
            icd_ids = [icd_codebook[code] for code in encounter_codes if code in icd_codebook]
            if not icd_ids: continue
            
            # Assemble record with completely redacted and structured patient identifier
            record = {
                'mabn': f"patient_{censored_mabn_id}_{int(tgt_date.timestamp())}",
                'timestamps': " ".join([str(e[0]) for e in final_timeline]),
                'feature_ids': " ".join([str(e[1]) for e in final_timeline]),
                'numeric_values': " ".join([str(e[2]) for e in final_timeline]),
                'cat_result_ids': " ".join([str(e[3]) for e in final_timeline]),
                'icd_targets': " ".join([str(i) for i in icd_ids])
            }
            
            if is_train: train_rows.append(record)
            else: val_rows.append(record)
            
    # Commit sanitized csv assets to disk
    pd.DataFrame(train_rows).to_csv("train_patient_grouped.csv", index=False)
    pd.DataFrame(val_rows).to_csv("val_patient_grouped.csv", index=False)
    
    # Save synchronized codebook lookup registries
    master_codebooks = {
        "metadata": {
            "num_total_features": len(feature_codebook), 
            "num_cat_results": len(cat_result_vocab), 
            "num_icd_classes": len(icd_codebook)
        },
        "forward_maps": {
            "features": feature_codebook, 
            "categorical_results": cat_result_vocab, 
            "icd_codes": icd_codebook
        },
        "inverse_maps": {str(v): k for k, v in feature_codebook.items()},
        "inverse_categorical_results": {str(v): k for k, v in cat_result_vocab.items()},
        "inverse_icd_codes": {str(v): k for k, v in icd_codebook.items()}
    }
    
    with open("clinical_codebooks.json", "w", encoding="utf-8") as f:
        json.dump(master_codebooks, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Success. Anonymized Data Tracks Compiled: Train={len(train_rows)} | Val={len(val_rows)}")