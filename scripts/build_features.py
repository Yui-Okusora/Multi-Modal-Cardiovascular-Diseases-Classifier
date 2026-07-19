# build_features.py
import pandas as pd
import numpy as np
import re
import json
from config import CardioConfig

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

CLINICAL_STOP_WORDS = {
    'triển', 'khai', 'thí', 'điểm', 'không', 'in', 'phim', 'theo', 'đề', 'án', 'byt',
    'và', 'các', 'của', 'tại', 'khoa', 'đề_án', 'thí_điểm'
}

def clean_and_tokenize_text(text_str):
    if pd.isna(text_str): return []
    normalized = str(text_str).strip().lower()
    normalized = re.sub(r'[\\/\\\n\t.,;:()\[\]\-#?+*!]', ' ', normalized)
    words = [w.strip() for w in normalized.split() if w.strip()]
    return [w for w in words if w not in CLINICAL_STOP_WORDS]

def build_unified_vocabularies(xn_df, cdha_df):
    feature_codebook = {
        'sbp': 0, 'dbp': 1, 'mach': 2, 'nhietdo': 3, 'cannang': 4, 'chieucao': 5,
        'tuoi': 6, 'phai': 7
    }
    
    unique_labs = xn_df['tenxn'].dropna().unique()
    for lab_name in unique_labs:
        normalized_key = str(lab_name).strip().lower()
        if normalized_key not in feature_codebook:
            feature_codebook[normalized_key] = len(feature_codebook)
            
    unique_modalities = cdha_df['kythuatcdha'].dropna().unique()
    for technique in unique_modalities:
        normalized_tech = str(technique).strip().lower()
        if normalized_tech not in feature_codebook:
            feature_codebook[normalized_tech] = len(feature_codebook)
        
    cat_result_vocab = {
        '[NUMERIC_ONLY]': 0, 'nam': 1, 'nu': 2
    }
    cat_idx = 3
    
    for val in xn_df['ketqua'].dropna().unique():
        val_str = str(val).strip().lower()
        if clean_and_parse_numeric(val_str) is None and val_str != "":
            if val_str not in cat_result_vocab:
                cat_result_vocab[val_str] = cat_idx
                cat_idx += 1
                
    for text_block in cdha_df['ketluan'].dropna().unique():
        words = clean_and_tokenize_text(text_block)
        for word in words:
            if word not in cat_result_vocab:
                cat_result_vocab[word] = cat_idx
                cat_idx += 1
                
    return feature_codebook, cat_result_vocab

def greedy_multilabel_stratification(patients, patient_labels, train_ratio=0.8):
    num_patients = len(patients)
    assignments = np.zeros(num_patients, dtype=int)
    class_counts = patient_labels.sum(axis=0)
    sorted_classes = np.argsort(class_counts)
    
    target_train = int(num_patients * train_ratio)
    target_val = num_patients - target_train
    c_train, c_val = 0, 0
    
    for c_idx in sorted_classes:
        match_pats = np.where((patient_labels[:, c_idx] == 1) & (assignments == 0))[0]
        if len(match_pats) == 0: continue
        np.random.shuffle(match_pats)
        
        for p_idx in match_pats:
            if (c_train / max(1, target_train)) <= (c_val / max(1, target_val)):
                assignments[p_idx] = 1
                c_train += 1
            else:
                assignments[p_idx] = 2
                c_val += 1
                
    unassigned = np.where(assignments == 0)[0]
    np.random.shuffle(unassigned)
    for p_idx in unassigned:
        if c_train < target_train:
            assignments[p_idx] = 1
            c_train += 1
        else:
            assignments[p_idx] = 2
            c_val += 1
            
    return {patients[i] for i in range(num_patients) if assignments[i] == 1}

if __name__ == "__main__":
    print("=== Launching Offline Dataset Trajectory Unrolling Pipeline ===")
    
    cdha_df = pd.read_csv("master_cdha_cleaned.csv", dtype=str)
    xn_df = pd.read_csv("master_xn_cleaned.csv", dtype=str)
    
    cdha_df['parsed_date'] = pd.to_datetime(cdha_df['mmyy'].astype(str).str.zfill(4), format='%m%y', errors='coerce')
    xn_df['parsed_date'] = pd.to_datetime(xn_df['ddmmyyyy'], errors='coerce', format='mixed')
    
    cdha_df = cdha_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    xn_df = xn_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    
    feature_codebook, cat_result_vocab = build_unified_vocabularies(xn_df, cdha_df)
    
    cdha_df['maicd'] = cdha_df['maicd'].fillna("UNKNOWN").astype(str).str.strip()
    all_icd_classes = sorted(cdha_df['maicd'].unique())
    icd_codebook = {code: idx for idx, code in enumerate(all_icd_classes)}
    
    all_patients = sorted(list(set(cdha_df['mabn']).intersection(set(xn_df['mabn']))))
    mabn_anonymizer = {raw_id: idx for idx, raw_id in enumerate(all_patients, start=1)}
    
    num_patients = len(all_patients)
    patient_labels = np.zeros((num_patients, len(icd_codebook)), dtype=np.float32)
    p_icd_map = cdha_df.groupby('mabn')['maicd'].apply(set).to_dict()
    
    for idx, p in enumerate(all_patients):
        for code in p_icd_map.get(p, set()):
            if code in icd_codebook:
                patient_labels[idx, icd_codebook[code]] = 1.0
                
    np.random.seed(42)
    train_mabns = greedy_multilabel_stratification(all_patients, patient_labels, train_ratio=0.65)

    cfg = CardioConfig()
    MAX_SEQ_LEN = cfg.max_sequence_len  
    train_flattened_rows, val_flattened_rows = [], []
    
    train_patients_scanned, val_patients_scanned = 0, 0
    xn_grouped = xn_df.groupby('mabn', sort=False)
    
    print("⏳ Unrolling patient trajectories into pre-computed step slices...")
    for mabn, p_cdha in cdha_df.groupby('mabn', sort=False):
        if mabn not in xn_grouped.groups: continue
        p_xn = xn_grouped.get_group(mabn)
        is_train = mabn in train_mabns
        censored_mabn_id = mabn_anonymizer[mabn]
        
        if is_train: train_patients_scanned += 1
        else: val_patients_scanned += 1
        
        raw_interleaved_events = []
        
        first_cdha_row = p_cdha.iloc[0]
        raw_age = clean_and_parse_numeric(first_cdha_row.get('tuoi', 0)) or 0.0
        normalized_age = max(min(raw_age / 100.0, 1.0), 0.0)
        
        raw_gender = str(first_cdha_row.get('phai', '')).strip().lower()
        gender_key = 'nam' if raw_gender in ['nam', 'm', 'male', '1'] else 'nu'
        gender_cat_id = cat_result_vocab.get(gender_key, 0)
        
        # Ingest Laboratory Tracks
        for _, x_row in p_xn.iterrows():
            evt_date = x_row['parsed_date']
            hp = str(x_row.get('huyetap', ''))
            if '/' in hp:
                try:
                    s_str, d_str = hp.split('/')
                    s_num, d_num = clean_and_parse_numeric(s_str), clean_and_parse_numeric(d_str)
                    if s_num:
                        clipped_s = float(max(min(s_num, CLINICAL_BOUNDS['sbp'][1]), CLINICAL_BOUNDS['sbp'][0]))
                        raw_interleaved_events.append((evt_date, feature_codebook['sbp'], clipped_s, 0))
                    if d_num:
                        clipped_d = float(max(min(d_num, CLINICAL_BOUNDS['dbp'][1]), CLINICAL_BOUNDS['dbp'][0]))
                        raw_interleaved_events.append((evt_date, feature_codebook['dbp'], clipped_d, 0))
                except ValueError: pass

            for field in ['mach', 'nhietdo', 'cannang', 'chieucao']:
                v_num = clean_and_parse_numeric(x_row.get(field))
                if v_num:
                    clipped_v = float(max(min(v_num, CLINICAL_BOUNDS[field][1]), CLINICAL_BOUNDS[field][0]))
                    raw_interleaved_events.append((evt_date, feature_codebook[field], clipped_v, 0))
                    
            lab_name = str(x_row.get('tenxn', '')).strip().lower()
            if lab_name in feature_codebook:
                f_id = feature_codebook[lab_name]
                res_str = str(x_row.get('ketqua', '')).strip().lower()
                num_parsed = clean_and_parse_numeric(res_str)
                if num_parsed is not None:
                    raw_interleaved_events.append((evt_date, f_id, float(num_parsed), 0))
                elif res_str in cat_result_vocab:
                    raw_interleaved_events.append((evt_date, f_id, 0.0, cat_result_vocab[res_str]))

        # Ingest Diagnostic Report Tracks
        for _, c_row in p_cdha.iterrows():
            evt_date = c_row['parsed_date']
            technique = str(c_row.get('kythuatcdha', '')).strip().lower()
            text_summary = str(c_row.get('ketluan', ''))
            
            if technique in feature_codebook:
                f_id = feature_codebook[technique]
                ef_match = re.search(r"ef\s*=\s*(\d+)", text_summary.lower())
                extracted_numeric = float(ef_match.group(1)) if ef_match else 0.0
                
                words = clean_and_tokenize_text(text_summary)
                if words:
                    for word in words:
                        raw_interleaved_events.append((evt_date, f_id, extracted_numeric, cat_result_vocab.get(word, 0)))
                else:
                    raw_interleaved_events.append((evt_date, f_id, extracted_numeric, 0))

        if not raw_interleaved_events: continue
        raw_interleaved_events.sort(key=lambda x: x[0])
        
        encounter_codes = p_cdha['maicd'].unique()
        icd_ids = [icd_codebook[code] for code in encounter_codes if code in icd_codebook]
        if not icd_ids: continue

        # 🚀 REFACTORED DIRECTIONAL INFRASTRUCTURE PATHS
        if is_train:
            for step_idx in range(1, len(raw_interleaved_events)):
                cutoff_date = raw_interleaved_events[step_idx][0]
                active_history_raw = raw_interleaved_events[:step_idx + 1]
                
                if len(active_history_raw) > MAX_SEQ_LEN:
                    active_history_raw = active_history_raw[-MAX_SEQ_LEN:]
                    
                # ⏳ COMPUTE REVERSED LOOKBACK DURATIONS (Target step acts as 0.0 anchor)
                final_timeline = []
                for evt_date, f_id, v_num, c_id in active_history_raw:
                    lookback_hours = float((cutoff_date - evt_date).total_seconds() / 3600.0)
                    final_timeline.append((lookback_hours, f_id, v_num, c_id))
                
                record = {
                    'mabn': f"patient_{censored_mabn_id}_step_{step_idx}",
                    'cutoff_idx': len(final_timeline) - 1, # Direct absolute coordinate matching
                    'age': float(normalized_age),          # Explicit DataFrame extraction
                    'gender': int(gender_cat_id),          # Explicit DataFrame extraction
                    'timestamps': " ".join([str(e[0]) for e in final_timeline]),
                    'feature_ids': " ".join([str(e[1]) for e in final_timeline]),
                    'numeric_values': " ".join([str(e[2]) for e in final_timeline]),
                    'cat_result_ids': " ".join([str(e[3]) for e in final_timeline]),
                    'icd_targets': " ".join([str(i) for i in icd_ids])
                }
                train_flattened_rows.append(record)
        else:
            cutoff_date = raw_interleaved_events[-1][0]
            active_history_raw = raw_interleaved_events

            if len(active_history_raw) > MAX_SEQ_LEN:
                active_history_raw = active_history_raw[-MAX_SEQ_LEN:]
                
            final_timeline = []
            for evt_date, f_id, v_num, c_id in active_history_raw:
                lookback_hours = float((cutoff_date - evt_date).total_seconds() / 3600.0)
                final_timeline.append((lookback_hours, f_id, v_num, c_id))
            
            record = {
                'mabn': f"patient_{censored_mabn_id}_terminal",
                'cutoff_idx': len(final_timeline) - 1,
                'age': float(normalized_age),
                'gender': int(gender_cat_id),
                'timestamps': " ".join([str(e[0]) for e in final_timeline]),
                'feature_ids': " ".join([str(e[1]) for e in final_timeline]),
                'numeric_values': " ".join([str(e[2]) for e in final_timeline]),
                'cat_result_ids': " ".join([str(e[3]) for e in final_timeline]),
                'icd_targets': " ".join([str(i) for i in icd_ids])
            }
            val_flattened_rows.append(record)

    pd.DataFrame(train_flattened_rows).to_csv(cfg.train_csv_path, index=False)
    pd.DataFrame(val_flattened_rows).to_csv(cfg.val_csv_path, index=False)
    
    master_codebooks = {
        "metadata": {
            "num_total_features": len(feature_codebook), 
            "num_cat_results": len(cat_result_vocab), 
            "num_icd_classes": len(icd_codebook)
        },
        "forward_maps": {
            "features": feature_codebook, "categorical_results": cat_result_vocab, "icd_codes": icd_codebook
        },
        "inverse_maps": {str(v): k for k, v in feature_codebook.items()},
        "inverse_categorical_results": {str(v): k for k, v in cat_result_vocab.items()},
        "inverse_icd_codes": {str(v): k for k, v in icd_codebook.items()}
    }
    
    with open(cfg.codebook_json_path, "w", encoding="utf-8") as f:
        json.dump(master_codebooks, f, indent=4, ensure_ascii=False)
        
    print("\n" + "═"*80)
    print(" 📊 OFFLINE STRATIFIED TRAJECTORY UNROLLING COMPILATION REPORT")
    print("═"*80)
    print(f" 📑 TRAINING COHORT CONFIGURATION:")
    print(f"   • Raw Patient Timelines Scanned:      {train_patients_scanned:,} cases")
    print(f"   • Active Pre-Flattened Slices Written: {len(train_flattened_rows):,} samples")
    print("-" * 80)
    print(f" 📑 VALIDATION COHORT CONFIGURATION:")
    print(f"   • Raw Patient Timelines Scanned:      {val_patients_scanned:,} cases")
    print(f"   • Active Pre-Flattened Slices Written: {len(val_flattened_rows):,} samples")
    print("═"*80 + "\n")