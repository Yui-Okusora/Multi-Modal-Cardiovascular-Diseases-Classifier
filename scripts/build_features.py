# scripts/build_features.py
r"""
====================================================================================================
CHRONOS-JEPA OFFLINE DATASET TRAJECTORY UNROLLING & BPE TOKENIZATION PIPELINE
====================================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from config import CardioConfig
from src.Utils import clean_and_parse_numeric, clean_and_tokenize_text


# ─── 1. BPE TOKENIZER TRAINER / LOADER ───
def train_or_load_bpe_tokenizer(
    corpus_texts: List[str], 
    json_path: str, 
    vocab_size: int
) -> Tokenizer:
    """
    Trains or loads a custom Byte-Pair Encoding (BPE) subword tokenizer 
    directly on medical lab outcomes and diagnostic report texts.
    """
    if os.path.exists(json_path):
        print(f"📥 Loading existing medical BPE Tokenizer from -> {json_path}")
        return Tokenizer.from_file(json_path)
    
    print(f"⚙️ Training custom medical BPE Tokenizer (Target Vocab Size = {vocab_size})...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]"]
    )
    
    valid_corpus = [t for t in corpus_texts if isinstance(t, str) and t.strip()]
    tokenizer.train_from_iterator(valid_corpus, trainer)
    tokenizer.save(json_path)
    print(f"💾 Saved trained BPE Tokenizer artifact cleanly to -> {json_path}")
    return tokenizer


def encode_text_to_subwords(tokenizer: Tokenizer, text: str, max_subwords: int) -> List[int]:
    """
    Encodes clinical text into a fixed-width subword array padded/truncated with ID 0.
    """
    if not isinstance(text, str) or not text.strip():
        return [0] * max_subwords
        
    encoded = tokenizer.encode(text.strip().lower())
    subword_ids = encoded.ids[:max_subwords]
    if len(subword_ids) < max_subwords:
        subword_ids += [0] * (max_subwords - len(subword_ids))
    return subword_ids


# ─── 2. VOCABULARY BUILDER ───
def build_feature_codebook(
    xn_df: pd.DataFrame, 
    cdha_df: pd.DataFrame
) -> Dict[str, int]:
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
            
    return feature_codebook


# ─── 3. GREEDY MULTI-LABEL STRATIFICATION ───
def greedy_multilabel_stratification(
    patients: List[str], 
    patient_labels: np.ndarray, 
    train_ratio: float = 0.65
) -> set:
    num_patients = len(patients)
    assignments = np.zeros(num_patients, dtype=int)
    class_counts = patient_labels.sum(axis=0)
    sorted_classes = np.argsort(class_counts)
    
    target_train = int(num_patients * train_ratio)
    target_val = num_patients - target_train
    c_train, c_val = 0, 0
    
    for c_idx in sorted_classes:
        match_pats = np.where((patient_labels[:, c_idx] == 1) & (assignments == 0))[0]
        if len(match_pats) == 0:
            continue
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


# ─── 4. TIMELINE RECORD CONSTRUCTOR HELPER (3D SERIALIZATION) ───
def create_timeline_record(
    sample_id: str,
    cutoff_date: pd.Timestamp,
    active_history_raw: List[Tuple[pd.Timestamp, int, float, List[int]]],
    max_seq_len: int,
    normalized_age: float,
    gender_cat_id: int,
    icd_ids: List[int]
) -> Dict[str, Any]:
    """
    Slices active history to max_seq_len and formats subword arrays as 
    comma-separated blocks inside space-separated timeline sequences.
    """
    if len(active_history_raw) > max_seq_len:
        active_history_raw = active_history_raw[-max_seq_len:]
        
    final_timeline = []
    for evt_date, f_id, v_num, c_subwords_block in active_history_raw:
        lookback_hours = float((cutoff_date - evt_date).total_seconds() / 3600.0)
        subwords_str = ",".join(str(s) for s in c_subwords_block)
        final_timeline.append((lookback_hours, f_id, v_num, subwords_str))
    
    return {
        'mabn': sample_id,
        'cutoff_idx': len(final_timeline) - 1,
        'age': float(normalized_age),
        'gender': int(gender_cat_id),
        'timestamps': " ".join([f"{e[0]:.4f}" for e in final_timeline]),
        'feature_ids': " ".join([str(e[1]) for e in final_timeline]),
        'numeric_values': " ".join([f"{e[2]:.4f}" for e in final_timeline]),
        'cat_result_ids': " ".join([e[3] for e in final_timeline]),  # Space-separated subword blocks
        'icd_targets': " ".join([str(i) for i in icd_ids])
    }


# ─── 5. MAIN UNROLLING PIPELINE ───
if __name__ == "__main__":
    cfg = CardioConfig()
    print("=== Launching Offline Dataset Trajectory Unrolling Pipeline (3D BPE Embedder Edition) ===")
    
    cdha_df = pd.read_csv(cfg.master_cdha_csv, dtype=str)
    xn_df = pd.read_csv(cfg.master_xn_csv, dtype=str)
    
    cdha_df['parsed_date'] = pd.to_datetime(cdha_df['mmyy'].astype(str).str.zfill(4), format='%m%y', errors='coerce')
    xn_df['parsed_date'] = pd.to_datetime(xn_df['ddmmyyyy'], errors='coerce', format='mixed')
    
    cdha_df = cdha_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    xn_df = xn_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    
    # 🚀 Train or load medical BPE Tokenizer using config attributes
    bpe_corpus = xn_df['ketqua'].dropna().tolist() + cdha_df['ketluan'].dropna().tolist()
    bpe_json_path = os.path.join(cfg.checkpoint_dir, cfg.bpe_tokenizer_filename)
    bpe_tokenizer = train_or_load_bpe_tokenizer(
        corpus_texts=bpe_corpus, 
        json_path=bpe_json_path, 
        vocab_size=cfg.bpe_vocab_size
    )
    
    feature_codebook = build_feature_codebook(xn_df, cdha_df)
    
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
                
    np.random.seed(cfg.random_seed)
    train_mabns = greedy_multilabel_stratification(all_patients, patient_labels, train_ratio=cfg.train_split_ratio)

    MAX_SEQ_LEN = cfg.max_sequence_len  
    train_flattened_rows, val_flattened_rows = [], []
    
    train_patients_scanned, val_patients_scanned = 0, 0
    xn_grouped = xn_df.groupby('mabn', sort=False)
    
    # 🚀 Zero-subword array for pure numeric events using cfg.max_subwords
    NUMERIC_SUBWORDS_PAD = [0] * cfg.max_subwords

    print("⏳ Unrolling patient trajectories into pre-computed 3D step slices...")
    for mabn, p_cdha in cdha_df.groupby('mabn', sort=False):
        if mabn not in xn_grouped.groups:
            continue
        p_xn = xn_grouped.get_group(mabn)
        is_train = mabn in train_mabns
        censored_mabn_id = mabn_anonymizer[mabn]
        
        if is_train:
            train_patients_scanned += 1
        else:
            val_patients_scanned += 1
        
        raw_interleaved_events = []
        
        first_cdha_row = p_cdha.iloc[0]
        raw_age = clean_and_parse_numeric(first_cdha_row.get('tuoi', 0)) or 0.0
        normalized_age = max(min(raw_age / cfg.age_normalization_factor, 1.0), 0.0)
        
        raw_gender = str(first_cdha_row.get('phai', '')).strip().lower()
        gender_cat_id = 1 if raw_gender in ['nam', 'm', 'male', '1'] else 2
        
        # 🚀 Ingest Laboratory Tracks
        for _, x_row in p_xn.iterrows():
            evt_date = x_row['parsed_date']
            hp = str(x_row.get('huyetap', ''))
            if '/' in hp:
                try:
                    s_str, d_str = hp.split('/')
                    s_num, d_num = clean_and_parse_numeric(s_str), clean_and_parse_numeric(d_str)
                    if s_num:
                        clipped_s = float(max(min(s_num, cfg.clinical_bounds['sbp'][1]), cfg.clinical_bounds['sbp'][0]))
                        raw_interleaved_events.append((evt_date, feature_codebook['sbp'], clipped_s, NUMERIC_SUBWORDS_PAD))
                    if d_num:
                        clipped_d = float(max(min(d_num, cfg.clinical_bounds['dbp'][1]), cfg.clinical_bounds['dbp'][0]))
                        raw_interleaved_events.append((evt_date, feature_codebook['dbp'], clipped_d, NUMERIC_SUBWORDS_PAD))
                except ValueError:
                    pass

            for field in ['mach', 'nhietdo', 'cannang', 'chieucao']:
                v_num = clean_and_parse_numeric(x_row.get(field))
                if v_num:
                    clipped_v = float(max(min(v_num, cfg.clinical_bounds[field][1]), cfg.clinical_bounds[field][0]))
                    raw_interleaved_events.append((evt_date, feature_codebook[field], clipped_v, NUMERIC_SUBWORDS_PAD))
                    
            lab_name = str(x_row.get('tenxn', '')).strip().lower()
            if lab_name in feature_codebook:
                f_id = feature_codebook[lab_name]
                res_str = str(x_row.get('ketqua', '')).strip().lower()
                num_parsed = clean_and_parse_numeric(res_str)
                if num_parsed is not None:
                    raw_interleaved_events.append((evt_date, f_id, float(num_parsed), NUMERIC_SUBWORDS_PAD))
                elif res_str:
                    subwords_block = encode_text_to_subwords(bpe_tokenizer, res_str, max_subwords=cfg.max_subwords)
                    raw_interleaved_events.append((evt_date, f_id, 0.0, subwords_block))

        # 🚀 Ingest Diagnostic Report Tracks (1 Report = Exactly 1 Event Tuple!)
        for _, c_row in p_cdha.iterrows():
            evt_date = c_row['parsed_date']
            technique = str(c_row.get('kythuatcdha', '')).strip().lower()
            text_summary = str(c_row.get('ketluan', ''))
            
            if technique in feature_codebook:
                f_id = feature_codebook[technique]
                ef_match = re.search(cfg.ef_regex_pattern, text_summary.lower())
                extracted_numeric = float(ef_match.group(1)) if ef_match else 0.0
                
                subwords_block = encode_text_to_subwords(bpe_tokenizer, text_summary, max_subwords=cfg.max_subwords)
                # Appends 1 single positionally-locked event tuple for the entire report:
                raw_interleaved_events.append((evt_date, f_id, extracted_numeric, subwords_block))

        if not raw_interleaved_events:
            continue
        raw_interleaved_events.sort(key=lambda x: x[0])
        
        encounter_codes = p_cdha['maicd'].unique()
        icd_ids = [icd_codebook[code] for code in encounter_codes if code in icd_codebook]
        if not icd_ids:
            continue

        # 🚀 1-Step Shift Trajectory Slicing Loop Preserved
        if is_train:
            for step_idx in range(1, len(raw_interleaved_events)):
                cutoff_date = raw_interleaved_events[step_idx][0]
                active_history_raw = raw_interleaved_events[:step_idx + 1]
                
                record = create_timeline_record(
                    sample_id=f"patient_{censored_mabn_id}_step_{step_idx}",
                    cutoff_date=cutoff_date,
                    active_history_raw=active_history_raw,
                    max_seq_len=MAX_SEQ_LEN,
                    normalized_age=normalized_age,
                    gender_cat_id=gender_cat_id,
                    icd_ids=icd_ids
                )
                train_flattened_rows.append(record)
        else:
            cutoff_date = raw_interleaved_events[-1][0]
            active_history_raw = raw_interleaved_events

            record = create_timeline_record(
                sample_id=f"patient_{censored_mabn_id}",
                cutoff_date=cutoff_date,
                active_history_raw=active_history_raw,
                max_seq_len=MAX_SEQ_LEN,
                normalized_age=normalized_age,
                gender_cat_id=gender_cat_id,
                icd_ids=icd_ids
            )
            val_flattened_rows.append(record)

    pd.DataFrame(train_flattened_rows).to_csv(cfg.train_csv_path, index=False)
    pd.DataFrame(val_flattened_rows).to_csv(cfg.val_csv_path, index=False)
    
    master_codebooks = {
        "metadata": {
            "num_total_features": len(feature_codebook), 
            "num_cat_results": bpe_tokenizer.get_vocab_size(), 
            "num_icd_classes": len(icd_codebook)
        },
        "forward_maps": {
            "features": feature_codebook, 
            "icd_codes": icd_codebook
        },
        "inverse_maps": {str(v): k for k, v in feature_codebook.items()},
        "inverse_icd_codes": {str(v): k for k, v in icd_codebook.items()}
    }
    
    with open(cfg.codebook_json_path, "w", encoding="utf-8") as f:
        json.dump(master_codebooks, f, indent=4, ensure_ascii=False)
        
    print("\n" + "═"*80)
    print(" 📊 OFFLINE STRATIFIED TRAJECTORY UNROLLING COMPILATION REPORT")
    print("═"*80)
    print(f" 📑 TRAINING COHORT CONFIGURATION:")
    print(f"   • Raw Patient Timelines Scanned:       {train_patients_scanned:,} cases")
    print(f"   • Active Pre-Flattened Slices Written:  {len(train_flattened_rows):,} samples")
    print("-" * 80)
    print(f" 📑 VALIDATION COHORT CONFIGURATION:")
    print(f"   • Raw Patient Timelines Scanned:       {val_patients_scanned:,} cases")
    print(f"   • Active Pre-Flattened Slices Written:  {len(val_flattened_rows):,} samples")
    print("-" * 80)
    print(f" 📑 BPE SUBWORD EMBEDDER CONFIGURATION:")
    print(f"   • Medical BPE Vocabulary Size:        {bpe_tokenizer.get_vocab_size():,} subwords")
    print(f"   • Max Subwords Per Event Step:         {cfg.max_subwords} subwords")
    print("═"*80 + "\n")