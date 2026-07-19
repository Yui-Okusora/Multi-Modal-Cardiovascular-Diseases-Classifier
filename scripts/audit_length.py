# audit_lengths.py
import pandas as pd
import numpy as np
import re

CLINICAL_STOP_WORDS = {
    'triển', 'khai', 'thí', 'điểm', 'không', 'in', 'phim', 'theo', 'đề', 'án', 'byt',
    'và', 'các', 'của', 'tại', 'khoa', 'đề_án', 'thí_điểm'
}

def clean_and_parse_numeric(val_str):
    if pd.isna(val_str): return None
    cleaned = str(val_str).strip().replace(',', '.')
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    return float(match.group()) if match else None

def clean_and_tokenize_text(text_str):
    if pd.isna(text_str): return []
    normalized = re.sub(r'[\\/\\\n\t.,;:()\[\]\-#?+*!]', ' ', str(text_str).strip().lower())
    return [w.strip() for w in normalized.split() if w.strip() and w.strip() not in CLINICAL_STOP_WORDS]

if __name__ == "__main__":
    print("🔍 Loading raw medical databases to trace patient histories...")
    cdha_df = pd.read_csv("master_cdha_cleaned.csv", dtype=str)
    xn_df = pd.read_csv("master_xn_cleaned.csv", dtype=str)
    
    cdha_df['parsed_date'] = pd.to_datetime(cdha_df['mmyy'].astype(str).str.zfill(4), format='%m%y', errors='coerce')
    xn_df['parsed_date'] = pd.to_datetime(xn_df['ddmmyyyy'], errors='coerce', format='mixed')
    
    cdha_df = cdha_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    xn_df = xn_df.dropna(subset=['mabn', 'parsed_date']).reset_index(drop=True)
    
    all_patients = sorted(list(set(cdha_df['mabn']).intersection(set(xn_df['mabn']))))
    print(f"👥 Auditing {len(all_patients):,} overlapping patient cohorts...")
    
    xn_grouped = xn_df.groupby('mabn', sort=False)
    untruncated_sequence_lengths = []

    for mabn, p_cdha in cdha_df.groupby('mabn', sort=False):
        if mabn not in xn_grouped.groups: continue
        p_xn = xn_grouped.get_group(mabn)
        
        # We start at 2 representing the static header tokens: [tuoi, phai]
        raw_events_count = 2 
        
        # 1. Labs and Vitals
        for _, x_row in p_xn.iterrows():
            hp = str(x_row.get('huyetap', ''))
            if '/' in hp:
                try:
                    s_str, d_str = hp.split('/')
                    if clean_and_parse_numeric(s_str) is not None: raw_events_count += 1
                    if clean_and_parse_numeric(d_str) is not None: raw_events_count += 1
                except ValueError: pass

            for f in ['mach', 'nhietdo', 'cannang', 'chieucao']:
                if clean_and_parse_numeric(x_row.get(f)) is not None:
                    raw_events_count += 1
                    
            lab_name = str(x_row.get('tenxn', '')).strip().lower()
            if lab_name:
                res_str = str(x_row.get('ketqua', '')).strip().lower()
                num_parsed = clean_and_parse_numeric(res_str)
                if num_parsed is not None or res_str:
                    raw_events_count += 1

        # 2. CDHA Reports
        for _, c_row in p_cdha.iterrows():
            tech = str(c_row.get('kythuatcdha', '')).strip().lower()
            if tech:
                text = str(c_row.get('ketluan', ''))
                words = clean_and_tokenize_text(text)
                if words:
                    raw_events_count += len(words)
                else:
                    raw_events_count += 1

        untruncated_sequence_lengths.append(raw_events_count)

    # Calculate statistics
    lengths = np.array(untruncated_sequence_lengths)
    stats = pd.Series(lengths).describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    
    print("\n" + "═"*60)
    print(" 📈 PATIENT TIMELINE SEQUENCE LENGTH PROFILE")
    print("═"*60)
    print(f"   • Total Active Cohort:      {len(lengths):,} patients")
    print(f"   • Minimum Sequence Length:  {stats['min']:.0f} tokens")
    print(f"   • 25th Percentile (Q1):     {stats['25%']:.0f} tokens")
    print(f"   • Median (50th%):           {stats['50%']:.0f} tokens")
    print(f"   • 75th Percentile (Q3):     {stats['75%']:.0f} tokens")
    print(f"   • 90th Percentile:          {stats['90%']:.0f} tokens")
    print(f"   • 95th Percentile:          {stats['95%']:.0f} tokens")
    print(f"   • 99th Percentile:          {stats['99%']:.0f} tokens")
    print(f"   • Absolute Max Length:      {stats['max']:.0f} tokens")
    print("-" * 60)
    print(f"   • Average Sequence Length:  {stats['mean']:.1f} tokens")
    print("═"*60 + "\n")