# scripts/extract_sql.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import io
import re
import pandas as pd
from config import CardioConfig


def extract_table_from_sql(sql_file: str, table_keyword: str) -> pd.DataFrame:
    print(f"Searching for table matching '{table_keyword}'...")
    data_lines = []
    column_names = None
    inside_copy_block = False

    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"Could not find your SQL dump file at: {sql_file}")

    copy_re = re.compile(r"COPY\s+\S+\s*\((.*?)\)\s+FROM", re.IGNORECASE)

    with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not inside_copy_block and "COPY " in line and table_keyword in line:
                match = copy_re.search(line)
                if match:
                    column_names = [col.strip() for col in match.group(1).split(",")]
                inside_copy_block = True
                print(f"-> Found data block for {table_keyword}. Extracting rows...")
                continue
            
            if inside_copy_block and line.strip() == r"\.":
                inside_copy_block = False
                break
            
            if inside_copy_block:
                data_lines.append(line)

    if not data_lines:
        print(f"⚠️ Warning: No data rows found for table '{table_keyword}'.")
        return pd.DataFrame()

    raw_text_data = "".join(data_lines)
    return pd.read_csv(io.StringIO(raw_text_data), sep="\t", header=None, names=column_names, dtype=str)


def clean_extracted_dataframe(df: pd.DataFrame, identity_column: str = None) -> pd.DataFrame:
    if df.empty: 
        return df
        
    target_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in target_cols:
        df[col] = df[col].str.strip()
        
    df = df.replace(r'^\s*$', None, regex=True)
    
    if identity_column and identity_column in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[identity_column], keep='first')
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"🛡️ CDHA Integrity Check: Dropped {dropped} duplicate rows based on {identity_column}.")
            
    return df


if __name__ == "__main__":
    cfg = CardioConfig()
    print("=== Starting Phase 1: Database Extraction Pipeline ===")
    
    # 1. CDHA (Imaging Reports)
    raw_cdha = extract_table_from_sql(cfg.sql_file_path, "medical_records_cdha")
    raw_cdha = raw_cdha.rename(columns=str.lower)
    cdha_df = clean_extracted_dataframe(raw_cdha, identity_column="idphieu")
    
    if not cdha_df.empty:
        cdha_df.to_csv(cfg.master_cdha_csv, index=False)
        print(f"✅ CDHA Extraction Complete! Saved {len(cdha_df)} records to {cfg.master_cdha_csv}.\n")

    # 2. XN (Vitals / Labs)
    raw_xn = extract_table_from_sql(cfg.sql_file_path, "medical_records_xn")
    raw_xn = raw_xn.rename(columns=str.lower)
    xn_df = clean_extracted_dataframe(raw_xn, identity_column=None)
    
    if not xn_df.empty:
        initial_count = len(xn_df)
        if 'idxetnghiem' in xn_df.columns and 'tenxn' in xn_df.columns:
            xn_df = xn_df.drop_duplicates(subset=['idxetnghiem', 'tenxn'], keep='first')
        else:
            fallback_cols = [c for c in ['mabn', 'tenxn', 'ketqua', 'ddmmyyyy'] if c in xn_df.columns]
            xn_df = xn_df.drop_duplicates(subset=fallback_cols, keep='first')
            
        dropped = initial_count - len(xn_df)
        if dropped > 0:
            print(f"🛡️ Requisition Check: Filtered out {dropped} duplicate panel rows.")
            
        xn_df.to_csv(cfg.master_xn_csv, index=False)
        print(f"✅ XN Extraction Complete! Saved {len(xn_df)} rows to {cfg.master_xn_csv}.\n")

    if not cdha_df.empty and not xn_df.empty:
        common_patients = set(cdha_df['mabn']).intersection(set(xn_df['mabn']))
        print(f"🎉 Success! Extracted {len(common_patients)} overlapping patient charts.")
