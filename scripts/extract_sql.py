import os
import io
import re
import pandas as pd

# CONFIGURATION: Target bare-metal database source file path
SQL_FILE_PATH = r"./bvtd_db_20251208.sql" 

def extract_table_from_sql(sql_file, table_keyword):
    """
    Scans a PostgreSQL dump file, automatically extracts column names from 
    the COPY statement, and loads the data rows into a Pandas DataFrame.
    """
    print(f"Searching for table matching '{table_keyword}'...")
    data_lines = []
    column_names = None
    inside_copy_block = False

    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"Could not find your SQL dump file at: {sql_file}")

    # Regex to grab column names inside the parentheses of the COPY statement
    copy_re = re.compile(r"COPY\s+\S+\s*\((.*?)\)\s+FROM", re.IGNORECASE)

    with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1. Parse columns and detect the start of the data block
            if not inside_copy_block and "COPY " in line and table_keyword in line:
                match = copy_re.search(line)
                if match:
                    column_names = [col.strip() for col in match.group(1).split(",")]
                inside_copy_block = True
                print(f"-> Found data block for {table_keyword}. Extracting rows...")
                continue
            
            # 2. Detect the end of the data block
            if inside_copy_block and line.strip() == r"\.":
                inside_copy_block = False
                break
            
            # 3. Collect data lines
            if inside_copy_block:
                data_lines.append(line)

    if not data_lines:
        print(f"⚠️ Warning: No data rows found for table '{table_keyword}'.")
        return pd.DataFrame()

    raw_text_data = "".join(data_lines)
    df = pd.read_csv(
        io.StringIO(raw_text_data), 
        sep="\t", 
        header=None, 
        names=column_names, 
        dtype=str
    )
    return df

def clean_extracted_dataframe(df, identity_column=None):
    """Cleans text fields, removes duplicates, and satisfies modern Pandas requirements."""
    if df.empty:
        return df
        
    target_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in target_cols:
        df[col] = df[col].str.strip()
        
    # Turn empty strings into real Python None values
    df = df.replace(r'^\s*$', None, regex=True)
    
    # Remove duplicate records if an identity primary key is provided
    if identity_column and identity_column in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[identity_column], keep='first')
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"🛡️ CDHA Integrity Check: Dropped {dropped} duplicate rows based on {identity_column}.")
            
    return df

if __name__ == "__main__":
    print("=== Starting Phase 1: Database Extraction Pipeline ===")
    
    # 1. Extract and Clean CDHA (Imaging Reports)
    raw_cdha = extract_table_from_sql(SQL_FILE_PATH, "medical_records_cdha")
    raw_cdha = raw_cdha.rename(columns=str.lower)
    
    # Locked to 'idphieu' for high-level encounter slip tracking
    cdha_df = clean_extracted_dataframe(raw_cdha, identity_column="idphieu")
    
    if not cdha_df.empty:
        cdha_df.to_csv("master_cdha_cleaned.csv", index=False)
        print(f"✅ CDHA Extraction Complete! Saved {len(cdha_df)} records.\n")

    # 2. Extract and Clean XN (Vitals / Labs)
    raw_xn = extract_table_from_sql(SQL_FILE_PATH, "medical_records_xn")
    raw_xn = raw_xn.rename(columns=str.lower)
    
    # Bypass single-column grouping here to protect structural panel contents
    xn_df = clean_extracted_dataframe(raw_xn, identity_column=None)
    
    if not xn_df.empty:
        initial_count = len(xn_df)
        
        # Locked to composite key ['idxetnghiem', 'tenxn'] to preserve sibling panel data
        if 'idxetnghiem' in xn_df.columns and 'tenxn' in xn_df.columns:
            xn_df = xn_df.drop_duplicates(subset=['idxetnghiem', 'tenxn'], keep='first')
            print("🛡️ Panel Integrity Active: Preserving unique test metrics per panel via ['idxetnghiem', 'tenxn'].")
        else:
            fallback_cols = [c for c in ['mabn', 'tenxn', 'ketqua', 'ddmmyyyy'] if c in xn_df.columns]
            xn_df = xn_df.drop_duplicates(subset=fallback_cols, keep='first')
            print(f"⚠️ Column Warning: Keys not matched. Executing backup signature fallback: {fallback_cols}")
            
        dropped = initial_count - len(xn_df)
        if dropped > 0:
            print(f"🛡️ Requisition Check: Filtered out {dropped} duplicate panel rows.")
            
        xn_df.to_csv("master_xn_cleaned.csv", index=False)
        print(f"✅ XN Extraction Complete! Saved {len(xn_df)} high-fidelity lab panel rows.\n")

    # 3. Master Tracking Telemetry Overlap
    if not cdha_df.empty and not xn_df.empty:
        common_patients = set(cdha_df['mabn']).intersection(set(xn_df['mabn']))
        print(f"🎉 Success! Extracted {len(common_patients)} overlapping patient charts across the database.")