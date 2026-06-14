import os
import io
import re
import pandas as pd

# CONFIGURATION: Set the path to your single .sql file here
SQL_FILE_PATH = r".\bvtd_db_20251208.sql"  # <-- CHANGE THIS to your actual .sql file name

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
    # e.g., COPY public.table_name (col1, col2) FROM stdin;
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

    if not column_names:
        print(f"⚠️ Warning: Could not automatically parse columns for {table_keyword}. Using default numbering.")

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
        
    # FIX: Explicitly include 'str' alongside 'object' to silence the Pandas4Warning
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
            print(f"🛡️ Integrity Check: Dropped {dropped} duplicate rows based on {identity_column}.")
            
    return df

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Dual Table SQL Extraction Pipeline ---")

    # 1. Extract and Clean CDHA (Imaging Reports)
    raw_cdha = extract_table_from_sql(SQL_FILE_PATH, "medical_records_cdha")
    cdha_df = clean_extracted_dataframe(raw_cdha, identity_column="IDPHIEU")
    
    # Standardize column casing to lowercase for matching with the PyTorch Dataset
    cdha_df = cdha_df.rename(columns=str.lower)
    if not cdha_df.empty:
        cdha_df.to_csv("master_cdha_cleaned.csv", index=False)
        print(f"✅ CDHA Processing Complete! Saved {len(cdha_df)} rows to 'master_cdha_cleaned.csv'.\n")

    # 2. Extract and Clean XN (Vitals / Labs)
    # The script looks for 'medical_records_xn' inside the dump file
    raw_xn = extract_table_from_sql(SQL_FILE_PATH, "medical_records_xn")
    
    # If your XN table uses IDPHIEU as a primary key, change identity_column to "IDPHIEU"
    xn_df = clean_extracted_dataframe(raw_xn, identity_column="idphieu")
    xn_df = xn_df.rename(columns=str.lower)
    
    if not xn_df.empty:
        xn_df.to_csv("master_xn_cleaned.csv", index=False)
        print(f"✅ XN Processing Complete! Saved {len(xn_df)} rows to 'master_xn_cleaned.csv'.\n")

    # 3. Final Multi-Modal Cross-Check Verification
    if not cdha_df.empty and not xn_df.empty:
        common_patients = set(cdha_df['mabn']).intersection(set(xn_df['mabn']))
        print(f"🎉 Success! Found {len(common_patients)} matching patient IDs across both extracted files.")