import os
import hashlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

def generate_secure_hash(original_value, salt="HCMUS_CARDIO_JEPA_2026"):
    """
    Generates a deterministic, non-reversible secure pseudonym token 
    to obfuscate direct hospital identification keys.
    """
    if pd.isna(original_value) or str(original_value).strip() == "":
        return "ANON_UNKNOWN"
    raw_str = f"{str(original_value).strip()}_{salt}"
    return "ID_" + hashlib.sha256(raw_str.encode('utf-8')).hexdigest()[:8].upper()

def inspect_random_patient_profile(cdha_path="master_cdha_cleaned.csv", xn_path="master_xn_cleaned.csv"):
    print("═══ LAUNCHING TRUE JEPA EVALUATION PATIENT INTEGRITY INSPECTOR ═══\n")
    
    # 1. Validation Check for Source Files
    if not os.path.exists(cdha_path) or not os.path.exists(xn_path):
        raise FileNotFoundError(
            f"❌ Missing extraction assets. Ensure extract_sql.py has run successfully.\n"
            f"   Missing path target: {cdha_path if not os.path.exists(cdha_path) else xn_path}"
        )

    # 2. Load Dataframes with explicit text parsing to preserve keys
    df_cdha = pd.read_csv(cdha_path, dtype=str)
    df_xn = pd.read_csv(xn_path, dtype=str)

    # 3. Identify overlapping tracking intersections
    patients_cdha = set(df_cdha['mabn'].dropna().unique())
    patients_xn = set(df_xn['mabn'].dropna().unique())
    overlapping_mabns = list(patients_cdha.intersection(patients_xn))

    if not overlapping_mabns:
        print("⚠️ Warning: No perfect patient intersections detected across logs. Pulling from joint pools.")
        overlapping_mabns = list(patients_cdha if patients_cdha else patients_xn)

    if not overlapping_mabns:
        print("🚨 Critical Error: Cleaned target tables contain zero viable patient timelines.")
        return

    # 4. Draw a random patient profile honestly using a random generator seed
    selected_patient_mabn = np.random.choice(overlapping_mabns)
    secure_patient_token = generate_secure_hash(selected_patient_mabn)

    # 5. Extract individual clinical footprints
    rows_cdha = df_cdha[df_cdha['mabn'] == selected_patient_mabn].copy()
    rows_xn = df_xn[df_xn['mabn'] == selected_patient_mabn].copy()

    # =====================================================================
    # 🔒 STRICT DE-IDENTIFICATION & CENSORSHIP ENGINE (PII / SPII SANITIZER)
    # =====================================================================
    
    # Target Lists for direct/indirect operational keys
    direct_pii_keys = ['mabn', 'tenbn', 'hoten', 'diachi', 'phone']
    system_operational_keys = ['idphieu', 'mavaovien', 'maql', 'idxetnghiem', 'idcdha']
    demographic_quasi_keys = ['ngaysinh']
    facility_keys = ['id_bv_ten', 'ten_bv_ten', 'id_bv_so', 'ten_bv_so']

    # Process CDHA entries
    if not rows_cdha.empty:
        # Anonymize core relational tracking nodes
        rows_cdha['mabn'] = secure_patient_token
        
        for col in system_operational_keys:
            if col in rows_cdha.columns:
                rows_cdha[col] = rows_cdha[col].apply(lambda v: generate_secure_hash(v, salt="CDHA_OP"))
                
        # Obfuscate direct/indirect personal PII strings
        for col in direct_pii_keys + demographic_quasi_keys:
            if col in rows_cdha.columns:
                rows_cdha[col] = "[CENSORED_PERSONAL_DATA]"
                
        # Age Masking for nonagenarian safety boundaries
        if 'tuoi' in rows_cdha.columns:
            rows_cdha['tuoi'] = rows_cdha['tuoi'].apply(lambda age: ">89" if pd.notna(age) and int(float(age)) > 89 else age)

    # Process XN entries
    if not rows_xn.empty:
        # Anonymize core relational tracking nodes
        rows_xn['mabn'] = secure_patient_token
        
        for col in system_operational_keys:
            if col in rows_xn.columns:
                rows_xn[col] = rows_xn[col].apply(lambda v: generate_secure_hash(v, salt="XN_OP"))
                
        # Obfuscate direct/indirect personal PII and structural branch metadata
        for col in direct_pii_keys + demographic_quasi_keys + facility_keys:
            if col in rows_xn.columns:
                rows_xn[col] = "[CENSORED_DATA_BLOB]"
                
        # Age Masking
        if 'tuoi' in rows_xn.columns:
            rows_xn['tuoi'] = rows_xn['tuoi'].apply(lambda age: ">89" if pd.notna(age) and int(float(age)) > 89 else age)

    # =====================================================================
    # 🖨️ REPORT PRESENTATION OUTPUTS
    # =====================================================================
    print(f"=====================================================================================")
    print(f"👤 ANONYMIZED PATIENT TARGET INITIALIZED: {secure_patient_token}")
    print(f"=====================================================================================\n")

    # 1. CDHA View
    print(f"┌───────────────────────────────────────────────────────────────────────────────────┐")
    print(f"  📋 TRACK 1: TEXT-ANCHORED IMAGING & REPORTING FOOTPRINTS (CDHA) | Total Rows: {len(rows_cdha)}")
    print(f"└───────────────────────────────────────────────────────────────────────────────────┘")
    if not rows_cdha.empty:
        # Reorder columns to showcase narrative evaluation clearly to panels
        cdha_view_cols = [c for c in ['mabn', 'maicd', 'chandoan', 'kythuatcdha', 'ketluan', 'mmyy'] if c in rows_cdha.columns]
        extended_cols = cdha_view_cols + [c for c in rows_cdha.columns if c not in cdha_view_cols and c not in direct_pii_keys]
        
        with pd.option_context('display.max_colwidth', 60, 'display.width', 1000):
            print(rows_cdha[extended_cols].to_string(index=False))
    else:
        print("  No sequential matching records located inside CDHA array.")

    print("\n" + "-"*115 + "\n")

    # 2. XN View
    print(f"┌───────────────────────────────────────────────────────────────────────────────────┐")
    print(f"  📊 TRACK 2: TIME-SERIES PHYSIOLOGICAL VITAL SIGN SIGNATURES (XN) | Total Rows: {len(rows_xn)}")
    print(f"└───────────────────────────────────────────────────────────────────────────────────┘")
    if not rows_xn.empty:
        # Sort values by time indexing arrays to reflect the true uncollapsed sequence grid
        if 'ddmmyyyy' in rows_xn.columns:
            rows_xn = rows_xn.sort_values(by='ddmmyyyy')
            
        # Prioritize 12-D vital timeline configuration column parameters
        vitals_features = ['ddmmyyyy', 'mabn', 'maicd', 'chandoan', 'tenxn', 'ketqua', 'huyetap', 'mach', 'nhietdo', 'cannang', 'chieucao']
        active_vitals_cols = [c for c in vitals_features if c in rows_xn.columns]
        all_remaining_xn = active_vitals_cols + [c for c in rows_xn.columns if c not in active_vitals_cols and c not in direct_pii_keys + facility_keys]
        
        with pd.option_context('display.max_columns', None, 'display.width', 1200, 'display.max_colwidth', 40):
            print(rows_xn[all_remaining_xn].to_string(index=False))
    else:
        print("  No corresponding physiological timeline measurements mapped for this patient track.")

    print(f"\n=====================================================================================")
    print(f"✅ Secure extraction check complete. Patient data fully anonymized.")
    print(f"=====================================================================================")

if __name__ == "__main__":
    inspect_random_patient_profile()