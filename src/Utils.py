# src/Utils.py
r"""
====================================================================================================
CHRONOS-JEPA SHARED CLINICAL PROCESSING & PARSING UTILITIES
====================================================================================================
"""

import re
import hashlib
import pandas as pd
from typing import List, Optional, Any, Set


def clean_and_parse_numeric(val_str: Any) -> Optional[float]:
    """Parses numeric floating-point values from dirty clinical string records."""
    if pd.isna(val_str): 
        return None
    cleaned = str(val_str).strip().replace(',', '.')
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    return float(match.group()) if match else None


def clean_and_tokenize_text(text_str: Any, stop_words: Set[str]) -> List[str]:
    """Cleans clinical text strings, removes punctuation/noise, and filters stop words."""
    if pd.isna(text_str): 
        return []
    normalized = re.sub(r'[\\/\\\n\t.,;:()\[\]\-#?+*!]', ' ', str(text_str).strip().lower())
    words = [w.strip() for w in normalized.split() if w.strip()]
    return [w for w in words if w not in stop_words]


def generate_secure_hash(original_value: Any, salt: str = "HCMUS_CARDIO_JEPA_2026") -> str:
    """Generates a deterministic SHA-256 secure token to obfuscate direct PII keys."""
    if pd.isna(original_value) or str(original_value).strip() == "":
        return "ANON_UNKNOWN"
    raw_str = f"{str(original_value).strip()}_{salt}"
    return "ID_" + hashlib.sha256(raw_str.encode('utf-8')).hexdigest()[:8].upper()