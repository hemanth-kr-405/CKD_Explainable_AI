"""
Comprehensive exploratory analysis of the raw UCI Chronic Kidney Disease dataset.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "chronic_kidney_disease.csv")

def analyze_raw_dataset(filepath: str = RAW_DATA_PATH):
    print("=" * 80)
    print("CHRONIC KIDNEY DISEASE DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"\n1. Basic Dimensions:")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    print(f"\n2. Columns & Raw Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:10s}: {str(dtype):10s} (Unique values count: {df[col].nunique(dropna=False)})")
        
    print(f"\n3. Duplicate Rows Count: {df.duplicated().sum()}")
    
    print(f"\n4. Target Column Analysis ('class'):")
    print(df["class"].value_counts(dropna=False))
    
    print(f"\n5. Missing Values Summary (NaNs):")
    missing_counts = df.isna().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"   - {col:10s}: {count:3d} missing ({pct:5.2f}%)")
            
    print(f"\n6. Unique Values for Categorical / Object Columns (Checking for Whitespace / Inconsistencies):")
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        unique_vals = df[col].unique().tolist()
        print(f"   - {col:10s}: {unique_vals}")

if __name__ == "__main__":
    analyze_raw_dataset()
