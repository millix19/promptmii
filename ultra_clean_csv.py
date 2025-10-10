#!/usr/bin/env python3
import pandas as pd
import csv
import re
import os

def ultra_clean_csv(input_path, output_path):
    """Ultra-clean CSV for maximum compatibility"""
    print(f'Loading CSV: {input_path}')
    
    try:
        # Try multiple loading approaches
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(input_path, encoding='latin-1')
            except:
                df = pd.read_csv(input_path, encoding='cp1252')
        
        print(f'Loaded {len(df)} rows')
    except Exception as e:
        print(f'Failed to load: {e}')
        return False
    
    print('Ultra-cleaning data...')
    
    # Ultra-aggressive cleaning
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f'Cleaning column: {col}')
            
            # Convert to string, handle NaN
            df[col] = df[col].fillna('').astype(str)
            
            # Remove problematic characters but preserve newlines and tabs
            df[col] = df[col].str.replace(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', regex=True)
            
            # Clean up multiple spaces but preserve newlines
            df[col] = df[col].str.replace(r'[ \t]+', ' ', regex=True)  # Multiple spaces/tabs to single space
            df[col] = df[col].str.replace(r'\n+', '\n', regex=True)  # Multiple newlines to single newline
            
            # Truncate very long fields
            df[col] = df[col].str[:4000]  # Much shorter limit
            
            # Strip whitespace
            df[col] = df[col].str.strip()
            
            # Remove any remaining problematic strings
            df[col] = df[col].str.replace('"', "'")
            df[col] = df[col].str.replace(',', ';')  # Replace commas with semicolons
    
    print('Saving ultra-clean CSV...')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with newline-preserving compatibility
    df.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig',  # UTF-8 with BOM for Google Sheets
        quoting=csv.QUOTE_ALL,  # Quote all fields to handle newlines properly
        quotechar='"',
        escapechar='\\',
        sep=','
    )
    
    print(f'Saved to: {output_path}')
    print(f'Final shape: {df.shape}')
    
    # Test if it can be opened
    try:
        test_df = pd.read_csv(output_path)
        print('✅ CSV can be read successfully!')
        return True
    except Exception as e:
        print(f'❌ CSV still has issues: {e}')
        return False

def main():
    # Default paths
    input_file = '/home/emilyx/verl/out/baseline_eval_v3_llama_validation/analysis/baseline_eval_results_small.csv'
    output_file = '/home/emilyx/verl/out/baseline_eval_v3_llama_validation/analysis/baseline_eval_results_small_ultra_clean.csv'
    
    # Allow command line arguments
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print('='*60)
    print('ULTRA-CLEAN CSV FOR MAXIMUM COMPATIBILITY')
    print('='*60)
    
    success = ultra_clean_csv(input_file, output_file)
    
    if success:
        print('\n🎉 Ultra-clean CSV created!')
        print(f'File: {output_file}')
        print('\nThis version:')
        print('- Preserves newlines for readability')
        print('- Removes problematic control characters')
        print('- Quotes all fields to handle newlines properly')
        print('- UTF-8-BOM encoding for Google Sheets compatibility')
    else:
        print('\n💥 Failed to create ultra-clean CSV')

if __name__ == "__main__":
    main() 