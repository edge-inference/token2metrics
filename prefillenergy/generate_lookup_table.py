#!/usr/bin/env python3
"""
Simple script to generate energy lookup tables from actual fitted model data.
Reads from energy_performance_correlation.xlsx and creates clean lookup tables.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

def read_energy_data_from_excel(excel_file: str) -> Dict:
    """
    Read energy data from the Excel file with model sheets.
    
    Args:
        excel_file: Path to the energy_performance_correlation.xlsx file
        
    Returns:
        Dictionary with model data
    """
    excel_path = Path(excel_file)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    
    model_data = {}
    
    skip_sheets = {'Model_Summary', 'Subject_Analysis'}
    
    for sheet_name, df in all_sheets.items():
        if sheet_name in skip_sheets:
            continue
            
        required_columns = ['ttft_ms', 'tokens_per_second', 'input_tokens', 
                          'avg_power_w', 'total_energy_j', 'energy_per_token']
        
        if not all(col in df.columns for col in required_columns):
            continue
        
        df_clean = df[required_columns].dropna()
        
        if len(df_clean) == 0:
            continue
        
        df_clean = df_clean.sort_values('input_tokens')
        
        model_data[sheet_name] = df_clean
        print(f"✓ Loaded {len(df_clean)} data points from sheet '{sheet_name}'")
    
    return model_data

def create_lookup_table_from_data(model_data: Dict) -> Dict:
    """
    Create a complete lookup table from the actual data with all columns.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        
    Returns:
        Complete lookup table with all metrics as lists
    """
    lookup_table = {}
    
    for model_name, df in model_data.items():
        df_sorted = df.sort_values('input_tokens').copy()
        
        model_data_dict = {
            "ttft_ms": df_sorted['ttft_ms'].round(2).tolist(),
            "tokens_per_second": df_sorted['tokens_per_second'].round(2).tolist(),
            "input_tokens": df_sorted['input_tokens'].astype(int).tolist(),
            "avg_power_w": df_sorted['avg_power_w'].round(3).tolist(),
            "total_energy_j": df_sorted['total_energy_j'].round(4).tolist(),
            "energy_per_token": df_sorted['energy_per_token'].round(6).tolist()
        }
        
        lookup_table[model_name] = model_data_dict
    
    return lookup_table

def create_interpolated_lookup_table(model_data: Dict, target_tokens: List[int] = None) -> Dict:
    """
    Create an interpolated lookup table with specific token counts.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        target_tokens: List of token counts to interpolate to
        
    Returns:
        Interpolated lookup table
    """
    if target_tokens is None:
        target_tokens = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    lookup_table = {}
    
    for model_name, df in model_data.items():
        input_tokens = df['input_tokens'].values
        energy_per_token = df['energy_per_token'].values
        
        interpolated_energy = np.interp(target_tokens, input_tokens, energy_per_token)
        
        model_lookup = {}
        for tokens, energy in zip(target_tokens, interpolated_energy):
            model_lookup[str(tokens)] = round(float(energy), 6)
        
        lookup_table[model_name] = model_lookup
    
    return lookup_table

def main():
    """Generate lookup table from actual Excel data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate energy lookup tables')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[3]
    possible_paths = [
        str(repo_root / "outputs/prefill/energy_performance_correlation_prefill.xlsx"),
        str(repo_root / "outputs/prefill/energy_performance_correlation.xlsx"),
        "output/energy_performance_correlation.xlsx",
        "../output/energy_performance_correlation.xlsx", 
        "energy_performance_correlation.xlsx",
    ]
    
    excel_file = None
    for path in possible_paths:
        if Path(path).exists():
            excel_file = path
            break
    
    if excel_file is None:
        print("✗ Could not find energy_performance_correlation.xlsx")
        print("Please provide the correct path to the Excel file")
        return
    
    print(f"Reading data from: {excel_file}")
    
    try:
        model_data = read_energy_data_from_excel(excel_file)
        
        if not model_data:
            print("✗ No valid model data found in Excel file")
            return
        
        print("\nCreating lookup tables...")
        
        table_lookup = create_lookup_table_from_data(model_data)
        
        interpolated_lookup = create_interpolated_lookup_table(model_data)
        
        out_dir = repo_root / "outputs/prefill"
        out_dir.mkdir(parents=True, exist_ok=True)
        complete_file = out_dir / "prefill_lookup.json"
        interpolated_file = out_dir / "prefill_interpolated_lookup.json"
        
        with open(complete_file, 'w') as f:
            json.dump(table_lookup, f, indent=2)
        
        # with open(interpolated_file, 'w') as f:
        #     json.dump(interpolated_lookup, f, indent=2)
        
        print(f"Saved complete data lookup to: {complete_file}")
        print(f"Saved interpolated lookup to: {interpolated_file}")
        
        if args.verbose:
            # Show preview of complete table structure
            print("\nComplete lookup table structure:")
            for model_name, data in table_lookup.items():
                print(f"{model_name}: {len(data['input_tokens'])} data points")
                print(f"  Columns: {list(data.keys())}")
                print(f"  Input token range: {min(data['input_tokens'])} - {max(data['input_tokens'])}")
        
        return table_lookup, interpolated_lookup
        
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        return None

if __name__ == "__main__":
    main()
