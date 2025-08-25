#!/usr/bin/env python3
"""
Simple script to generate energy and power lookup tables from decodenergy fitted model data.
Reads from energy_performance_correlation.xlsx and creates clean lookup tables for decode operations.
"""
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List

_config_path = Path(__file__).resolve().parent.parent / "config"
if str(_config_path) not in sys.path:
    sys.path.insert(0, str(_config_path))

try:
    from common import get_decode_paths
    _USE_COMMON_CONFIG = True
except ImportError:
    _USE_COMMON_CONFIG = False


def get_decode_correlation_file() -> Path:
    """Find decode correlation file using centralized config or fallback."""
    if _USE_COMMON_CONFIG:
        paths = get_decode_paths()
        possible_paths = [
            paths['output_dir'] / "energy_performance_correlation.xlsx",
            paths['output_dir'] / "energy_performance_correlation_decode.xlsx"
        ]
    else:
        repo_root = Path(__file__).resolve().parents[3]
        possible_paths = [
            repo_root / "outputs/decode/energy_performance_correlation.xlsx",
            repo_root / "outputs/decode/energy_performance_correlation_decode.xlsx"
        ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None


def get_decode_output_dir() -> Path:
    """Get decode output directory using centralized config or fallback."""
    if _USE_COMMON_CONFIG:
        paths = get_decode_paths()
        return paths['output_dir'] / "fitting"
    else:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "outputs/decode/fitting"

def read_decode_energy_data_from_excel(excel_file: str) -> Dict:
    """
    Read decode energy data from the Excel file with model sheets.
    
    Args:
        excel_file: Path to the energy_performance_correlation.xlsx file
        
    Returns:
        Dictionary with model data
    """
    excel_path = Path(excel_file)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Read all sheets from the Excel file
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    
    model_data = {}
    
    for sheet_name, df in all_sheets.items():
        # Skip summary sheets
        if sheet_name in ['Model_Summary', 'Subject_Analysis']:
            continue
            
        required_columns = ['input_tokens', 'output_tokens', 'avg_power_w', 
                          'total_energy_j', 'energy_per_token']
        
        if not all(col in df.columns for col in required_columns):
            print(f"! Skipping sheet '{sheet_name}' - missing required columns")
            continue
        
        df_clean = df[required_columns].dropna()
        
        if len(df_clean) == 0:
            print(f"! No valid data in sheet '{sheet_name}'")
            continue
        
        df_clean = df_clean.sort_values(['input_tokens', 'output_tokens'])
        
        model_data[sheet_name] = df_clean
        print(f"✓ Loaded {len(df_clean)} data points from sheet '{sheet_name}'")
    
    return model_data

def create_decode_lookup_table_from_data(model_data: Dict) -> Dict:
    """
    Create a complete lookup table from decode data with nested structure.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        
    Returns:
        Complete lookup table with input tokens as keys and output data as values
    """
    lookup_table = {}
    
    for model_name, df in model_data.items():
        df_sorted = df.sort_values(['input_tokens', 'output_tokens']).copy()
        
        model_lookup = {}
        for input_tokens, group in df_sorted.groupby('input_tokens'):
            group_sorted = group.sort_values('output_tokens')
            
            model_lookup[str(int(input_tokens))] = {
                "output_tokens": group_sorted['output_tokens'].astype(int).tolist(),
                "avg_power_w": group_sorted['avg_power_w'].round(3).tolist(),
                "total_energy_j": group_sorted['total_energy_j'].round(4).tolist(),
                "energy_per_token": group_sorted['energy_per_token'].round(6).tolist()
            }
        
        lookup_table[model_name] = model_lookup
    
    return lookup_table

def create_power_scaling_lookup_table(model_data: Dict, target_output_tokens: List[int] = None) -> Dict:
    """
    Create a power scaling lookup table averaging across input lengths.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        target_output_tokens: List of output token counts to interpolate to
        
    Returns:
        Power scaling lookup table
    """
    if target_output_tokens is None:
        target_output_tokens = [1, 16, 32, 64, 96] + list(range(128, 2049, 128))
    
    lookup_table = {}
    
    for model_name, df in model_data.items():
        output_avg = df.groupby('output_tokens').agg({
            'avg_power_w': 'mean',
            'energy_per_token': 'mean'
        }).reset_index()
        
        output_tokens = output_avg['output_tokens'].values
        avg_power = output_avg['avg_power_w'].values
        energy_per_token = output_avg['energy_per_token'].values
        
        interpolated_power = np.interp(target_output_tokens, output_tokens, avg_power)
        interpolated_energy = np.interp(target_output_tokens, output_tokens, energy_per_token)
        
        model_lookup = {}
        for tokens, power, energy in zip(target_output_tokens, interpolated_power, interpolated_energy):
            model_lookup[str(tokens)] = {
                "avg_power_w": round(float(power), 3),
                "energy_per_token": round(float(energy), 6)
            }
        
        lookup_table[model_name] = model_lookup
    
    return lookup_table

def create_input_specific_lookup_table(model_data: Dict, target_input_length: int = 514) -> Dict:
    """
    Create a lookup table for a specific input length.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        target_input_length: Target input length to filter for
        
    Returns:
        Input-specific lookup table
    """
    lookup_table = {}
    
    for model_name, df in model_data.items():
        available_inputs = df['input_tokens'].unique()
        closest_input = min(available_inputs, key=lambda x: abs(x - target_input_length))
        
        df_filtered = df[df['input_tokens'] == closest_input].copy()
        df_sorted = df_filtered.sort_values('output_tokens')
        
        if len(df_sorted) == 0:
            continue
        
        model_data_dict = {
            "input_tokens": int(closest_input),
            "output_tokens": df_sorted['output_tokens'].astype(int).tolist(),
            "avg_power_w": df_sorted['avg_power_w'].round(3).tolist(),
            "energy_per_token": df_sorted['energy_per_token'].round(6).tolist()
        }
        
        lookup_table[model_name] = model_data_dict
    
    return lookup_table

def main():
    """Generate decode energy/power lookup tables from actual Excel data."""
    
    # Find correlation file
    excel_file = get_decode_correlation_file()
    
    if excel_file is None:
        print("✗ Could not find energy_performance_correlation.xlsx")
        print("Please ensure correlation analysis has been run first")
        return
    
    print(f"Reading decode energy data from: {excel_file}")
    
    try:
        model_data = read_decode_energy_data_from_excel(excel_file)
        
        if not model_data:
            print("✗ No valid model data found in Excel file")
            return
        
        print("\nCreating decode lookup tables...")
        
        out_dir = get_decode_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        complete_lookup = create_decode_lookup_table_from_data(model_data)
        
        power_scaling_lookup = create_power_scaling_lookup_table(model_data)
        
        input_specific_lookup = create_input_specific_lookup_table(model_data, target_input_length=512)
        
        complete_file = out_dir / "decode_energy_lookup_complete.json"
        power_scaling_file = out_dir / "decode_power_scaling_lookup.json"
        input_specific_file = out_dir / "decode_input_512_lookup.json"
        
        with open(complete_file, 'w') as f:
            json.dump(complete_lookup, f, indent=2)
        
        with open(power_scaling_file, 'w') as f:
            json.dump(power_scaling_lookup, f, indent=2)
        
        with open(input_specific_file, 'w') as f:
            json.dump(input_specific_lookup, f, indent=2)
        
        print(f"Saved complete decode data lookup to: {complete_file}")
        print(f"Saved power scaling lookup to: {power_scaling_file}")
        print(f"Saved input-specific lookup to: {input_specific_file}")
        
        # Show preview of lookup table structure
        print("\nDecode lookup table structure:")
        for model_name, input_data in complete_lookup.items():
            total_points = sum(len(input_group['output_tokens']) for input_group in input_data.values())
            print(f"{model_name}: {total_points} data points across {len(input_data)} input lengths")
            print(f"  Input token lengths: {sorted([int(k) for k in input_data.keys()])}")
            
            # Show sample from first input length
            first_input = next(iter(input_data.values()))
            print(f"  Sample columns: {list(first_input.keys())}")
            print(f"  Sample output range: {min(first_input['output_tokens'])} - {max(first_input['output_tokens'])}")
        
        return complete_lookup, power_scaling_lookup, input_specific_lookup
        
    except Exception as e:
        print(f"✗ Error processing decode data: {e}")
        return None

if __name__ == "__main__":
    main()
