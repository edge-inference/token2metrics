#!/usr/bin/env python3
"""
Simple script to generate energy and power lookup tables from decodenergy fitted model data.
Reads from energy_performance_correlation.xlsx and creates clean lookup tables for decode operations.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

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
            print(f"⚠️  Skipping sheet '{sheet_name}' - missing required columns")
            continue
        
        df_clean = df[required_columns].dropna()
        
        if len(df_clean) == 0:
            print(f"⚠️  No valid data in sheet '{sheet_name}'")
            continue
        
        df_clean = df_clean.sort_values(['input_tokens', 'output_tokens'])
        
        model_data[sheet_name] = df_clean
        print(f"✅ Loaded {len(df_clean)} data points from sheet '{sheet_name}'")
    
    return model_data

def create_decode_lookup_table_from_data(model_data: Dict) -> Dict:
    """
    Create a complete lookup table from the actual decode data with all columns.
    
    Args:
        model_data: Dictionary with DataFrames for each model
        
    Returns:
        Complete lookup table with all decode metrics as lists
    """
    lookup_table = {}
    
    for model_name, df in model_data.items():
        df_sorted = df.sort_values(['input_tokens', 'output_tokens']).copy()
        
        model_data_dict = {
            "input_tokens": df_sorted['input_tokens'].astype(int).tolist(),
            "output_tokens": df_sorted['output_tokens'].astype(int).tolist(),
            "avg_power_w": df_sorted['avg_power_w'].round(3).tolist(),
            "total_energy_j": df_sorted['total_energy_j'].round(4).tolist(),
            "energy_per_token": df_sorted['energy_per_token'].round(6).tolist()
        }
        
        lookup_table[model_name] = model_data_dict
    
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
        target_output_tokens = [1, 16, 32, 64, 96, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    
    lookup_table = {}
    
    for model_name, df in model_data.items():
        # Average across input lengths for each output length
        output_avg = df.groupby('output_tokens').agg({
            'avg_power_w': 'mean',
            'energy_per_token': 'mean'
        }).reset_index()
        
        output_tokens = output_avg['output_tokens'].values
        avg_power = output_avg['avg_power_w'].values
        energy_per_token = output_avg['energy_per_token'].values
        
        # Interpolate to target output tokens
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
        # Find closest input length
        available_inputs = df['input_tokens'].unique()
        closest_input = min(available_inputs, key=lambda x: abs(x - target_input_length))
        
        # Filter to that input length
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
    
    repo_root = Path(__file__).resolve().parents[3]
    possible_paths = [
        str(repo_root / "outputs/decode/energy_performance_correlation_decode.xlsx"),
        str(repo_root / "outputs/decode/energy_performance_correlation.xlsx"),
        "output/energy_performance_correlation.xlsx",
        "../output/energy_performance_correlation.xlsx", 
        "energy_performance_correlation.xlsx",
        "decodenergy/output/energy_performance_correlation.xlsx"
    ]
    
    excel_file = None
    for path in possible_paths:
        if Path(path).exists():
            excel_file = path
            break
    
    if excel_file is None:
        print("❌ Could not find energy_performance_correlation.xlsx")
        print("📍 Please provide the correct path to the Excel file")
        return
    
    print(f"Reading decode energy data from: {excel_file}")
    
    try:
        model_data = read_decode_energy_data_from_excel(excel_file)
        
        if not model_data:
            print("❌ No valid model data found in Excel file")
            return
        
        print("\n🔧 Creating decode lookup tables...")
        
        # Create output directory
        out_dir = repo_root / "outputs/decode/fitting"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Complete data lookup table
        complete_lookup = create_decode_lookup_table_from_data(model_data)
        
        # 2. Power scaling lookup table (averaged across input lengths)
        power_scaling_lookup = create_power_scaling_lookup_table(model_data)
        
        # 3. Input-specific lookup table (for input ~512 tokens)
        input_specific_lookup = create_input_specific_lookup_table(model_data, target_input_length=512)
        
        # Save all lookup tables
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
        for model_name, data in complete_lookup.items():
            print(f"{model_name}: {len(data['output_tokens'])} data points")
            print(f"  Columns: {list(data.keys())}")
            print(f"  Output token range: {min(data['output_tokens'])} - {max(data['output_tokens'])}")
            print(f"  Input token range: {min(data['input_tokens'])} - {max(data['input_tokens'])}")
        
        return complete_lookup, power_scaling_lookup, input_specific_lookup
        
    except Exception as e:
        print(f"❌ Error processing decode data: {e}")
        return None

if __name__ == "__main__":
    main()
