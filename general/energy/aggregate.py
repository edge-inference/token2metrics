"""
Energy aggregation and analysis functions.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from energy.energy import EnergyProcessor
from energy.utils import PathManager, save_dataframe, collect_energy_files


def aggregate_energy_metrics(base_dir: str) -> pd.DataFrame:
    """
    Aggregate energy metrics from all energy CSV files in base_dir.

    Args:
        base_dir: Base directory containing energy CSV files

    Returns:
        DataFrame with aggregated energy metrics
    """
    print("Collecting energy files...")
    energy_files = collect_energy_files(base_dir)

    if not energy_files:
        print("No energy files found in the specified directory")
        return pd.DataFrame()

    total_files = sum(len(subjects) for subjects in energy_files.values())
    print(f"Found {total_files} energy files across {len(energy_files)} models")

    all_results = []
    processed_count = 0

    processor = EnergyProcessor()

    for model_name in sorted(energy_files.keys()):
        subjects = energy_files[model_name]
        print(f"\nProcessing model: {model_name}")
        print(f"  Found {len(subjects)} subjects")

        for subject_name in sorted(subjects.keys()):
            csv_path = subjects[subject_name]
            print(f"  Processing subject: {subject_name}")

            # Process the energy CSV file
            metrics = processor.process_energy_csv(csv_path, model_name)

            if metrics:
                result_row = {
                    'Model': model_name,
                    'Subject': subject_name,
                    'CSV_Path': csv_path,
                    **metrics
                }
                all_results.append(result_row)
                processed_count += 1
            else:
                print(f"    No metrics extracted from {csv_path}")

    if not all_results:
        print("No valid energy data found")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_results)
    print(f"\nSuccessfully processed {processed_count}/{total_files} files")
    print(f"Generated {len(df)} energy metric records")

    return df


def generate_model_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate model-level summary from detailed results.

    Args:
        detailed_df: DataFrame with detailed energy metrics

    Returns:
        DataFrame with model-level summary
    """
    if detailed_df.empty:
        return pd.DataFrame()

    summary_stats = detailed_df.groupby('Model').agg({
        'avg_power_w': 'mean',
        'peak_power_w': 'max',
        'min_power_w': 'min',
        'total_energy_kj': 'sum',
        'duration_s': 'sum',
        'Subject': 'count'
    }).round(3)

    summary_stats = summary_stats.rename(columns={'Subject': 'num_subjects'})
    summary_stats = summary_stats.reset_index()

    return summary_stats


def load_raw_energy_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load raw energy data for all models to include in detailed summary.

    Args:
        base_dir: Base directory containing energy CSV files

    Returns:
        Dictionary mapping model names to combined raw energy DataFrames
    """
    print("Collecting raw energy data for detailed model sheets...")
    energy_files = collect_energy_files(base_dir)

    model_raw_data = {}

    for model_name in sorted(energy_files.keys()):
        subjects = energy_files[model_name]
        print(f"  Collecting raw data for model: {model_name}")

        model_dfs = []

        for subject_name in sorted(subjects.keys()):
            csv_path = subjects[subject_name]
            
            try:
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    continue
                    
                # Add metadata columns
                df['Model'] = model_name
                df['Subject'] = subject_name
                df['CSV_Path'] = csv_path
                
                # Calculate total power if not already present
                power_columns = ['vdd_cpu_cv_current_mw', 'vdd_gpu_soc_current_mw', 'vin_sys_5v0_current_mw']
                if all(col in df.columns for col in power_columns):
                    df['Total_Power_mW'] = (df['vdd_cpu_cv_current_mw'] + 
                                           df['vdd_gpu_soc_current_mw'] + 
                                           df['vin_sys_5v0_current_mw'])
                    df['Power_W'] = df['Total_Power_mW'] / 1000.0
                
                # Calculate memory utilization percentage if available
                if 'ram_total_mb' in df.columns and 'ram_used_mb' in df.columns:
                    df['RAM_Utilization_Pct'] = (df['ram_used_mb'] / df['ram_total_mb']) * 100
                
                model_dfs.append(df)
                
            except Exception as e:
                print(f"    Error loading {csv_path}: {e}")
                continue

        if model_dfs:
            # Combine all subjects for this model
            combined_df = pd.concat(model_dfs, ignore_index=True)
            model_raw_data[model_name] = combined_df
            print(f"    Loaded {len(combined_df)} records for {model_name}")

    return model_raw_data


def generate_detailed_model_summary(base_dir: str) -> pd.DataFrame:
    """
    Generate detailed model summary with multi-sheet Excel containing raw data.

    Args:
        base_dir: Base directory containing energy CSV files

    Returns:
        DataFrame with model summary
    """
    detailed_df = aggregate_energy_metrics(base_dir)

    if detailed_df.empty:
        return pd.DataFrame()

    model_summary = generate_model_summary(detailed_df)
    raw_data = load_raw_energy_data(base_dir)
    output_path = PathManager.get_output_path('energy_model_summary.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        model_summary.to_excel(writer, sheet_name='Summary', index=False)

        for model_name, raw_df in raw_data.items():
            sheet_name = model_name[:31] if len(model_name) > 31 else model_name
            sheet_name = sheet_name.replace('/', '_').replace('\\', '_')

            if not raw_df.empty:
                cols_to_include = ['Model', 'Subject', 'timestamp', 'Power_W', 'Elapsed_Seconds',
                                   'Energy_Increment', 'Cumulative_Energy', 'gpu_usage_pct', 
                                   'ram_total_mb', 'ram_used_mb', 'RAM_Utilization_Pct', 'temp_gpu_c']
                available_cols = [col for col in cols_to_include if col in raw_df.columns]

                if available_cols:
                    sheet_df = raw_df[available_cols].copy()
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Generated detailed model summary with {len(raw_data)} model sheets containing raw energy data")

    return model_summary
