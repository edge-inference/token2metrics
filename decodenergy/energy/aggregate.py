"""
Energy aggregation and analysis functions.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .energy import EnergyProcessor
from .utils import PathManager, save_dataframe, collect_energy_files, extract_token_length


def aggregate_energy_metrics(base_dir: str, prefer_gpu: bool = True) -> pd.DataFrame:
    """
    Aggregate energy metrics from all CSV files in base_dir.

    Args:
        base_dir: Base directory containing energy CSV files
        prefer_gpu: Prioritize GPU power metrics over CPU (default: False for decode)

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

    for model_name in sorted(energy_files.keys()):
        subjects = energy_files[model_name]
        print(f"Processing model: {model_name}")

        for subject_name in sorted(subjects.keys()):
            csv_path = subjects[subject_name]
            print(f"  Processing subject: {subject_name}")

            processor = EnergyProcessor({f"{model_name}_{subject_name}": csv_path})
            metrics = processor.process_energy_csv(csv_path, f"{model_name}_{subject_name}", prefer_gpu=prefer_gpu)

            if metrics:
                metrics['Model'] = model_name
                metrics['length'] = extract_token_length(subject_name)
                all_results.append(metrics)
                processed_count += 1

    print(f"\nProcessed {processed_count} energy files successfully")

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['Model', 'length'] + [col for col in df.columns if col not in ['Model', 'length']]
        df = df[cols]
        print(f"Created summary with {len(df)} rows and {len(df.columns)} columns")
        
        # Save single sheet version
        output_path = save_dataframe(df, 'energy_detailed_results.xlsx')
        print(f"Saved aggregated results to: {output_path}")
        
        # Save multi-sheet version for fitting analysis
        fitting_output_path = PathManager.get_output_path('energy_detailed_results_by_model.xlsx')
        with pd.ExcelWriter(fitting_output_path, engine='openpyxl') as writer:
            # Create sheets for each model
            for model_name in df['Model'].unique():
                model_df = df[df['Model'] == model_name].copy()
                # Use safe sheet name (Excel limit is 31 chars)
                sheet_name = model_name.replace('-', '_').replace('.', '_')[:31]
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  Created sheet '{sheet_name}' with {len(model_df)} configurations")
        
        print(f"Saved model-specific sheets to: {fitting_output_path}")
        return df

    return pd.DataFrame()


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

    # Determine which columns are present for aggregation
    aggregation_dict = {
        'avg_power_w': 'mean',
        'peak_power_w': 'max',
        'min_power_w': 'min',
        'total_energy_j': 'sum',
        'duration_s': 'sum',
        'avg_energy_per_second': 'mean',
        'length': 'count'
    }

    # Add SYS5V rail aggregations if they exist in the data
    if {'avg_power_w_sys5v', 'peak_power_w_sys5v', 'min_power_w_sys5v', 'total_energy_j_sys5v', 'avg_energy_per_second_sys5v'}.issubset(detailed_df.columns):
        aggregation_dict.update({
            'avg_power_w_sys5v': 'mean',
            'peak_power_w_sys5v': 'max',
            'min_power_w_sys5v': 'min',
            'total_energy_j_sys5v': 'sum',
            'avg_energy_per_second_sys5v': 'mean'
        })

    summary_stats = detailed_df.groupby('Model').agg(aggregation_dict).round(3)

    summary_stats = summary_stats.rename(columns={'length': 'num_configurations'})
    summary_stats = summary_stats.reset_index()

    return summary_stats


def load_raw_energy_data(base_dir: str, prefer_gpu: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load raw energy data for all models to include in detailed summary.

    Args:
        base_dir: Base directory containing energy CSV files
        prefer_gpu: Prioritize GPU power metrics over CPU (default: False for decode)

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
                df = pd.read_csv(csv_path, comment='/')
                df['Model'] = model_name
                df['length'] = extract_token_length(subject_name)

                # Power column selection with CPU/GPU preference support
                power_columns = [
                    ('vdd_gpu_soc_current_mw', 1000) if prefer_gpu else ('vdd_cpu_cv_current_mw', 1000),
                    ('vdd_cpu_cv_current_mw', 1000) if prefer_gpu else ('vdd_gpu_soc_current_mw', 1000),
                    ('power_w', 1),
                    ('vin_sys_5v0_current_mw', 1000)
                ]
                
                for col_name, divisor in power_columns:
                    if col_name in df.columns:
                        df['Power_W'] = df[col_name] / divisor
                        break

                if 'timestamp' in df.columns:
                    times = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S', errors='coerce')
                    if times.isna().all():
                        times = pd.to_datetime(df['timestamp'], errors='coerce')
                    sec = (times - times.iloc[0]).dt.total_seconds().fillna(0)
                    df['Elapsed_Seconds'] = sec
                else:
                    df['Elapsed_Seconds'] = pd.Series(range(len(df)), dtype=float)

                if 'Power_W' in df.columns:
                    td = df['Elapsed_Seconds'].diff().fillna(1.0)
                    df['Energy_Increment'] = df['Power_W'] * td
                    df['Cumulative_Energy'] = df['Energy_Increment'].cumsum()

                model_dfs.append(df)

            except Exception as e:
                print(f"    Warning: Could not load raw data for {subject_name}: {e}")
                continue

        if model_dfs:
            combined_df = pd.concat(model_dfs, ignore_index=True)
            model_raw_data[model_name] = combined_df
            print(f"    Added {len(combined_df)} raw energy measurements for {model_name}")

    return model_raw_data


def generate_detailed_model_summary(base_dir: str, prefer_gpu: bool = True) -> pd.DataFrame:
    """
    Generate detailed model summary with multi-sheet Excel containing raw data.

    Args:
        base_dir: Base directory containing energy CSV files
        prefer_gpu: Prioritize GPU power metrics over CPU (default: False for decode)

    Returns:
        DataFrame with model summary
    """
    detailed_df = aggregate_energy_metrics(base_dir, prefer_gpu=prefer_gpu)

    if detailed_df.empty:
        return pd.DataFrame()

    model_summary = generate_model_summary(detailed_df)
    raw_data = load_raw_energy_data(base_dir, prefer_gpu=prefer_gpu)
    output_path = PathManager.get_output_path('energy_model_summary.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        model_summary.to_excel(writer, sheet_name='Summary', index=False)

        for model_name, raw_df in raw_data.items():
            sheet_name = model_name[:31] if len(model_name) > 31 else model_name
            sheet_name = sheet_name.replace('/', '_').replace('\\', '_')

            if not raw_df.empty:
                cols_to_include = ['Model', 'length', 'timestamp', 'Power_W', 'Elapsed_Seconds',
                                   'Energy_Increment', 'Cumulative_Energy']
                available_cols = [col for col in cols_to_include if col in raw_df.columns]

                if available_cols:
                    sheet_df = raw_df[available_cols].copy()
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Generated detailed model summary with {len(raw_data)} model sheets containing raw energy data")

    return model_summary
