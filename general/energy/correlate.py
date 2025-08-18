"""
Energy-Performance Correlation Module

Correlates energy consumption data with token-level performance metrics
using the existing energy_model_summary.xlsx file that already has processed energy data.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import re

from energy.utils import PathManager, save_dataframe


class EnergyPerformanceCorrelator:
    """Correlates energy consumption with performance metrics using processed energy data."""
    
    def __init__(self, energy_base_dir: str, performance_file: str):
        """
        Initialize correlator with energy and performance data paths.
        
        Args:
            energy_base_dir: Directory containing energy CSV files (used for file discovery)
            performance_file: Path to performance results Excel file
        """
        self.energy_base_dir = energy_base_dir
        self.performance_file = performance_file
        self.energy_data = {}
        self.performance_data = {}
        
        # Model name mapping from short names to full names
        # Note: Excluding L1-Max as it's identical to 1.5B model
        self.model_name_mapping = {
            '8B': 'DeepSeek-R1-Distill-Llama-8B',
            '1.5B': 'DeepSeek-R1-Distill-Qwen-1.5B', 
            '14B': 'DeepSeek-R1-Distill-Qwen-14B'
            # 'Max': 'L1-Qwen-1.5B-Max'  # Removed - duplicate of 1.5B
        }
        
    def load_performance_data(self) -> Dict[str, pd.DataFrame]:
        """Load performance data from Excel file."""
        print("Loading performance data...")
        
        # Read Excel file to get sheet names
        xl_file = pd.ExcelFile(self.performance_file)
        
        # Skip overview and comparison sheets, load model-specific sheets
        model_sheets = [sheet for sheet in xl_file.sheet_names 
                       if sheet not in ['Overview', 'Subject_Comparison']]
        
        performance_data = {}
        for sheet in model_sheets:
            print(f"  Loading performance data for: {sheet}")
            df = pd.read_excel(self.performance_file, sheet_name=sheet)
            
            # Clean model name (remove underscores that Excel adds)
            model_name = sheet.replace('_', '-').replace('1-5B', '1.5B')
            performance_data[model_name] = df
            
        print(f"Loaded performance data for {len(performance_data)} models")
        return performance_data
    
    def load_processed_energy_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load processed energy data from energy_model_summary.xlsx.
        
        Returns:
            Dictionary mapping full model names to energy DataFrames
        """
        print("Loading processed energy data from energy_model_summary.xlsx...")
        
        energy_summary_path = PathManager.get_output_path('energy_model_summary.xlsx')
        
        if not energy_summary_path.exists():
            print(f"Error: Energy summary file not found: {energy_summary_path}")
            print("Please run basic energy analysis first: python -m energy.cli --base-dir ./tegra/figure2")
            return {}
        
        # Read the Excel file to get available sheets
        xl_file = pd.ExcelFile(energy_summary_path)
        
        # Skip the Summary sheet, load individual model sheets
        model_sheets = [sheet for sheet in xl_file.sheet_names if sheet != 'Summary']
        
        energy_data = {}
        for sheet in model_sheets:
            print(f"  Loading energy data for: {sheet}")
            df = pd.read_excel(energy_summary_path, sheet_name=sheet)
            
            # Normalize variants (remove extra hyphens/underscores spacing)
            def _norm_model(name: str) -> str:
                return name.replace('_', '-').replace('--', '-').replace(' ', '').strip()

            full_model_name = self.model_name_mapping.get(sheet, sheet)
            full_model_name = _norm_model(full_model_name)
            energy_data[full_model_name] = df
            
            print(f"    Loaded {len(df)} energy measurements for {full_model_name}")
        
        print(f"Loaded energy data for {len(energy_data)} models")
        return energy_data
    
    def parse_energy_filename_from_data(self, subject: str, filepath_or_data: str) -> Dict[str, str]:
        """
        Parse energy metadata from subject and any available filename info.
        Since we're using processed data, we'll extract what we can from the subject.
        """
        # For now, we'll use subject directly and try to match by subject name
        # The detailed matching will happen in the correlation step
        return {
            'subject': subject,
            'question_id': None,  # Will be matched later
            'input_tokens': None,  # Will be matched later
            'output_tokens': None  # Will be matched later
        }
    
    def calculate_energy_metrics_from_processed_data(self, energy_rows: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate energy metrics from processed energy data rows.
        
        Args:
            energy_rows: DataFrame rows containing energy measurements for a question
            
        Returns:
            Dictionary with energy metrics
        """
        if len(energy_rows) == 0:
            return self._get_zero_energy_metrics()
        
        try:
            # Check if we have Power_W column (raw energy data)
            if 'Power_W' in energy_rows.columns:
                # Calculate metrics from raw power measurements
                avg_power = energy_rows['Power_W'].mean()
                peak_power = energy_rows['Power_W'].max()
                min_power = energy_rows['Power_W'].min()
                
                # Calculate duration and total energy
                duration_s = len(energy_rows)  # Assume 1 second per measurement
                
                # If we have timestamp data, calculate actual duration
                if 'timestamp' in energy_rows.columns:
                    try:
                        # Convert timestamps to datetime if they're strings
                        timestamps = pd.to_datetime(energy_rows['timestamp'])
                        duration_s = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
                        if duration_s <= 0:
                            duration_s = len(energy_rows)
                    except:
                        duration_s = len(energy_rows)
                
                metrics = {
                    'avg_power_w': round(avg_power, 3),
                    'peak_power_w': round(peak_power, 3),
                    'min_power_w': round(min_power, 3),
                    'measurements_count': len(energy_rows)
                }
                
                # Add GPU utilization metrics if available
                if 'gpu_usage_pct' in energy_rows.columns:
                    metrics['avg_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].mean(), 2)
                    metrics['peak_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].max(), 2)
                    metrics['min_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].min(), 2)
                
                # Add memory usage metrics if available
                if 'ram_total_mb' in energy_rows.columns and 'ram_used_mb' in energy_rows.columns:
                    metrics['avg_ram_used_mb'] = round(energy_rows['ram_used_mb'].mean(), 1)
                    metrics['peak_ram_used_mb'] = round(energy_rows['ram_used_mb'].max(), 1)
                    metrics['avg_ram_total_mb'] = round(energy_rows['ram_total_mb'].mean(), 1)
                    # Calculate memory utilization percentage
                    ram_utilization_pct = (energy_rows['ram_used_mb'] / energy_rows['ram_total_mb']) * 100
                    metrics['avg_ram_utilization_pct'] = round(ram_utilization_pct.mean(), 2)
                    metrics['peak_ram_utilization_pct'] = round(ram_utilization_pct.max(), 2)
                
                # Add GPU temperature metrics if available
                if 'temp_gpu_c' in energy_rows.columns:
                    metrics['avg_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].mean(), 1)
                    metrics['peak_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].max(), 1)
                    metrics['min_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].min(), 1)
                
                return metrics
            
            # Fallback: try to use existing processed columns
            elif 'avg_power_w' in energy_rows.columns:
                # This is already aggregated data
                avg_power = energy_rows['avg_power_w'].iloc[0]
                peak_power = energy_rows.get('peak_power_w', [avg_power]).iloc[0]
                min_power = energy_rows.get('min_power_w', [avg_power]).iloc[0]
                total_energy_j = energy_rows.get('total_energy_kj', [0]).iloc[0] * 1000  # Convert kJ to J
                duration_s = energy_rows.get('duration_s', [1]).iloc[0]
                
                metrics = {
                    'avg_power_w': round(avg_power, 3),
                    'peak_power_w': round(peak_power, 3),
                    'min_power_w': round(min_power, 3),
                    'measurements_count': len(energy_rows)
                }
                
                # Add additional metrics if available in aggregated data
                for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct', 'min_gpu_usage_pct',
                                   'avg_ram_used_mb', 'peak_ram_used_mb', 'avg_ram_total_mb',
                                   'avg_ram_utilization_pct', 'peak_ram_utilization_pct',
                                   'avg_temp_gpu_c', 'peak_temp_gpu_c', 'min_temp_gpu_c']:
                    if metric_name in energy_rows.columns:
                        metrics[metric_name] = energy_rows[metric_name].iloc[0]
                
                return metrics
            
            else:
                print(f"Warning: No recognized power columns in energy data. Available columns: {energy_rows.columns.tolist()}")
                return self._get_zero_energy_metrics()
            
        except Exception as e:
            print(f"Error calculating energy metrics: {e}")
            return self._get_zero_energy_metrics()
    
    def _get_zero_energy_metrics(self) -> Dict[str, float]:
        """Return zero energy metrics for error cases."""
        return {
            'avg_power_w': 0.0,
            'peak_power_w': 0.0,
            'min_power_w': 0.0,
            'measurements_count': 0
        }
    
    def match_energy_to_performance_by_question(self, model_name: str, 
                                               performance_df: pd.DataFrame,
                                               energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match energy data to performance questions by exact question details.
        
        Args:
            model_name: Full model name
            performance_df: Performance data for this model
            energy_df: Processed energy data for this model
            
        Returns:
            Combined DataFrame with matched energy and performance data
        """
        print(f"  Matching energy to performance by question details for {model_name}...")
        
        def _norm(subject_str: str) -> str:
            """Normalize subject name for robust matching."""
            return subject_str.strip().lower().replace(' ', '_').replace('-', '_')

        combined_rows = []
        matched_count = 0
        
        # Create question-level energy metrics from the processed energy data
        question_energy_metrics = {}
        
        # Check if we have question-level data (with Question_Key) or just subject-level
        if 'Question_Key' in energy_df.columns:
            print(f"    Using question-level energy matching")
            # Group by Question_Key for precise matching
            for question_key in energy_df['Question_Key'].unique():
                question_energy_rows = energy_df[energy_df['Question_Key'] == question_key]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(question_energy_rows)
                question_energy_metrics[question_key] = energy_metrics
            
            # Also calculate subject-level averages as fallback
            subject_energy_metrics = {}
            for subject in energy_df['Subject'].unique():
                subject_energy_rows = energy_df[energy_df['Subject'] == subject]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(subject_energy_rows)
                norm_sub = _norm(str(subject))
                question_energy_metrics[norm_sub] = energy_metrics
                subject_energy_metrics[norm_sub] = energy_metrics
                
        else:
            print(f"    Falling back to subject-level energy matching")
            # Fallback to subject-level matching
            subject_energy_metrics = {}
            for subject in energy_df['Subject'].unique():
                subject_energy_rows = energy_df[energy_df['Subject'] == subject]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(subject_energy_rows)
                norm_sub = _norm(str(subject))
                question_energy_metrics[norm_sub] = energy_metrics
                subject_energy_metrics[norm_sub] = energy_metrics
        
        print(f"    Calculated energy metrics for {len(question_energy_metrics)} questions/subjects")
        
        # Match each performance question to energy metrics
        for idx, perf_row in performance_df.iterrows():
            subject = _norm(str(perf_row['subject']))
            question_id = perf_row['question_id']
            input_tokens = int(perf_row['input_tokens'])
            output_tokens = int(perf_row['output_tokens'])
            
            # Create question key to match against energy data
            # Since performance data has question_id=0, we need to match by subject + tokens
            # Energy data has format: subject_qXXX_inYYY_outZZZ
            # We'll try multiple matching strategies
            
            energy_metrics = None
            matching_method = None
            
            # Strategy 1: Try to find exact match by subject and tokens (ignoring question_id)
            for energy_key in question_energy_metrics.keys():
                if energy_key.startswith(f"{subject}_") and f"_in{input_tokens}_out{output_tokens}" in energy_key:
                    energy_metrics = question_energy_metrics[energy_key]
                    matching_method = 'exact_tokens'
                    matched_count += 1
                    break
            
            # Strategy 2: If no exact match, find closest input token match for the same subject
            if energy_metrics is None:
                best_match_key = None
                min_token_diff = float('inf')
                
                for energy_key in question_energy_metrics.keys():
                    if energy_key.startswith(f"{subject}_") and f"_out{output_tokens}" in energy_key:
                        # Extract input tokens from energy key
                        match = re.search(r'_in(\d+)_out', energy_key)
                        if match:
                            energy_input_tokens = int(match.group(1))
                            token_diff = abs(energy_input_tokens - input_tokens)
                            if token_diff < min_token_diff:
                                min_token_diff = token_diff
                                best_match_key = energy_key
                
                if best_match_key:
                    energy_metrics = question_energy_metrics[best_match_key]
                    matching_method = f'closest_tokens_diff_{min_token_diff}'
                    matched_count += 1
            
            # Strategy 3: If still no match, try subject-level matching
            if energy_metrics is None and subject in subject_energy_metrics:
                energy_metrics = subject_energy_metrics[subject]
                matching_method = 'subject_average'
                matched_count += 1
            
            # Strategy 4: If still no match, return zeros
            if energy_metrics is None:
                question_key = f"{subject}_{question_id}_in{input_tokens}_out{output_tokens}"
                print(f"    Warning: No energy data found for {subject} with {input_tokens} input tokens and {output_tokens} output tokens")
                energy_metrics = self._get_zero_energy_metrics()
                matching_method = 'no_match'
            
            # Create combined row
            combined_row = {
                # Model and question info
                'model_name': model_name,
                'subject': subject,
                'question_id': question_id,
                'question': perf_row['question'],
                'correct_answer': perf_row['correct_answer'],
                'predicted_choice': perf_row['predicted_choice'],
                'is_correct': perf_row['is_correct'],
                
                # Performance metrics
                'ttft_ms': perf_row['ttft'],
                'decode_time_ms': perf_row['decode_time'],
                'total_time_ms': perf_row['total_time_ms'],
                'tokens_per_second': perf_row['tokens_per_second'],
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                
                # Energy metrics
                'avg_power_w': energy_metrics['avg_power_w'],
                'peak_power_w': energy_metrics['peak_power_w'],
                'min_power_w': energy_metrics['min_power_w'],
                'energy_measurements': energy_metrics['measurements_count'],
                
                # Calculate energy using performance timing data (convert ms to seconds)
                'duration_s': round(perf_row['total_time_ms'] / 1000.0, 3),
                'total_energy_j': round(energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0), 3),
                
                # Efficiency metrics
                'energy_per_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / 
                                   (input_tokens + output_tokens) 
                                   if (input_tokens + output_tokens) > 0 else 0),
                'energy_per_input_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / input_tokens 
                                         if input_tokens > 0 else 0),
                'energy_per_output_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / output_tokens 
                                          if output_tokens > 0 else 0),
                'power_per_token_per_second': (energy_metrics['avg_power_w'] / perf_row['tokens_per_second'] 
                                             if perf_row['tokens_per_second'] > 0 else 0),
                
                # Matching info
                'energy_matched_by': matching_method,
                'energy_question_key': f"{subject}_{question_id}_in{input_tokens}_out{output_tokens}",
                'energy_found': matching_method != 'no_match'
            }
            
            # Add GPU utilization metrics if available
            for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct', 'min_gpu_usage_pct']:
                if metric_name in energy_metrics:
                    combined_row[metric_name] = energy_metrics[metric_name]
            
            # Add memory usage metrics if available
            for metric_name in ['avg_ram_used_mb', 'peak_ram_used_mb', 'avg_ram_total_mb',
                               'avg_ram_utilization_pct', 'peak_ram_utilization_pct']:
                if metric_name in energy_metrics:
                    combined_row[metric_name] = energy_metrics[metric_name]
            
            # Add temperature metrics if available
            for metric_name in ['avg_temp_gpu_c', 'peak_temp_gpu_c', 'min_temp_gpu_c']:
                if metric_name in energy_metrics:
                    combined_row[metric_name] = energy_metrics[metric_name]
            
            combined_rows.append(combined_row)
        
        exact_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'exact_tokens')
        subject_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'subject_average')
        
        print(f"    Successfully matched {matched_count}/{len(performance_df)} questions")
        print(f"      Exact token matches: {exact_matches}")
        print(f"      Subject-level matches: {subject_matches}")
        
        return pd.DataFrame(combined_rows)
    
    def generate_correlation_analysis(self) -> pd.DataFrame:
        """
        Generate comprehensive correlation analysis between energy and performance.
        
        Returns:
            Combined DataFrame with all models and questions
        """
        print("🔄 Starting Energy-Performance Correlation Analysis")
        print("="*60)
        
        # Load data
        self.performance_data = self.load_performance_data()
        energy_data = self.load_processed_energy_data()
        
        if not energy_data:
            print("❌ No processed energy data found. Please run basic energy analysis first.")
            return pd.DataFrame()
        
        all_combined_data = []
        
        # Process each model
        for model_name in sorted(self.performance_data.keys()):
            if model_name not in energy_data:
                print(f"Warning: No energy data found for model {model_name}")
                continue
            
            performance_df = self.performance_data[model_name]
            energy_df = energy_data[model_name]
            
            # Match energy data to performance questions
            combined_df = self.match_energy_to_performance_by_question(model_name, performance_df, energy_df)
            all_combined_data.append(combined_df)
        
        # Combine all models
        if all_combined_data:
            final_df = pd.concat(all_combined_data, ignore_index=True)
            
            # Print matching statistics
            total_questions = len(final_df)
            matched_questions = len(final_df[final_df['energy_found'] == True])
            zero_power_questions = len(final_df[final_df['avg_power_w'] == 0])
            
            print(f"\n✅ Correlation complete!")
            print(f"Total questions analyzed: {total_questions}")
            print(f"Successfully matched: {matched_questions}/{total_questions} ({matched_questions/total_questions*100:.1f}%)")
            print(f"Zero power questions: {zero_power_questions} (should be 0!)")
            print(f"Models: {final_df['model_name'].nunique()}")
            print(f"Subjects: {final_df['subject'].nunique()}")
            
            return final_df
        
        return pd.DataFrame()
    
    def generate_summary_statistics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for energy-performance correlation."""
        if combined_df.empty:
            return pd.DataFrame()
        
        print("Generating summary statistics...")
        
        # Model-level summary - base metrics
        agg_dict = {
            'is_correct': 'mean',  # accuracy
            'total_time_ms': 'mean',
            'tokens_per_second': 'mean',
            'total_tokens': 'sum',
            'avg_power_w': 'mean',
            'total_energy_j': 'sum',
            'energy_per_token': 'mean',
            'power_per_token_per_second': 'mean',
            'question_id': 'count'  # question count
        }
        
        # Add GPU utilization metrics if available
        for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        
        # Add memory usage metrics if available
        for metric_name in ['avg_ram_used_mb', 'avg_ram_utilization_pct', 'peak_ram_utilization_pct']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        
        # Add temperature metrics if available
        for metric_name in ['avg_temp_gpu_c', 'peak_temp_gpu_c']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        
        model_summary = combined_df.groupby('model_name').agg(agg_dict).round(4)
        
        # Build column names dynamically based on what metrics are available
        base_columns = [
            'accuracy', 'avg_time_ms', 'avg_tokens_per_sec', 'total_tokens',
            'avg_power_w', 'total_energy_j', 'avg_energy_per_token',
            'avg_power_per_token_per_sec', 'total_questions'
        ]
        
        # Add names for additional metrics that were included
        additional_columns = []
        for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct', 'avg_ram_used_mb', 
                           'avg_ram_utilization_pct', 'peak_ram_utilization_pct', 
                           'avg_temp_gpu_c', 'peak_temp_gpu_c']:
            if metric_name in agg_dict:
                additional_columns.append(metric_name)
        
        model_summary.columns = base_columns + additional_columns
        
        # Add efficiency rankings
        model_summary['energy_efficiency_rank'] = model_summary['avg_energy_per_token'].rank()
        model_summary['power_efficiency_rank'] = model_summary['avg_power_per_token_per_sec'].rank()
        model_summary['speed_rank'] = model_summary['avg_tokens_per_sec'].rank(ascending=False)
        model_summary['accuracy_rank'] = model_summary['accuracy'].rank(ascending=False)
        
        return model_summary.reset_index()
    
    def save_correlation_results(self, combined_df: pd.DataFrame, 
                               summary_df: pd.DataFrame) -> str:
        """Save correlation results to Excel file."""
        output_path = PathManager.get_output_path('energy_performance_correlation.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
            
            # Detailed data by model
            for model_name in sorted(combined_df['model_name'].unique()):
                model_data = combined_df[combined_df['model_name'] == model_name]
                
                # Clean sheet name
                sheet_name = model_name.replace('-', '_')[:31]
                model_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Subject-level analysis
            subject_summary = combined_df.groupby(['subject', 'model_name']).agg({
                'is_correct': 'mean',
                'avg_power_w': 'mean',
                'energy_per_token': 'mean',
                'tokens_per_second': 'mean'
            }).round(4).reset_index()
            
            subject_summary.to_excel(writer, sheet_name='Subject_Analysis', index=False)
        
        print(f"Correlation results saved to: {output_path}")
        return output_path


def main():
    """Main function to run energy-performance correlation."""
    # Default paths
    energy_dir = "./tegra"
    performance_file = "./processed_results/all_results_by_model_20250624_133750.xlsx"
    
    # Check if files exist
    if not os.path.exists(energy_dir):
        print(f"Error: Energy directory not found: {energy_dir}")
        return
    
    if not os.path.exists(performance_file):
        print(f"Error: Performance file not found: {performance_file}")
        return
    
    # Create correlator
    correlator = EnergyPerformanceCorrelator(energy_dir, performance_file)
    
    # Generate correlation analysis
    combined_df = correlator.generate_correlation_analysis()
    
    if combined_df.empty:
        print("No correlation data generated")
        return
    
    # Generate summary statistics
    summary_df = correlator.generate_summary_statistics(combined_df)
    
    # Save results
    output_path = correlator.save_correlation_results(combined_df, summary_df)
    
    print("\n🎉 Energy-Performance Correlation Complete!")
    print("="*60)
    print(f"📊 Generated comprehensive analysis with {len(combined_df)} question-level correlations")
    print(f"📁 Results saved to: {output_path}")
    print("\n💡 The Excel file contains:")
    print("  • Model Summary: Overall efficiency rankings")
    print("  • Individual Model Sheets: Question-level data")
    print("  • Subject Analysis: Performance by subject")


if __name__ == "__main__":
    main() 