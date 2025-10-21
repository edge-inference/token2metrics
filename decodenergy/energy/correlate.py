# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from energy.utils import PathManager, save_dataframe, extract_token_length


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
        """Load performance data from an Excel or CSV file."""
        print("Loading performance data...")

        performance_data: Dict[str, pd.DataFrame] = {}

        if self.performance_file.lower().endswith('.csv'):
            # Consolidated CSV with all models
            df_all = pd.read_csv(self.performance_file)
            if 'model_name' not in df_all.columns:
                raise ValueError("CSV performance file must contain 'model_name' column")
            for model_name, df_model in df_all.groupby('model_name'):
                performance_data[model_name] = df_model.reset_index(drop=True)
                print(f"  Loaded performance data for: {model_name} (CSV)")
            print(f"Loaded performance data for {len(performance_data)} models from CSV")
            return performance_data

        # Default: assume Excel workbook
        xl_file = pd.ExcelFile(self.performance_file)
        model_sheets = [sheet for sheet in xl_file.sheet_names
                       if sheet not in ['Overview', 'Subject_Comparison']]
        for sheet in model_sheets:
            print(f"  Loading performance data for: {sheet}")
            df = pd.read_excel(self.performance_file, sheet_name=sheet)
            model_name = sheet.replace('_', '-').replace('1-5B', '1.5B')
            performance_data[model_name] = df
        print(f"Loaded performance data for {len(performance_data)} models from Excel")
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
            print("Please run basic energy analysis first: python -m energy.cli --base-dir ../datasets/synthetic/gpu/decode/fine")
            return {}
        
        # Read the Excel file to get available sheets
        xl_file = pd.ExcelFile(energy_summary_path)
        
        # Skip the Summary sheet, load individual model sheets
        model_sheets = [sheet for sheet in xl_file.sheet_names if sheet != 'Summary']
        
        energy_data = {}
        for sheet in model_sheets:
            # Skip Max model as it's identical to 1.5B
            if sheet == 'Max':
                print(f"  Skipping {sheet} (duplicate of 1.5B model)")
                continue
                
            print(f"  Loading energy data for: {sheet}")
            df = pd.read_excel(energy_summary_path, sheet_name=sheet)
            
            # Map short name to full name
            full_model_name = self.model_name_mapping.get(sheet, sheet)
            energy_data[full_model_name] = df
            
            print(f"    Loaded {len(df)} energy measurements for {full_model_name}")
        
        print(f"Loaded energy data for {len(energy_data)} models")
        return energy_data
    
    def parse_energy_filename_from_data(self, subject: str, filepath_or_data: str) -> Dict[str, str]:
        """
        Parse energy metadata from subject and any available filename info.
        Since we're using processed data, we'll extract what we can from the subject.
        """

        return {
            'subject': subject,
            'question_id': None,
            'input_tokens': None,
            'output_tokens': None  
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
            # Calculate metrics from the processed data
            avg_power = energy_rows['Power_W'].mean()
            peak_power = energy_rows['Power_W'].max()
            min_power = energy_rows['Power_W'].min()
            
            # Calculate total energy from cumulative energy or energy increments
            if 'Cumulative_Energy' in energy_rows.columns:
                total_energy_j = energy_rows['Cumulative_Energy'].iloc[-1]
            elif 'Energy_Increment' in energy_rows.columns:
                total_energy_j = energy_rows['Energy_Increment'].sum()
            else:
                # Fallback: estimate from power and time
                if 'Elapsed_Seconds' in energy_rows.columns:
                    duration_s = energy_rows['Elapsed_Seconds'].iloc[-1] - energy_rows['Elapsed_Seconds'].iloc[0]
                else:
                    duration_s = len(energy_rows)  
                total_energy_j = avg_power * duration_s
            
            # Duration calculation
            if 'Elapsed_Seconds' in energy_rows.columns:
                duration_s = energy_rows['Elapsed_Seconds'].iloc[-1] - energy_rows['Elapsed_Seconds'].iloc[0]
            else:
                duration_s = len(energy_rows)
            
            return {
                'avg_power_w': round(avg_power, 3),
                'peak_power_w': round(peak_power, 3),
                'min_power_w': round(min_power, 3),
                'total_energy_j': round(total_energy_j, 3),
                'duration_s': round(duration_s, 3),
                'measurements_count': len(energy_rows)
            }
            
        except Exception as e:
            print(f"Error calculating energy metrics: {e}")
            return self._get_zero_energy_metrics()
    
    def _get_zero_energy_metrics(self) -> Dict[str, float]:
        """Return zero energy metrics for error cases."""
        return {
            'avg_power_w': 0.0,
            'peak_power_w': 0.0,
            'min_power_w': 0.0,
            'total_energy_j': 0.0,
            'duration_s': 0.0,
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
        print(f"  Processing {model_name}...")
        
        combined_rows = []
        matched_count = 0
        
        question_energy_metrics = {}
        

        
        if 'Question_Key' in energy_df.columns:

            for question_key in energy_df['Question_Key'].unique():
                question_energy_rows = energy_df[energy_df['Question_Key'] == question_key]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(question_energy_rows)
                question_energy_metrics[question_key] = energy_metrics
            
            subject_energy_metrics = {}
            for length in energy_df['length'].unique():
                length_energy_rows = energy_df[energy_df['length'] == length]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(length_energy_rows)
                subject_energy_metrics[length] = energy_metrics
                
        else:
            subject_energy_metrics = {}
            available_lengths = energy_df['length'].unique() if 'length' in energy_df.columns else []
            base_subjects = {}
            for length in available_lengths:
                if length not in base_subjects:
                    base_subjects[length] = []
                length_energy_rows = energy_df[energy_df['length'] == length]
                base_subjects[length].append(length_energy_rows)
            
            for base_subject, rows_list in base_subjects.items():
                all_rows = pd.concat(rows_list, ignore_index=True) if rows_list else pd.DataFrame()
                energy_metrics = self.calculate_energy_metrics_from_processed_data(all_rows)
                subject_energy_metrics[base_subject] = energy_metrics
            question_energy_metrics.update(subject_energy_metrics)

        

        
        # Match each performance question to energy metrics
        for idx, perf_row in performance_df.iterrows():
            subject = perf_row['subject']
            question_id = perf_row['question_id']
            input_tokens = int(perf_row['input_tokens'])
            output_tokens = int(perf_row['output_tokens'])
            
            
            energy_metrics = None
            matching_method = None
            
            # Strategy 1: Try exact token length matching
            expected_length = f"in_{input_tokens}_out_{output_tokens}"
            if expected_length in question_energy_metrics:
                energy_metrics = question_energy_metrics[expected_length]
                matching_method = 'exact_tokens'
                matched_count += 1
            
            # Strategy 2: Find closest energy configuration within ±10 tokens tolerance
            if energy_metrics is None:
                best_match = None
                min_total_diff = float('inf')
                
                for energy_key in question_energy_metrics.keys():
                    # Extract input and output tokens from energy key like "in_768_out_1024"
                    import re
                    match = re.match(r'in_(\d+)_out_(\d+)', energy_key)
                    if match:
                        energy_input = int(match.group(1))
                        energy_output = int(match.group(2))
                        
                        input_diff = abs(energy_input - input_tokens)
                        output_diff = abs(energy_output - output_tokens)
                        
                        # Only consider matches within ±10 tokens tolerance
                        if input_diff <= 10 and output_diff <= 10:
                            total_diff = input_diff + output_diff
                            
                            if total_diff < min_total_diff:
                                min_total_diff = total_diff
                                best_match = energy_key
                
                if best_match:
                    energy_metrics = question_energy_metrics[best_match]
                    matching_method = f'within_10_tokens_diff_{min_total_diff}'
                    matched_count += 1
            
            # Strategy 4: If still no match, return zeros
            if energy_metrics is None:
                energy_metrics = self._get_zero_energy_metrics()
                matching_method = 'no_match'
            
            combined_row = {
                'model_name': model_name,
                
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
                'total_energy_j': energy_metrics['total_energy_j'],
                'duration_s': energy_metrics['duration_s'],
                'energy_measurements': energy_metrics['measurements_count'],
                
                # Efficiency metrics
                'energy_per_token': (energy_metrics['total_energy_j'] / 
                                   (input_tokens + output_tokens) 
                                   if (input_tokens + output_tokens) > 0 else 0),
                'energy_per_input_token': (energy_metrics['total_energy_j'] / input_tokens 
                                         if input_tokens > 0 else 0),
                'energy_per_output_token': (energy_metrics['total_energy_j'] / output_tokens 
                                          if output_tokens > 0 else 0),
                
                # Matching info
                'energy_matched_by': matching_method,
                'energy_question_key': f"in_{input_tokens}_out_{output_tokens}",
                'energy_found': matching_method != 'no_match'
            }
            
            combined_rows.append(combined_row)
        
        exact_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'exact_tokens')
        subject_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'subject_average')
        
        matched_pct = (matched_count / len(performance_df) * 100) if len(performance_df) > 0 else 0
        print(f"    Matched {matched_count}/{len(performance_df)} questions ({matched_pct:.1f}%)")
        
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
            
            total_questions = len(final_df)
            matched_questions = len(final_df[final_df['energy_found'] == True])
            zero_power_questions = len(final_df[final_df['avg_power_w'] == 0])
            
            print(f"\n✅ Correlation complete!")
            print(f"Total questions analyzed: {total_questions}")
            print(f"Matched: {matched_questions}/{total_questions} ({matched_questions/total_questions*100:.1f}%)")
            print(f"Models: {final_df['model_name'].nunique()}")
            
            return final_df
        
        return pd.DataFrame()
    
    def generate_summary_statistics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for energy-performance correlation."""
        if combined_df.empty:
            return pd.DataFrame()
        
        print("Generating summary statistics...")
        
        model_summary = combined_df.groupby('model_name').agg({
            'total_time_ms': 'mean',
            'tokens_per_second': 'mean',
            'total_tokens': 'sum',
            'avg_power_w': 'mean',
            'total_energy_j': 'sum',
            'energy_per_token': 'mean',
            'energy_question_key': 'count'  
        }).round(4)
        
        model_summary.columns = [
            'avg_time_ms', 'avg_tokens_per_sec', 'total_tokens',
            'avg_power_w', 'total_energy_j', 'avg_energy_per_token',
            'total_questions'
        ]
        
        model_summary['energy_efficiency_rank'] = model_summary['avg_energy_per_token'].rank()
        model_summary['power_efficiency_rank'] = model_summary['avg_power_w'].rank()
        model_summary['speed_rank'] = model_summary['avg_tokens_per_sec'].rank(ascending=False)
        
        return model_summary.reset_index()
    
    def save_correlation_results(self, combined_df: pd.DataFrame, 
                               summary_df: pd.DataFrame) -> str:
        """Save correlation results to Excel file."""
        output_path = PathManager.get_output_path('energy_performance_correlation.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
            
            for model_name in sorted(combined_df['model_name'].unique()):
                model_data = combined_df[combined_df['model_name'] == model_name]
                
                sheet_name = model_name.replace('-', '_')[:31]
                model_data.to_excel(writer, sheet_name=sheet_name, index=False)
            

        
        print(f"Correlation results saved to: {output_path}")
        return output_path


def main():
    """Main function to run energy-performance correlation."""
    energy_dir = "./tegra"
    performance_file = "./dataset/all_results_by_model_20250628_042635.xlsx"
    
    if not os.path.exists(energy_dir):
        print(f"Error: Energy directory not found: {energy_dir}")
        return
    
    if not os.path.exists(performance_file):
        print(f"Error: Performance file not found: {performance_file}")
        return
    
    correlator = EnergyPerformanceCorrelator(energy_dir, performance_file)
    
    combined_df = correlator.generate_correlation_analysis()
    
    if combined_df.empty:
        print("No correlation data generated")
        return
    
    summary_df = correlator.generate_summary_statistics(combined_df)
    
    output_path = correlator.save_correlation_results(combined_df, summary_df)
    
    print("\n🎉 Energy-Performance Correlation Complete!")
    print("="*60)
    print(f"📊 Generated comprehensive analysis with {len(combined_df)} question-level correlations")
    print(f"📁 Results saved to: {output_path}")
    print("\n💡 The Excel file contains:")
    print("  • Model Summary: Overall efficiency rankings")
    print("  • Individual Model Sheets: Question-level data")


if __name__ == "__main__":
    main() 