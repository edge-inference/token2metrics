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

from .utils import PathManager, save_dataframe

class EnergyPerformanceCorrelator:
    def __init__(self, energy_base_dir: str, performance_file: str):
        self.energy_base_dir = energy_base_dir
        self.performance_file = performance_file
        self.energy_data = {}
        self.performance_data = {}
        self.model_name_mapping = {
            '8B': 'DeepSeek-R1-Distill-Llama-8B',
            '1.5B': 'DeepSeek-R1-Distill-Qwen-1.5B', 
            '14B': 'DeepSeek-R1-Distill-Qwen-14B'
        }
        
    def get_energy_model_name(self, performance_model_name: str) -> str:
        return next((k for k, v in self.model_name_mapping.items() if v == performance_model_name), '1.5B' if '1_5B' in performance_model_name else performance_model_name)
        
    def load_performance_data(self) -> Dict[str, pd.DataFrame]:
        print("Loading performance data...")
        try:
            xl_file = pd.read_excel(self.performance_file, sheet_name=None, engine='openpyxl')
            model_sheets = list(xl_file.keys())
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            model_sheets = []
        performance_data = {}
        for sheet in model_sheets:
            if sheet in ['Overview', 'Subject_Comparison'] or not sheet.startswith('DeepSeek'):
                continue
            print(f"  Loading performance data for: {sheet}")
            df = pd.read_excel(self.performance_file, sheet_name=sheet, engine='openpyxl')
            model_name = sheet.replace('_', '-').replace('1-5B', '1.5B')
            performance_data[model_name] = df
        print(f"Loaded performance data for {len(performance_data)} models")
        return performance_data
    
    def load_processed_energy_data(self) -> Dict[str, pd.DataFrame]:
        print("Loading processed energy data from energy_model_summary.xlsx...")
        energy_summary_path = PathManager.get_output_path('energy_detailed_results.xlsx')
        print(f"Looking for energy summary at: {energy_summary_path}")
        if not energy_summary_path.exists():
            print(f"Error: Energy summary file not found: {energy_summary_path}")
            print("Please run basic energy analysis first: python -m energy.cli --base-dir ./tegra/figure2")
            return {}
        try:
            df = pd.read_excel(energy_summary_path, engine='openpyxl')
            energy_data = {}
            for model_name in df['Model'].unique():
                model_df = df[df['Model'] == model_name].copy()
                energy_data[model_name] = model_df
            return energy_data
        except Exception as e:
            print(f"Error loading energy data: {e}")
            return {}
        energy_data = {}
        for sheet in model_sheets:
            if sheet == 'Max':
                print(f"  Skipping {sheet} (duplicate of 1.5B model)")
                continue
            print(f"  Loading energy data for: {sheet}")
            df = pd.read_excel(energy_summary_path, sheet_name=sheet)
            full_model_name = self.model_name_mapping.get(sheet, sheet)
            energy_data[full_model_name] = df
            print(f"    Loaded {len(df)} energy measurements for {full_model_name}")
        print(f"Loaded energy data for {len(energy_data)} models")
        return energy_data
    
    def parse_energy_filename_from_data(self, subject: str, filepath_or_data: str) -> Dict[str, str]:
        return {
            'subject': subject,
            'question_id': None,
            'input_tokens': None,
            'output_tokens': None
        }
    
    def calculate_energy_metrics_from_processed_data(self, energy_rows: pd.DataFrame) -> Dict[str, float]:
        if len(energy_rows) == 0:
            return self._get_zero_energy_metrics()
        try:
            avg_power = energy_rows['avg_power_w'].mean()
            peak_power = energy_rows['peak_power_w'].mean()
            min_power = energy_rows['min_power_w'].mean()
            if 'Cumulative_Energy' in energy_rows.columns:
                total_energy_j = energy_rows['Cumulative_Energy'].iloc[-1]
            elif 'Energy_Increment' in energy_rows.columns:
                total_energy_j = energy_rows['Energy_Increment'].sum()
            else:
                if 'Elapsed_Seconds' in energy_rows.columns:
                    duration_s = energy_rows['Elapsed_Seconds'].iloc[-1] - energy_rows['Elapsed_Seconds'].iloc[0]
                else:
                    duration_s = len(energy_rows)
                total_energy_j = avg_power * duration_s
            metrics = {
                'avg_power_w': round(avg_power, 3),
                'peak_power_w': round(peak_power, 3),
                'min_power_w': round(min_power, 3),
                'measurements_count': len(energy_rows)
            }
            if 'gpu_usage_pct' in energy_rows.columns:
                metrics['avg_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].mean(), 2)
                metrics['peak_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].max(), 2)
                metrics['min_gpu_usage_pct'] = round(energy_rows['gpu_usage_pct'].min(), 2)
            if 'ram_total_mb' in energy_rows.columns and 'ram_used_mb' in energy_rows.columns:
                metrics['avg_ram_used_mb'] = round(energy_rows['ram_used_mb'].mean(), 1)
                metrics['peak_ram_used_mb'] = round(energy_rows['ram_used_mb'].max(), 1)
                metrics['avg_ram_total_mb'] = round(energy_rows['ram_total_mb'].mean(), 1)
                ram_utilization_pct = (energy_rows['ram_used_mb'] / energy_rows['ram_total_mb']) * 100
                metrics['avg_ram_utilization_pct'] = round(ram_utilization_pct.mean(), 2)
                metrics['peak_ram_utilization_pct'] = round(ram_utilization_pct.max(), 2)
            if 'temp_gpu_c' in energy_rows.columns:
                metrics['avg_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].mean(), 1)
                metrics['peak_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].max(), 1)
                metrics['min_temp_gpu_c'] = round(energy_rows['temp_gpu_c'].min(), 1)
            return metrics
        except Exception as e:
            print(f"Error calculating energy metrics: {e}")
            return self._get_zero_energy_metrics()
    
    def _get_zero_energy_metrics(self) -> Dict[str, float]:
        return {
            'avg_power_w': 0.0,
            'peak_power_w': 0.0,
            'min_power_w': 0.0,
            'measurements_count': 0
        }
    
    def match_energy_to_performance_by_question(self, model_name: str, 
                                               performance_df: pd.DataFrame,
                                               energy_df: pd.DataFrame) -> pd.DataFrame:
        print(f"  Matching energy to performance by question details for {model_name}...")
        combined_rows = []
        matched_count = 0
        question_energy_metrics = {}
        if 'Question_Key' in energy_df.columns:
            print(f"    Using question-level energy matching")
            for question_key in energy_df['Question_Key'].unique():
                question_energy_rows = energy_df[energy_df['Question_Key'] == question_key]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(question_energy_rows)
                question_energy_metrics[question_key] = energy_metrics
            subject_energy_metrics = {}
            for subject in energy_df['Subject'].unique():
                subject_energy_rows = energy_df[energy_df['Subject'] == subject]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(subject_energy_rows)
                subject_energy_metrics[subject] = energy_metrics
        else:
            print(f"    Falling back to subject-level energy matching")
            subject_energy_metrics = {}
            for subject in energy_df['Subject'].unique():
                subject_energy_rows = energy_df[energy_df['Subject'] == subject]
                energy_metrics = self.calculate_energy_metrics_from_processed_data(subject_energy_rows)
                question_energy_metrics[subject] = energy_metrics
                subject_energy_metrics[subject] = energy_metrics
        print(f"    Calculated energy metrics for {len(question_energy_metrics)} questions/subjects")
        for idx, perf_row in performance_df.iterrows():
            subject = perf_row['subject']
            question_id = perf_row['question_id']
            input_tokens = int(perf_row['input_tokens'])
            output_tokens = 1
            energy_metrics = None
            matching_method = None
            for energy_key in question_energy_metrics.keys():
                if f"energy_{subject}_" in energy_key and f"_in{input_tokens}_out{output_tokens}" in energy_key:
                    energy_metrics = question_energy_metrics[energy_key]
                    matching_method = 'exact_tokens'
                    matched_count += 1
                    break
            if energy_metrics is None:
                best_match_key = None
                min_token_diff = float('inf')
                for energy_key in question_energy_metrics.keys():
                    if f"energy_{subject}_" in energy_key and f"_out{output_tokens}" in energy_key:
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
            if energy_metrics is None:
                if subject in subject_energy_metrics:
                    energy_metrics = subject_energy_metrics[subject]
                    matching_method = 'subject_average'
                    matched_count += 1
                else:
                    for energy_subject in subject_energy_metrics.keys():
                        if subject in energy_subject:
                            energy_metrics = subject_energy_metrics[energy_subject]
                            matching_method = 'subject_partial_match'
                            matched_count += 1
                            break
            if energy_metrics is None:
                question_key = f"{subject}_{question_id}_in{input_tokens}_out{output_tokens}"
                print(f"    Warning: No energy data found for {subject} with {input_tokens} input tokens and {output_tokens} output tokens")
                energy_metrics = {
                    'avg_power_w': 0.0,
                    'peak_power_w': 0.0,
                    'min_power_w': 0.0,
                    'measurements_count': 0,
                    'total_energy_j': 0.0,
                    'duration_s': 0.0
                }
                matching_method = 'no_match'
            try:
                combined_row = {
                    'model_name': model_name,
                    'subject': subject,
                    'question_id': question_id,
                    'question': perf_row.get('question', 'Unknown'),
                    'correct_answer': perf_row.get('correct_answer', 'Unknown'),
                    'predicted_choice': perf_row.get('predicted_choice', 'Unknown'),
                    'is_correct': perf_row.get('is_correct', False),
                    'ttft_ms': float(perf_row['ttft']),
                    'decode_time_ms': perf_row.get('decode_time', 0),
                    'total_time_ms': float(perf_row['total_time_ms']),
                    'tokens_per_second': float(perf_row['tokens_per_second']),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'avg_power_w': energy_metrics['avg_power_w'],
                    'peak_power_w': energy_metrics['peak_power_w'],
                    'min_power_w': energy_metrics['min_power_w'],
                    'energy_measurements': energy_metrics['measurements_count'],
                    'duration_s': round(perf_row['total_time_ms'] / 1000.0, 3),
                    'total_energy_j': round(energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0), 3),
                    'energy_per_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / 
                                   (input_tokens + output_tokens) 
                                   if (input_tokens + output_tokens) > 0 else 0),
                    'energy_per_input_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / input_tokens 
                                         if input_tokens > 0 else 0),
                    'energy_per_output_token': (energy_metrics['avg_power_w'] * (perf_row['total_time_ms'] / 1000.0) / output_tokens 
                                          if output_tokens > 0 else 0),
                    'power_per_token_per_second': (energy_metrics['avg_power_w'] / perf_row['tokens_per_second'] 
                                             if perf_row['tokens_per_second'] > 0 else 0),
                    'energy_matched_by': matching_method,
                    'energy_question_key': f"{subject}_{question_id}_in{input_tokens}_out{output_tokens}",
                    'energy_found': matching_method != 'no_match'
                }
                for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct', 'min_gpu_usage_pct']:
                    if metric_name in energy_metrics:
                        combined_row[metric_name] = energy_metrics[metric_name]
                for metric_name in ['avg_ram_used_mb', 'peak_ram_used_mb', 'avg_ram_total_mb',
                                   'avg_ram_utilization_pct', 'peak_ram_utilization_pct']:
                    if metric_name in energy_metrics:
                        combined_row[metric_name] = energy_metrics[metric_name]
                for metric_name in ['avg_temp_gpu_c', 'peak_temp_gpu_c', 'min_temp_gpu_c']:
                    if metric_name in energy_metrics:
                        combined_row[metric_name] = energy_metrics[metric_name]
                combined_rows.append(combined_row)
            except Exception as e:
                print(f"Error creating combined row: {e}")
                print("Combined row data:")
                print(f"  model_name: {model_name}")
                print(f"  subject: {subject}")
                print(f"  question_id: {question_id}")
                print(f"  total_time_ms: {perf_row['total_time_ms']}")
                print(f"  energy_metrics: {energy_metrics}")
        exact_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'exact_tokens')
        subject_matches = sum(1 for row in combined_rows if row['energy_matched_by'] == 'subject_average')
        print(f"    Successfully matched {matched_count}/{len(performance_df)} questions")
        print(f"      Exact token matches: {exact_matches}")
        print(f"      Subject-level matches: {subject_matches}")
        return pd.DataFrame(combined_rows)
    
    def generate_correlation_analysis(self) -> pd.DataFrame:
        print("Starting Energy-Performance Correlation Analysis")
        print("="*60)
        self.performance_data = self.load_performance_data()
        energy_data = self.load_processed_energy_data()
        if not energy_data:
            print("✗ No processed energy data found. Please run basic energy analysis first.")
            return pd.DataFrame()
        all_combined_data = []
        for model_name in sorted(self.performance_data.keys()):
            energy_model_name = self.get_energy_model_name(model_name)
            if energy_model_name not in energy_data:
                print(f"Warning: No energy data found for model {model_name} or {energy_model_name}")
                continue
            performance_df = self.performance_data[model_name]
            energy_df = energy_data[energy_model_name]
            combined_df = self.match_energy_to_performance_by_question(model_name, performance_df, energy_df)
            all_combined_data.append(combined_df)
        
        # Combine all models
        if all_combined_data:
            try:
                final_df = pd.concat(all_combined_data, ignore_index=True)
                total_questions = len(final_df)
                matched_questions = len(final_df[final_df['energy_found'] == True])
                zero_power_questions = len(final_df[final_df['avg_power_w'] == 0])
                print(f"\n✓ Correlation complete!")
                print(f"Total questions analyzed: {total_questions}")
                print(f"Successfully matched: {matched_questions}/{total_questions} ({matched_questions/total_questions*100:.1f}%)")
                print(f"Zero power questions: {zero_power_questions} (should be 0!)")
                print(f"Models: {final_df['model_name'].nunique()}")
                print(f"Subjects: {final_df['subject'].nunique()}")
                return final_df
            except Exception as e:
                print(f"Error in final DataFrame creation: {e}")
                print("First row of data:")
                print(all_combined_data[0][0] if all_combined_data[0] else "No data")
        return pd.DataFrame()
    
    def generate_summary_statistics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        if combined_df.empty:
            return pd.DataFrame()
        print("Generating summary statistics...")
        agg_dict = {
            'is_correct': 'mean',
            'total_time_ms': 'mean',
            'tokens_per_second': 'mean',
            'total_tokens': 'sum',
            'avg_power_w': 'mean',
            'total_energy_j': 'sum',
            'energy_per_token': 'mean',
            'power_per_token_per_second': 'mean',
            'question_id': 'count'
        }
        for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        for metric_name in ['avg_ram_used_mb', 'avg_ram_utilization_pct', 'peak_ram_utilization_pct']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        for metric_name in ['avg_temp_gpu_c', 'peak_temp_gpu_c']:
            if metric_name in combined_df.columns:
                agg_dict[metric_name] = 'mean'
        model_summary = combined_df.groupby('model_name').agg(agg_dict).round(4)
        base_columns = [
            'accuracy', 'avg_time_ms', 'avg_tokens_per_sec', 'total_tokens',
            'avg_power_w', 'total_energy_j', 'avg_energy_per_token',
            'avg_power_per_token_per_sec', 'total_questions'
        ]
        additional_columns = []
        for metric_name in ['avg_gpu_usage_pct', 'peak_gpu_usage_pct', 'avg_ram_used_mb', 
                           'avg_ram_utilization_pct', 'peak_ram_utilization_pct', 
                           'avg_temp_gpu_c', 'peak_temp_gpu_c']:
            if metric_name in agg_dict:
                additional_columns.append(metric_name)
        model_summary.columns = base_columns + additional_columns
        model_summary['energy_efficiency_rank'] = model_summary['avg_energy_per_token'].rank()
        model_summary['power_efficiency_rank'] = model_summary['avg_power_per_token_per_sec'].rank()
        model_summary['speed_rank'] = model_summary['avg_tokens_per_sec'].rank(ascending=False)
        model_summary['accuracy_rank'] = model_summary['accuracy'].rank(ascending=False)
        return model_summary.reset_index()
    
    def save_correlation_results(self, combined_df: pd.DataFrame, 
                               summary_df: pd.DataFrame) -> str:
        output_path = PathManager.get_output_path('energy_performance_correlation.xlsx')
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
            for model_name in sorted(combined_df['model_name'].unique()):
                model_data = combined_df[combined_df['model_name'] == model_name]
                sheet_name = model_name.replace('-', '_')[:31]
                model_data.to_excel(writer, sheet_name=sheet_name, index=False)
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
    energy_dir = "./tegra"
    performance_file = "./processed_results/all_results_by_model_20250624_133750.xlsx"
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
    print("  • Subject Analysis: Performance by subject")

if __name__ == "__main__":
    main() 
