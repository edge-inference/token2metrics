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
Core processor for scaling results parsing.

Extracts
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_structures import ScalingRunMetadata, ScalingMetrics, ParsedResult


class ScalingResultsProcessor:
    """
    Handles discovery, parsing, and organization of scaling test data
    from multiple result directories.
    """
    
    def __init__(self, results_base_dir: str = "./results"):
        self.results_base_dir = Path(results_base_dir)
        self.parsed_results: List[ParsedResult] = []
    
    def find_result_directories(self) -> List[Path]:
        """Find all result directories matching the scaling pattern."""
        if not self.results_base_dir.exists():
            print(f"❌ Base directory {self.results_base_dir} does not exist")
            return []
        
        pattern = re.compile(r'scale_synthetic_\d+_\d+_.+_\d+samples_\d+tokens_seed\d+')
        
        result_dirs = [
            d for d in self.results_base_dir.iterdir() 
            if d.is_dir() and pattern.match(d.name)
        ]
        
        return sorted(result_dirs)
    
    def parse_directory_name(self, dir_path: Path) -> Optional[ScalingRunMetadata]:
        """Parse metadata from directory name."""
        pattern = r'scale_synthetic_(\d{8})_(\d{6})_(.+)_(\d+)samples_(\d+)tokens_seed(\d+)'
        match = re.match(pattern, dir_path.name)
        
        if not match:
            print(f"❌ Cannot parse directory name: {dir_path.name}")
            return None
        
        date_part, time_part, model_name, num_samples, token_budget, seed = match.groups()
        timestamp = f"{date_part}_{time_part}"
        
        return ScalingRunMetadata(
            run_dir=str(dir_path),
            model_name=model_name,
            num_samples=int(num_samples),
            token_budget=int(token_budget),
            seed=int(seed),
            timestamp=timestamp
        )
    
    def parse_summary_json(self, dir_path: Path) -> Optional[Dict[str, Any]]:
        """Parse the summary.json file."""
        summary_file = dir_path / "summary.json"
        if not summary_file.exists():
            print(f"❌ No summary.json found in {dir_path.name}")
            return None
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error reading summary.json in {dir_path.name}: {e}")
            return None
    
    def parse_performance_csv(self, dir_path: Path) -> Optional[pd.DataFrame]:
        """Parse performance CSV file."""
        perf_files = list(dir_path.glob("*performance*.csv"))
        if not perf_files:
            return None
        
        try:
            return pd.read_csv(perf_files[0])
        except Exception as e:
            print(f"❌ Error reading performance CSV in {dir_path.name}: {e}")
            return None
    
    def parse_energy_csv(self, dir_path: Path) -> Optional[pd.DataFrame]:
        """Parse energy CSV file."""
        energy_files = list(dir_path.glob("*energy*.csv"))
        if not energy_files:
            return None
        
        try:
            return pd.read_csv(energy_files[0])
        except Exception as e:
            print(f"❌ Error reading energy CSV in {dir_path.name}: {e}")
            return None
    
    def parse_results_json(self, dir_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Parse detailed results JSON file."""
        results_files = list(dir_path.glob("*results*.json"))
        if not results_files:
            return None
        
        try:
            with open(results_files[0], 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'results' in data:
                    return data['results']
                else:
                    return [data]
        except Exception as e:
            print(f"❌ Error reading results JSON in {dir_path.name}: {e}")
            return None
    
    def parse_tegrastats_log(self, dir_path: Path) -> Optional[Dict[str, Any]]:
        """Parse tegrastats log for system metrics."""
        log_files = list(dir_path.glob("*tegrastats*.log"))
        if not log_files:
            return None
        
        try:
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
            
            ram_percentages = []
            cpu_percentages = []
            gpu_percentages = []
            
            for line in lines:
                ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
                if ram_match:
                    used, total = map(int, ram_match.groups())
                    ram_percentages.append((used / total) * 100)
                
                cpu_matches = re.findall(r'(\d+)%@\d+', line)
                if cpu_matches:
                    cpu_avg = np.mean([int(x) for x in cpu_matches])
                    cpu_percentages.append(cpu_avg)
                
                gpu_match = re.search(r'GPU (\d+)%@', line)
                if gpu_match:
                    gpu_percentages.append(int(gpu_match.group(1)))
            
            stats = {}
            if ram_percentages:
                stats['avg_ram_percent'] = np.mean(ram_percentages)
                stats['max_ram_percent'] = np.max(ram_percentages)
            if cpu_percentages:
                stats['avg_cpu_percent'] = np.mean(cpu_percentages)
                stats['max_cpu_percent'] = np.max(cpu_percentages)
            if gpu_percentages:
                stats['avg_gpu_percent'] = np.mean(gpu_percentages)
                stats['max_gpu_percent'] = np.max(gpu_percentages)
            
            return stats if stats else None
            
        except Exception as e:
            print(f"❌ Error parsing tegrastats in {dir_path.name}: {e}")
            return None
    
    def calculate_scaling_metrics(self, summary_data: Dict[str, Any], 
                                performance_data: Optional[pd.DataFrame],
                                energy_data: Optional[pd.DataFrame],
                                question_details: Optional[List[Dict[str, Any]]]) -> ScalingMetrics:
        """Calculate core scaling metrics from parsed data."""
        
        accuracy = summary_data.get('accuracy', 0.0)
        total_questions = summary_data.get('total_questions', 0)
        correct_answers = int(accuracy * total_questions) if total_questions > 0 else 0
        
        avg_ttft = None
        avg_decode_time = None
        avg_input_tokens = None
        avg_output_tokens = None
        avg_tokens_per_second = 0.0
        
        if performance_data is not None and not performance_data.empty:
            if 'ttft' in performance_data.columns:
                avg_ttft = performance_data['ttft'].mean()
            if 'decode_time' in performance_data.columns:
                avg_decode_time = performance_data['decode_time'].mean()
            if 'input_tokens' in performance_data.columns:
                avg_input_tokens = performance_data['input_tokens'].mean()
            if 'output_tokens' in performance_data.columns:
                avg_output_tokens = performance_data['output_tokens'].mean()
            if 'tokens_per_second' in performance_data.columns:
                avg_tokens_per_second = performance_data['tokens_per_second'].mean()
        
        avg_power_consumption = None
        total_energy_consumed = None
        
        if energy_data is not None and not energy_data.empty:
            power_columns = []
            if 'vdd_cpu_cv_avg_mw' in energy_data.columns:
                power_columns.append('vdd_cpu_cv_avg_mw')
            if 'vdd_gpu_soc_avg_mw' in energy_data.columns:
                power_columns.append('vdd_gpu_soc_avg_mw')
            if 'vin_sys_5v0_avg_mw' in energy_data.columns:
                power_columns.append('vin_sys_5v0_avg_mw')
            
            if power_columns:
                energy_data['total_power_mw'] = energy_data[power_columns].sum(axis=1)
                avg_power_consumption = energy_data['total_power_mw'].mean() / 1000.0
                
                time_interval_seconds = 1.0
                total_energy_consumed = (energy_data['total_power_mw'].sum() * time_interval_seconds) / 1000.0
        
        avg_time_per_question = summary_data.get('avg_time_per_question', 0.0)
        total_samples_generated = summary_data.get('total_samples_generated', 0)
        avg_voting_confidence = summary_data.get('avg_voting_confidence', 0.0)
        
        scaling_efficiency = None
        if total_samples_generated > 0:
            scaling_efficiency = accuracy / total_samples_generated
        
        return ScalingMetrics(
            accuracy=accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            avg_time_per_question=avg_time_per_question,
            avg_tokens_per_second=avg_tokens_per_second,
            total_samples_generated=total_samples_generated,
            avg_voting_confidence=avg_voting_confidence,
            scaling_efficiency=scaling_efficiency,
            avg_ttft=avg_ttft,
            avg_decode_time=avg_decode_time,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            avg_power_consumption=avg_power_consumption,
            total_energy_consumed=total_energy_consumed
        )
    
    def parse_single_result(self, dir_path: Path) -> Optional[ParsedResult]:
        """Parse a single result directory."""
        print(f"📋 Parsing {dir_path.name}")
        
        metadata = self.parse_directory_name(dir_path)
        if not metadata:
            return None
        
        # Parse all data files
        summary_data = self.parse_summary_json(dir_path)
        if not summary_data:
            return None
        
        performance_data = self.parse_performance_csv(dir_path)
        energy_data = self.parse_energy_csv(dir_path)
        question_details = self.parse_results_json(dir_path)
        system_stats = self.parse_tegrastats_log(dir_path)
        
        # Calculate metrics
        metrics = self.calculate_scaling_metrics(
            summary_data, performance_data, energy_data, question_details
        )
        
        # Create parsed result
        result = ParsedResult(
            metadata=metadata,
            metrics=metrics,
            question_details=question_details or [],
            performance_data=performance_data,
            energy_data=energy_data,
            system_stats=system_stats
        )
        
        return result
    
    def parse_all_results(self) -> List[ParsedResult]:
        """Parse all scaling results in the base directory."""
        print(f"🚀 Starting to parse scaling results from {self.results_base_dir}")
        
        result_dirs = self.find_result_directories()
        if not result_dirs:
            print("❌ No result directories found!")
            return []
        
        print(f"📁 Found {len(result_dirs)} scaling result directories")
        
        parsed_results = []
        successful_parses = 0
        
        for dir_path in result_dirs:
            try:
                result = self.parse_single_result(dir_path)
                if result:
                    parsed_results.append(result)
                    successful_parses += 1
                    print(f"✅ Successfully parsed {dir_path.name}")
                else:
                    print(f"❌ Failed to parse {dir_path.name}")
            except Exception as e:
                print(f"❌ Error parsing {dir_path.name}: {e}")
        
        print(f"\n📊 Parsing Summary:")
        print(f"Total directories: {len(result_dirs)}")
        print(f"Successfully parsed: {successful_parses}")
        print(f"Failed to parse: {len(result_dirs) - successful_parses}")
        
        self.parsed_results = parsed_results
        return parsed_results
    
    def get_results(self) -> List[ParsedResult]:
        """Get the parsed results."""
        return self.parsed_results 