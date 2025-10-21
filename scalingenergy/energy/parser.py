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
Energy CSV parser for Figure 6 - 
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import re


class EnergyParser:
    """ energy CSV parser with multiple format support."""
    
    def __init__(self):
        # Updated power columns based on actual CSV structure
        self.power_columns = ['vdd_gpu_soc_current_mw', 'power_w', 'Power_W']
        # Additional power columns to sum for total power
        self.additional_power_columns = ['vdd_cpu_cv_current_mw', 'vin_sys_5v0_current_mw']
        self.timestamp_formats = ['%m-%d-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S']
        # Updated system metrics columns to match actual CSV format
        self.system_metrics_columns = {
            'ram_total_mb': ['ram_total_mb', 'RAM_total_MB', 'memory_total_mb'],
            'ram_used_mb': ['ram_used_mb', 'RAM_used_MB', 'memory_used_mb'], 
            'gpu_usage_pct': ['gpu_usage_pct', 'GPU_usage_pct', 'gpu_utilization_pct'],
            'temp_tj_c': ['temp_tj_c', 'temperature_tj_c', 'temp_c', 'temperature_c']
        }
    
    def parse_csv(self, csv_path: str) -> Dict[str, any]:
        """Parse energy CSV and return metrics."""
        try:
            df = pd.read_csv(csv_path, comment='/')
            
            power_w = self._extract_power(df)
            if power_w is None:
                return {}
            
            elapsed_sec = self._extract_timestamps(df)
            
            energy_metrics = self._calculate_energy_metrics(power_w, elapsed_sec)
            
            # Extract additional system metrics
            system_metrics = self._extract_system_metrics(df)
            
            metadata = self._extract_filename_metadata(csv_path)
            
            return {**energy_metrics, **system_metrics, **metadata}
            
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}")
            return {}
    
    def _extract_power(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract GPU power data only (no summing of multiple sources)."""
        gpu_power = None
        
        # Look for primary GPU power column only
        for col in self.power_columns:
            if col in df.columns:
                if 'mw' in col.lower():
                    gpu_power = df[col] / 1000  # Convert mW to W
                else:
                    gpu_power = df[col]
                break
        
        # If no primary GPU power column found, try additional columns as fallback
        if gpu_power is None:
            for col in self.additional_power_columns:
                if col in df.columns:
                    if 'mw' in col.lower():
                        gpu_power = df[col] / 1000  # Convert mW to W
                    else:
                        gpu_power = df[col]
                    break
        
        return gpu_power
    
    def _extract_system_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract additional system metrics from CSV."""
        metrics = {}
        
        for metric_name, possible_columns in self.system_metrics_columns.items():
            found_column = None
            for col in possible_columns:
                if col in df.columns:
                    found_column = col
                    break
            
            if found_column is not None:
                try:
                    data = df[found_column].dropna()
                    if len(data) > 0:
                        metrics[f'avg_{metric_name}'] = round(data.mean(), 3)
                        metrics[f'max_{metric_name}'] = round(data.max(), 3)
                        metrics[f'min_{metric_name}'] = round(data.min(), 3)
                        
                        # Special handling for RAM usage percentage
                        if metric_name == 'ram_used_mb' and 'avg_ram_total_mb' in metrics:
                            if metrics['avg_ram_total_mb'] > 0:
                                ram_usage_pct = (metrics['avg_ram_used_mb'] / metrics['avg_ram_total_mb']) * 100
                                metrics['avg_ram_usage_pct'] = round(ram_usage_pct, 3)
                                
                except Exception as e:
                    print(f"Warning: Could not parse {metric_name} from column {found_column}: {e}")
                    continue
        
        return metrics
    
    def _extract_timestamps(self, df: pd.DataFrame) -> pd.Series:
        """ elapsed seconds from timestamps."""
        if 'timestamp' not in df.columns:
            return pd.Series(range(len(df)), dtype=float)
        
        times = None
        for fmt in self.timestamp_formats:
            try:
                times = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                if not times.isna().all():
                    break
            except:
                continue
        
        if times is None or times.isna().all():
            times = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if times.isna().all():
            return pd.Series(range(len(df)), dtype=float)
        
        return (times - times.iloc[0]).dt.total_seconds().fillna(0)
    
    def _calculate_energy_metrics(self, power_w: pd.Series, elapsed_sec: pd.Series) -> Dict[str, float]:
        """energy metrics."""
        time_deltas = elapsed_sec.diff().fillna(1.0)
        
        energy_increments = power_w * time_deltas
        cumulative_energy = energy_increments.cumsum()
        
        return {
            'avg_power_w': round(power_w.mean(), 3),
            'peak_power_w': round(power_w.max(), 3),
            'min_power_w': round(power_w.min(), 3),
            'total_energy_j': round(cumulative_energy.iloc[-1], 3),
            'duration_s': round(elapsed_sec.iloc[-1], 3),
            'measurement_count': len(power_w)
        }
    
    def _extract_filename_metadata(self, csv_path: str) -> Dict[str, any]:
        """Extract metadata from directory and filename patterns."""
        csv_path_obj = Path(csv_path)
        dir_name = csv_path_obj.parent.name
        filename = csv_path_obj.stem
        
        if dir_name.startswith('scale_synthetic_'):
            pattern = r'scale_synthetic_\d{8}_\d{6}_(.+?)_(\d+)samples_(\d+)tokens_seed(\d+)'
            match = re.search(pattern, dir_name)
            
            if match:
                model_name = match.group(1)
                ps_factor = int(match.group(2))
                tokens = int(match.group(3))
                seed = int(match.group(4))
                
                return {
                    'model_name': model_name,
                    'ps_factor': ps_factor,
                    'tokens': tokens,
                    'seed': seed,
                    'question_key': f"{model_name}_PS{ps_factor}_{tokens}_{seed}"
                }
        
        pattern = r'energy_(.+?)_PS(\d+)_(\d+)_(\d+)_\d{8}_\d{6}'
        match = re.search(pattern, filename)
        
        if match:
            model_name = match.group(1).replace('_', '-')
            ps_factor = int(match.group(2))
            tokens = int(match.group(3))
            seed = int(match.group(4))
            
            return {
                'model_name': model_name,
                'ps_factor': ps_factor,
                'tokens': tokens,
                'seed': seed,
                'question_key': f"{model_name}_PS{ps_factor}_{tokens}_{seed}"
            }
        
        return {'question_key': filename}


class EnergyCollector:
    """energy files by model and configuration."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.parser = EnergyParser()
    
    def collect_files(self) -> Dict[str, Dict[str, str]]:
        energy_files = {}
        
        for csv_path in self.base_dir.rglob("energy_*.csv"):
            model_name = self._extract_model_name(csv_path)
            if not model_name:
                continue
            
            metadata = self.parser._extract_filename_metadata(str(csv_path))
            question_key = metadata.get('question_key', csv_path.stem)
            
            if model_name not in energy_files:
                energy_files[model_name] = {}
            
            energy_files[model_name][question_key] = str(csv_path)
        
        return energy_files
    
    def _extract_model_name(self, csv_path: Path) -> Optional[str]:
        dir_name = csv_path.parent.name
        
        if dir_name.startswith('scale_synthetic_'):
            parts = dir_name.split('_')
            
            model_parts = []
            found_timestamp = False
            
            for i, part in enumerate(parts):
                if not found_timestamp and len(part) == 8 and part.isdigit():
                    found_timestamp = True
                    continue
                elif not found_timestamp:
                    continue
                elif len(part) == 6 and part.isdigit():
                    continue
                elif part.endswith('samples'):
                    break   
                else:
                    model_parts.append(part)
            
            if model_parts:
                return '-'.join(model_parts)
        
        known_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-14B']
        for model in known_models:
            if model in dir_name:
                return model
        
        return None
    
    def process_all_files(self) -> pd.DataFrame:
        energy_files = self.collect_files()
        all_results = []
        
        for model_name, files in energy_files.items():
            print(f"Processing {len(files)} files for {model_name}")
            
            for question_key, csv_path in files.items():
                metrics = self.parser.parse_csv(csv_path)
                if metrics:
                    metrics['model_name'] = model_name
                    metrics['file_path'] = csv_path
                    all_results.append(metrics)
        
        if all_results:
            df = pd.DataFrame(all_results)
            print(f"Processed {len(df)} energy measurements across {df['model_name'].nunique()} models")
            return df
        
        return pd.DataFrame()


def main():
    collector = EnergyCollector("../tegra/figure6")
    df = collector.process_all_files()
    
    if not df.empty:
        print("\nSample results:")
        print(df[['model_name', 'ps_factor', 'avg_power_w', 'total_energy_j']].head())
        
        # Save results
        df.to_csv("energy_parsed_results.csv", index=False)
        print(f"Saved {len(df)} results to energy_parsed_results.csv")


if __name__ == "__main__":
    main() 