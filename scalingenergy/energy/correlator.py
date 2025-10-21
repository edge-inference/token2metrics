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
Energy-Performance Correlator for Figure 6 
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .parser import EnergyCollector


class EnergyCorrelator:
    
    def __init__(self, energy_dir: str, performance_file: str):
        self.energy_dir = energy_dir
        self.performance_file = performance_file
        self.energy_data = {}
        self.performance_data = {}
    
    def load_data(self) -> Tuple[bool, bool]:
        """Load energy and performance data."""
        # Load energy data
        collector = EnergyCollector(self.energy_dir)
        energy_df = collector.process_all_files()
        
        if energy_df.empty:
            print("No energy data found")
            return False, False
        
        # Organize energy data by model and configuration
        self.energy_data = self._organize_energy_data(energy_df)
        print(f"Loaded energy data for {len(self.energy_data)} models")
        
        # Load performance data
        try:
            perf_df = pd.read_excel(self.performance_file)
            self.performance_data = self._organize_performance_data(perf_df)
            print(f"Loaded performance data for {len(self.performance_data)} models")
            return True, True
        except Exception as e:
            print(f"Error loading performance data: {e}")
            return True, False
    
    def _organize_energy_data(self, energy_df: pd.DataFrame) -> Dict[str, Dict[str, Dict]]:
        """Organize energy data by model -> config -> metrics."""
        organized = {}
        
        for _, row in energy_df.iterrows():
            model = row['model_name']
            ps_factor = row.get('ps_factor', 1)
            tokens = row.get('tokens', 512)
            
            config_key = f"PS{ps_factor}_{tokens}"
            
            if model not in organized:
                organized[model] = {}
            
            if config_key not in organized[model]:
                organized[model][config_key] = []
            
            # Include all available metrics, including new system metrics
            energy_record = {
                'avg_power_w': row['avg_power_w'],
                'peak_power_w': row['peak_power_w'],
                'total_energy_j': row['total_energy_j'],
                'duration_s': row['duration_s'],
                'ps_factor': ps_factor,
                'tokens': tokens,
                'seed': row.get('seed', 0)
            }
            
            # Add system metrics if available
            system_metrics = ['avg_ram_total_mb', 'avg_ram_used_mb', 'max_ram_total_mb', 'max_ram_used_mb', 'min_ram_total_mb', 'min_ram_used_mb',
                            'avg_gpu_usage_pct', 'max_gpu_usage_pct', 'min_gpu_usage_pct',
                            'avg_temp_tj_c', 'max_temp_tj_c', 'min_temp_tj_c', 'avg_ram_usage_pct']
            
            for metric in system_metrics:
                if metric in row and pd.notna(row[metric]):
                    energy_record[metric] = row[metric]
            
            organized[model][config_key].append(energy_record)
        
        return organized
    
    def _organize_performance_data(self, perf_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Organize performance data by model."""
        organized = {}
        
        for _, row in perf_df.iterrows():
            # Handle different column name formats
            model = row.get('model', row.get('model_name', 'unknown'))
            
            if model not in organized:
                organized[model] = []
            
            # Map column names to expected format
            ps_factor = row.get('ps', row.get('num_samples', 1))
            tokens = row.get('tokens', 128)  # Default based on your benchmark
            seed = row.get('seed', 42)
            
            # Convert time per question to latency in seconds
            decode_latency_s = row.get('decode_latency_s', 
                                      row.get('avg_time_per_question', 0))
            
            total_time_s = row.get('total_time_s', decode_latency_s)
            throughput = row.get('throughput', row.get('avg_tokens_per_second', 0))
            
            organized[model].append({
                'ps_factor': ps_factor,
                'tokens': tokens,
                'seed': seed,
                'decode_latency_s': decode_latency_s,
                'total_time_s': total_time_s,
                'throughput': throughput
            })
        
        return organized
    
    def correlate_data(self) -> pd.DataFrame:
        """Correlate energy and performance data, matching."""
        all_correlations = []
        
        print(f"\n🔍 Debug: Correlating data for {len(self.performance_data)} models")
        for model_name in self.performance_data.keys():
            perf_data = self.performance_data[model_name]
            print(f"  {model_name}: {len(perf_data)} performance records")
            for i, record in enumerate(perf_data):
                print(f"    Record {i}: PS={record['ps_factor']}, decode_time={record.get('decode_latency_s', 0):.3f}s")
            
            if model_name not in self.energy_data:
                print(f"No energy data for model {model_name}")
                continue
            
            energy_data = self.energy_data[model_name]
            print(f"  {model_name}: {len(energy_data)} energy configs: {list(energy_data.keys())}")
            
            model_correlations = self._correlate_model_data(
                model_name, 
                self.performance_data[model_name],
                self.energy_data[model_name]
            )
            all_correlations.extend(model_correlations)
        
        if all_correlations:
            df = pd.DataFrame(all_correlations)
            print(f"\n📊 Total correlations: {len(df)}")
            print(f"PS factors found: {sorted(df['ps_factor'].unique())}")
            print(f"Models correlated: {df['model_name'].unique().tolist()}")
            self._add_efficiency_metrics(df)
            return df
        
        return pd.DataFrame()
    
    def _correlate_model_data(self, model_name: str, perf_data: List[Dict], 
                            energy_data: Dict[str, Dict]) -> List[Dict]:
        """Correlate data for a single model using MEAN data"""
        # Load prefill latencies for correction
        prefill_latencies = self._load_prefill_latencies()
        prefill_latency = prefill_latencies.get(model_name, 0.0)
        
        correlations = []
        
        for perf_row in perf_data:
            ps_factor = perf_row['ps_factor']
            tokens = perf_row['tokens']
            seed = perf_row['seed']  # This will be 'MEAN'
            
            # Direct lookup since energy_data now contains averaged values
            config_key = f"PS{ps_factor}_{tokens}"
            energy_metrics = energy_data.get(config_key)
            
            if not energy_metrics:
                # Try with different token count fallback
                fallback_configs = [k for k in energy_data.keys() if k.startswith(f"PS{ps_factor}_")]
                if fallback_configs:
                    energy_metrics = energy_data[fallback_configs[0]]
                    match_method = 'ps_only'
                else:
                    continue
            else:
                match_method = 'exact_mean'
            
            # Apply prefill latency correction to get true decode time
            total_time_s = perf_row['total_time_s']
            decode_latency_s = max(0, total_time_s - prefill_latency)  # Subtract prefill, ensure non-negative
            
            correlation = {
                'model_name': model_name,
                'ps_factor': ps_factor,
                'tokens': tokens,
                'seed': seed,  # Will show 'MEAN'
                'match_method': match_method,
                
                # Performance metrics (all in seconds)
                'decode_latency_s': decode_latency_s,  # Corrected decode time
                'total_time_s': total_time_s,  # Total time including prefill
                'prefill_latency_s': prefill_latency,  # Prefill latency for reference
                'throughput': perf_row['throughput'],
                'accuracy': perf_row['accuracy'],
                'avg_ttft_s': perf_row['avg_ttft_s'],
                
                # Energy metrics
                'avg_power_w': energy_metrics['avg_power_w'],
                'peak_power_w': energy_metrics['peak_power_w'],
                'total_energy_j': energy_metrics['total_energy_j'],
                'duration_s': energy_metrics['duration_s']
            }
            
            # Add system metrics if available
            system_metrics_keys = ['avg_ram_total_mb', 'avg_ram_used_mb', 'max_ram_total_mb', 'max_ram_used_mb', 
                                 'min_ram_total_mb', 'min_ram_used_mb', 'avg_gpu_usage_pct', 'max_gpu_usage_pct', 
                                 'min_gpu_usage_pct', 'avg_temp_tj_c', 'max_temp_tj_c', 'min_temp_tj_c', 'avg_ram_usage_pct']
            
            for metric_key in system_metrics_keys:
                if metric_key in energy_metrics:
                    correlation[metric_key] = energy_metrics[metric_key]
            
            correlations.append(correlation)
        
        exact_matches = sum(1 for c in correlations if c['match_method'] == 'exact_mean')
        print(f"  {model_name}: {len(correlations)} correlations, {exact_matches} MEAN-based matches")
        print(f"    Applied prefill correction: {prefill_latency:.3f}s")
        
        return correlations
    
    def _find_exact_match(self, energy_data: Dict, ps_factor: int, tokens: int, seed: int) -> Optional[Dict]:
        """Find exact match by PS factor, tokens, and seed."""
        config_key = f"PS{ps_factor}_{tokens}"
        if config_key in energy_data:
            for energy_row in energy_data[config_key]:
                if energy_row['seed'] == seed:
                    return energy_row
        return None
    
    def _find_ps_tokens_match(self, energy_data: Dict, ps_factor: int, tokens: int) -> Optional[Dict]:
        """Find match by PS factor and tokens"""
        config_key = f"PS{ps_factor}_{tokens}"
        if config_key in energy_data:
            return self._average_energy_metrics(energy_data[config_key])
        return None
    
    def _find_ps_match(self, energy_data: Dict, ps_factor: int) -> Optional[Dict]:
        matching_configs = [config for config in energy_data.keys() if config.startswith(f"PS{ps_factor}_")]
        if matching_configs:
            all_metrics = []
            for config in matching_configs:
                all_metrics.extend(energy_data[config])
            return self._average_energy_metrics(all_metrics)
        return None
    
    def _get_model_average(self, energy_data: Dict) -> Optional[Dict]:
        """Get average energy metrics across all configurations."""
        all_metrics = []
        for config_metrics in energy_data.values():
            all_metrics.extend(config_metrics)
        
        if all_metrics:
            return self._average_energy_metrics(all_metrics)
        return None
    
    def _average_energy_metrics(self, metrics_list: List[Dict]) -> Dict:
        """average energy metrics from a list."""
        if not metrics_list:
            return {}
        
        # Basic energy metrics
        averaged = {
            'avg_power_w': round(np.mean([m['avg_power_w'] for m in metrics_list]), 3),
            'peak_power_w': round(np.max([m['peak_power_w'] for m in metrics_list]), 3),
            'total_energy_j': round(np.mean([m['total_energy_j'] for m in metrics_list]), 3),
            'duration_s': round(np.mean([m['duration_s'] for m in metrics_list]), 3)
        }
        
        # Average system metrics if available
        system_metrics_to_average = {
            'avg_ram_total_mb': lambda x: np.mean(x),
            'avg_ram_used_mb': lambda x: np.mean(x), 
            'max_ram_total_mb': lambda x: np.max(x),
            'max_ram_used_mb': lambda x: np.max(x),
            'min_ram_total_mb': lambda x: np.min(x),
            'min_ram_used_mb': lambda x: np.min(x),
            'avg_gpu_usage_pct': lambda x: np.mean(x),
            'max_gpu_usage_pct': lambda x: np.max(x),
            'min_gpu_usage_pct': lambda x: np.min(x),
            'avg_temp_tj_c': lambda x: np.mean(x),
            'max_temp_tj_c': lambda x: np.max(x),
            'min_temp_tj_c': lambda x: np.min(x),
            'avg_ram_usage_pct': lambda x: np.mean(x)
        }
        
        for metric_name, agg_func in system_metrics_to_average.items():
            values = [m.get(metric_name) for m in metrics_list if m.get(metric_name) is not None]
            if values:
                averaged[metric_name] = round(agg_func(values), 3)
        
        return averaged
    
    def _add_efficiency_metrics(self, df: pd.DataFrame) -> None:
        """Add efficiency metrics to correlation DataFrame."""
        # Calculate energy per decode using correlated power and actual decode time (in seconds)
        df['energy_per_decode_j'] = df['avg_power_w'] * df['decode_latency_s']
        df['energy_per_sample_j'] = df['energy_per_decode_j'] / df['ps_factor']
        
        df['throughput_per_watt'] = df['throughput'] / df['avg_power_w'].replace(0, np.nan)
        
        df['operations_per_joule'] = df['ps_factor'] / df['energy_per_decode_j'].replace(0, np.nan)
        
        df['latency_energy_product'] = df['decode_latency_s'] * df['energy_per_decode_j']
    
    def save_results(self, df: pd.DataFrame, output_path: str = "energy_correlation_results.xlsx") -> str:
        """correlation results to Excel sheets."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Correlations', index=False)
            
            model_summary = df.groupby('model_name').agg({
                'avg_power_w': 'mean',
                'total_energy_j': 'mean',
                'decode_latency_s': 'mean',
                'total_time_s': 'mean',
                'prefill_latency_s': 'mean',
                'throughput': 'mean',
                'throughput_per_watt': 'mean',
                'energy_per_decode_j': 'mean',
                'ps_factor': 'count'
            }).round(3)
            model_summary.columns = [
                'avg_power_w', 'avg_energy_j', 'avg_decode_latency_s', 'avg_total_time_s', 
                'avg_prefill_latency_s', 'avg_throughput', 'avg_throughput_per_watt', 
                'avg_energy_per_decode_j', 'data_points'
            ]
            model_summary.to_excel(writer, sheet_name='Model_Summary')
            
            # Summary by PS factor
            ps_summary = df.groupby('ps_factor').agg({
                'avg_power_w': 'mean',
                'decode_latency_s': 'mean',
                'throughput': 'mean',
                'energy_per_decode_j': 'mean'
            }).round(3)
            ps_summary.to_excel(writer, sheet_name='PS_Summary')
        
        print(f"Saved correlation results to {output_path}")
        return output_path

    def load_performance_data_from_means(self) -> Dict[str, List[Dict]]:
        """Load performance data from MEAN rows in individual model Excel files."""
        performance_dir = Path("results/tokens")
        organized = {}
        
        # Find all model Excel files
        for excel_file in performance_dir.glob("*.xlsx"):
            if excel_file.name == "scaling_summary.xlsx":
                continue  # Skip the summary file
                
            model_name = excel_file.stem  # Filename without extension
            organized[model_name] = []
            
            print(f"Reading MEAN data from {excel_file.name}")
            
            try:
                # Read all sheet names
                xl_file = pd.ExcelFile(excel_file)
                
                print(f"  Sheet names found: {xl_file.sheet_names}")
                
                for sheet_name in xl_file.sheet_names:
                    try:
                        # Extract PS factor from sheet name with more flexible parsing
                        # Handle formats like: "1_samples", "2_samples", "4_samples", "8_samples", "16_samples", "32_samples"
                        if '_samples' in sheet_name:
                            ps_factor = int(sheet_name.split('_')[0])
                        elif sheet_name.isdigit():
                            ps_factor = int(sheet_name)
                        else:
                            # Try to extract number from sheet name
                            import re
                            numbers = re.findall(r'\d+', sheet_name)
                            if numbers:
                                ps_factor = int(numbers[0])
                            else:
                                print(f"  Warning: Could not extract PS factor from sheet name '{sheet_name}', skipping")
                                continue
                    except ValueError:
                        print(f"  Warning: Could not parse PS factor from sheet name '{sheet_name}', skipping")
                        continue
                    
                    print(f"  Processing sheet '{sheet_name}' -> PS factor {ps_factor}")
                    
                    # Read the sheet
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Debug: print column names and first few rows
                    print(f"    Columns: {list(df.columns)}")
                    print(f"    First column values: {df.iloc[:, 0].tolist()[:10]}")
                    
                    # Find the MEAN row
                    mean_row = df[df.iloc[:, 0] == 'MEAN']
                    if mean_row.empty:
                        print(f"  Warning: No MEAN row found in {sheet_name}")
                        continue
                    
                    mean_data = mean_row.iloc[0]
                    print(f"    MEAN row data: {mean_data.to_dict()}")
                    
                    # Extract metrics from MEAN row (all in seconds, not milliseconds)
                    performance_record = {
                        'ps_factor': ps_factor,
                        'tokens': 128,  # Default assumption
                        'seed': 'MEAN',  # Indicate this is averaged data
                        'decode_latency_s': mean_data.get('avg_decode_time', 0),  # Already in seconds
                        'total_time_s': mean_data.get('avg_time_per_question', 0),  # Already in seconds  
                        'throughput': mean_data.get('avg_tokens_per_second', 0),
                        'accuracy': mean_data.get('accuracy', 0),
                        'avg_ttft_s': mean_data.get('avg_ttft', 0)  # Already in seconds
                    }
                    
                    organized[model_name].append(performance_record)
                    print(f"  Added MEAN data for PS{ps_factor}: decode_time={performance_record['decode_latency_s']:.1f}s")
                    
            except Exception as e:
                print(f"Error reading {excel_file.name}: {e}")
                continue
        
        return organized

    def _load_prefill_latencies(self) -> Dict[str, float]:
        """Load prefill latencies from prefill_latency.csv file."""
        prefill_file = Path("results/tokens/prefill_latency.csv")
        prefill_latencies = {}
        
        if not prefill_file.exists():
            print(f"⚠️  Prefill latency file not found: {prefill_file}")
            print("   Using zero prefill latency for all models")
            return prefill_latencies
        
        try:
            import pandas as pd
            df = pd.read_csv(prefill_file)
            
            for _, row in df.iterrows():
                model_name = row['model']  # e.g., 'DSR1-Qwen-14B'
                prefill_latency = float(row['prefill_latency_seconds'])
                
                # Map short names to full names
                if 'DSR1-Qwen-14B' in model_name:
                    full_name = 'DeepSeek-R1-Distill-Qwen-14B'
                elif 'DSR1-Qwen-1.5B' in model_name:
                    full_name = 'DeepSeek-R1-Distill-Qwen-1.5B'
                elif 'DSR1-LLaMA-8B' in model_name or 'DSR1-Llama-8B' in model_name:
                    full_name = 'DeepSeek-R1-Distill-Llama-8B'
                else:
                    full_name = model_name
                
                prefill_latencies[full_name] = prefill_latency
                print(f"   Loaded prefill latency for {full_name}: {prefill_latency:.3f}s")
            
        except Exception as e:
            print(f"⚠️  Error loading prefill latencies: {e}")
            print("   Using zero prefill latency for all models")
        
        return prefill_latencies

    def load_data_from_means(self) -> Tuple[bool, bool]:
        """Load energy and performance data using MEAN values instead of seed matching."""
        # Load energy data (still from raw files, but we'll average it)
        collector = EnergyCollector(self.energy_dir)
        energy_df = collector.process_all_files()
        
        if energy_df.empty:
            print("No energy data found")
            return False, False
        
        # Organize energy data and calculate means by PS factor
        self.energy_data = self._organize_energy_data_with_means(energy_df)
        print(f"Loaded energy data for {len(self.energy_data)} models")
        
        # Load performance data from MEAN rows
        self.performance_data = self.load_performance_data_from_means()
        print(f"Loaded performance MEAN data for {len(self.performance_data)} models")
        
        return True, len(self.performance_data) > 0

    def _organize_energy_data_with_means(self, energy_df: pd.DataFrame) -> Dict[str, Dict[str, Dict]]:
        """Organize energy data and calculate means by PS factor (ignoring seeds)."""
        organized = {}
        
        for _, row in energy_df.iterrows():
            model = row['model_name']
            ps_factor = row.get('ps_factor', 1)
            tokens = row.get('tokens', 512)
            
            config_key = f"PS{ps_factor}_{tokens}"
            
            if model not in organized:
                organized[model] = {}
            
            if config_key not in organized[model]:
                organized[model][config_key] = []
            
            # Include all available metrics, including new system metrics
            energy_record = {
                'avg_power_w': row['avg_power_w'],
                'peak_power_w': row['peak_power_w'],
                'total_energy_j': row['total_energy_j'],
                'duration_s': row['duration_s'],
                'ps_factor': ps_factor,
                'tokens': tokens,
                'seed': row.get('seed', 0)
            }
            
            # Add system metrics if available
            system_metrics = ['avg_ram_total_mb', 'avg_ram_used_mb', 'max_ram_total_mb', 'max_ram_used_mb', 'min_ram_total_mb', 'min_ram_used_mb',
                            'avg_gpu_usage_pct', 'max_gpu_usage_pct', 'min_gpu_usage_pct',
                            'avg_temp_tj_c', 'max_temp_tj_c', 'min_temp_tj_c', 'avg_ram_usage_pct']
            
            for metric in system_metrics:
                if metric in row and pd.notna(row[metric]):
                    energy_record[metric] = row[metric]
            
            organized[model][config_key].append(energy_record)
        
        # Now calculate means for each config
        for model in organized:
            for config_key in organized[model]:
                metrics_list = organized[model][config_key]
                # Replace list with averaged metrics
                organized[model][config_key] = self._average_energy_metrics(metrics_list)
        
        return organized


def main():
    """the correlator."""
    correlator = EnergyCorrelator("../tegra/figure6", "results/energy/excel_scaling_summary.xlsx")
    
    energy_loaded, perf_loaded = correlator.load_data()
    if not energy_loaded or not perf_loaded:
        print("Failed to load data")
        return
    
    df = correlator.correlate_data()
    if not df.empty:
        print(f"\nCorrelated {len(df)} data points")
        print(f"Models: {df['model_name'].nunique()}")
        print(f"PS factors: {sorted(df['ps_factor'].unique())}")
        
        correlator.save_results(df)
    else:
        print("No correlations found")


if __name__ == "__main__":
    main() 