"""
Individual Exponential Power Models for Each Configuration

POWER MODEL:
P_decode(t) = P_∞ - ΔP × exp(-t/τ)

Where:
- P_∞: Steady-state power (W) - final power when fully ramped up
- ΔP: Initial power deficit (W) - how far below steady-state we start  
- τ: Time constant (s) - how fast we ramp up (63% in τ seconds)
- t: Time since decode started (s)

ENERGY MODEL:
E_decode(T) = P_∞×T - ΔP×τ×(1 - exp(-T/τ))

Where:
- T: Total decode time (s)
- E_decode: Total energy consumed during decode (J)

USAGE:
Instead of: Energy = avg_power × total_time
Use: Energy = P_∞×T - ΔP×τ×(1 - exp(-T/τ))
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class IndividualExponentialModels:
    """Extract individual exponential models for each configuration"""
    
    def __init__(self, data_root: str = "../datasets/synthetic/gpu/decode/fine/", verbose: bool = False):
        self.data_root = Path(data_root)
        self.models = {}
        self.verbose = verbose
        
    def exponential_power_model(self, t: np.ndarray, P_inf: float, delta_P: float, tau: float) -> np.ndarray:
        """Exponential ramp power model"""
        return P_inf - delta_P * np.exp(-t / tau)
    
    def exponential_energy_model(self, T: float, P_inf: float, delta_P: float, tau: float) -> float:
        """Integrated energy for exponential power ramp"""
        return P_inf * T - delta_P * tau * (1 - np.exp(-T / tau))
    
    def calculate_avg_decode_energy(self, powers_w: np.ndarray, decode_time: float) -> float:
        """Calculate average decode energy: mean power × decode time"""
        return np.mean(powers_w) * decode_time
    
    def calculate_total_gpu_energy(self, powers_w: np.ndarray, sec: np.ndarray) -> float:
        """Calculate total GPU energy from power measurements and timestamps"""
        td = np.diff(sec, prepend=0)
        if len(td) > 0:
            td[0] = 1.0  # First interval
        return np.sum(powers_w * td)
    
    def extract_token_config(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract input and output tokens from filename"""
        try:
            # Parse patterns like: energy_synthetic_analysis_in128_out256_20250819_191357.csv
            parts = filename.split('_')
            input_tokens = None
            output_tokens = None
            
            for i, part in enumerate(parts):
                if part.startswith('in') and part[2:].isdigit():
                    input_tokens = int(part[2:])
                elif part.startswith('out') and part[3:].isdigit():
                    output_tokens = int(part[3:])
                    
            return input_tokens, output_tokens
        except:
            return None, None
    
    def get_performance_data(self, energy_csv: Path) -> Tuple[Optional[float], Optional[str]]:
        """Get decode time and actual model name from corresponding performance CSV file"""
        try:
            # Convert energy filename to performance filename
            # energy_synthetic_analysis_in128_out128_20250819_213948.csv
            # -> performance_synthetic_analysis_in128_out128_20250819_213948.csv
            perf_filename = energy_csv.name.replace('energy_', 'performance_')
            perf_path = energy_csv.parent / perf_filename
            
            if perf_path.exists():
                perf_df = pd.read_csv(perf_path)
                if len(perf_df) > 0:
                    decode_time = None
                    model_name = None
                    
                    if 'decode_time' in perf_df.columns:
                        decode_time_ms = perf_df['decode_time'].iloc[0]
                        decode_time = decode_time_ms / 1000.0  # Convert to seconds
                    
                    if 'model_name' in perf_df.columns:
                        model_name = perf_df['model_name'].iloc[0]
                    
                    return decode_time, model_name
        except Exception as e:
            if self.verbose:
                print(f"       Warning: Could not load performance file: {e}")
        return None, None

    def fit_exponential_to_file(self, energy_csv: Path, model_name: str) -> Optional[Dict]:
        """Fit exponential model to single energy CSV file"""
        try:
            # Load data
            df = pd.read_csv(energy_csv, comment='/')
            if len(df) < 3:
                if self.verbose:
                    print(f"       ❌ Too few data points: {len(df)} < 3")
                # Still extract basic info for these short sequences
                try:
                    # Parse timestamps and power for basic energy calculation
                    times = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S', errors='coerce')
                    if times.isna().all():
                        times = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    if not times.isna().all() and 'vdd_gpu_soc_current_mw' in df.columns:
                        sec = (times - times.iloc[0]).dt.total_seconds().values
                        powers_w = (df['vdd_gpu_soc_current_mw'] / 1000).values
                        
                        # Extract configuration
                        input_tokens, output_tokens = self.extract_token_config(energy_csv.name)
                        if input_tokens is not None and output_tokens is not None:
                            # Basic energy calculation 
                            tot_gpu_energy_calc = self.calculate_total_gpu_energy(powers_w, sec)
                            
                            # Get decode time and model name if available
                            actual_decode_time, actual_model_name = self.get_performance_data(energy_csv)
                            decode_time_for_energy = actual_decode_time if actual_decode_time is not None else sec[-1] if len(sec) > 0 else 0
                            final_model_name = actual_model_name if actual_model_name else model_name
                            
                            # Calculate avg_decode_energy even without exponential fit  
                            avg_decode_energy_calc = self.calculate_avg_decode_energy(powers_w, decode_time_for_energy)
                            
                            if self.verbose:
                                print(f"       ✓ Basic energy only: {tot_gpu_energy_calc:.1f}J")
                            
                            return {
                                'model_name': final_model_name,  # Will be removed later to avoid redundancy
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'P_inf': None,
                                'delta_P': None, 
                                'tau': None,
                                'r2': None,
                                'rmse': None,
                                'initial_power': None,
                                'final_power': None,
                                'decode_time': round(decode_time_for_energy, 2),
                                'monitoring_time': round(sec[-1], 2) if len(sec) > 0 else None,
                                'tot_gpu_energy': round(tot_gpu_energy_calc, 2),
                                'decode_energy': None,  # Can't calculate without exponential fit
                                'avg_decode_energy': round(avg_decode_energy_calc, 2),
                                'e_total_less_e_decode': None,  # Can't calculate without decode_energy
                                'e_decode_vs_e_avg_diff': None  # Can't calculate without decode_energy
                            }
                except Exception as e:
                    if self.verbose:
                        print(f"       ❌ Could not extract basic info: {e}")
                
                return None
                
            # Parse timestamps and power
            times = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S', errors='coerce')
            if times.isna().all():
                # Try fallback parsing
                times = pd.to_datetime(df['timestamp'], errors='coerce')
                if times.isna().all():
                    if self.verbose:
                        print(f"       ❌ Cannot parse timestamps")
                    return None
                
            sec = (times - times.iloc[0]).dt.total_seconds().values
            
            # Check if we have GPU SOC power data
            if 'vdd_gpu_soc_current_mw' not in df.columns:
                if self.verbose:
                    print(f"       ❌ Missing vdd_gpu_soc_current_mw column")
                return None
                
            powers_w = (df['vdd_gpu_soc_current_mw'] / 1000).values
            
            # Extract configuration from filename
            input_tokens, output_tokens = self.extract_token_config(energy_csv.name)
            if input_tokens is None or output_tokens is None:
                if self.verbose:
                    print(f"       ❌ Cannot extract token config from filename")
                return None
            
            # Get actual decode time and model name from performance file
            actual_decode_time, actual_model_name = self.get_performance_data(energy_csv)
            monitoring_duration = sec[-1]  # Total energy monitoring window
            
            # Use actual decode time if available, otherwise fall back to monitoring duration
            decode_time_for_energy = actual_decode_time if actual_decode_time is not None else monitoring_duration
            
            # Use actual model name if available, otherwise use fallback
            final_model_name = actual_model_name if actual_model_name else model_name
            
            # Check if sequence is too short for meaningful exponential fitting
            if monitoring_duration < 1.0:
                if self.verbose:
                    print(f"       ❌ Duration too short: {monitoring_duration:.2f}s < 1.0s")
                return None
                
            if actual_decode_time is not None and self.verbose:
                print(f"       ✓ Using decode time: {actual_decode_time:.2f}s (monitoring: {monitoring_duration:.2f}s)")
                
            # Check power variation - if too small, exponential fit may not be meaningful
            power_range = powers_w.max() - powers_w.min()
            power_mean = powers_w.mean()
            if power_range < power_mean * 0.05:  # Less than 5% variation
                if self.verbose:
                    print(f"       ❌ Power variation too small: {power_range:.2f}W ({power_range/power_mean*100:.1f}% of mean)")
                return None
            
            # Initial parameter estimates with better logic
            P_initial = powers_w[0]
            P_final = powers_w[-1] 
            P_mean = powers_w.mean()
            
            # More robust initial estimates
            # Model: P(t) = P_inf - delta_P * exp(-t/tau)
            # At t=0: P(0) = P_inf - delta_P = P_initial
            # At t=∞: P(∞) = P_inf ≈ P_final
            
            if P_final > P_initial:  # Power is ramping up
                # P_inf should be close to final power, but allow some headroom
                P_inf_init = max(P_final * 1.05, P_mean * 1.2)
                # delta_P = P_inf - P_initial, must be positive for ramp-up
                delta_P_init = P_inf_init - P_initial
                # Ensure delta_P is reasonable
                if delta_P_init < 0.5:  # Too small variation
                    delta_P_init = max(power_range * 0.8, 1.0)
                    P_inf_init = P_initial + delta_P_init
            else:  # Power is steady or declining
                P_inf_init = max(P_mean, P_initial * 0.9)
                delta_P_init = max(power_range * 0.3, 0.5)
            
            tau_init = max(monitoring_duration / 4, 0.3)  # Quarter of monitoring time or minimum
            
            # Fit exponential model with realistic bounds for 65W max system
            tau_upper = max(monitoring_duration, 30.0)  # Allow tau to be at least as long as the monitoring
            bounds = (
                [1.0, 0.01, 0.1],  # Lower bounds: min 1W, 0.01W deficit, 0.1s tau
                [65.0, 60.0, tau_upper]  # Upper bounds: 65W max power, 60W max deficit
            )
            
            # Ensure initial guess is within bounds
            P_inf_init = np.clip(P_inf_init, bounds[0][0], bounds[1][0])
            delta_P_init = np.clip(delta_P_init, bounds[0][1], bounds[1][1]) 
            tau_init = np.clip(tau_init, bounds[0][2], bounds[1][2])
            
            # Validate that our model makes physical sense
            expected_initial = P_inf_init - delta_P_init
            if abs(expected_initial - P_initial) > 5.0:  # More than 5W difference
                # Adjust parameters to better match initial conditions
                delta_P_init = P_inf_init - P_initial
                delta_P_init = np.clip(delta_P_init, bounds[0][1], bounds[1][1])
            
            popt, pcov = curve_fit(
                self.exponential_power_model,
                sec,
                powers_w,
                p0=[P_inf_init, delta_P_init, tau_init],
                bounds=bounds,
                maxfev=5000,
                method='trf'  # Trust Region Reflective - more robust
            )
            
            P_inf_fit, delta_P_fit, tau_fit = popt
            
            # Calculate fit quality
            p_pred = self.exponential_power_model(sec, P_inf_fit, delta_P_fit, tau_fit)
            ss_res = np.sum((powers_w - p_pred) ** 2)
            ss_tot = np.sum((powers_w - np.mean(powers_w)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((powers_w - p_pred) ** 2))
            
            # Energy comparison using actual decode time for energy calculation
            monitoring_time = sec[-1]  # Total monitoring window
            tot_gpu_energy_calc = self.calculate_total_gpu_energy(powers_w, sec)  # Total measured energy
            
            # Use actual decode time for exponential energy model
            decode_energy_calc = self.exponential_energy_model(decode_time_for_energy, P_inf_fit, delta_P_fit, tau_fit)
            avg_decode_energy_calc = self.calculate_avg_decode_energy(powers_w, decode_time_for_energy)
            
            return {
                'model_name': final_model_name,  # Will be removed later to avoid redundancy
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'P_inf': round(P_inf_fit, 3),
                'delta_P': round(delta_P_fit, 3),
                'tau': round(tau_fit, 3),
                'r2': round(r2, 4),
                'rmse': round(rmse, 3),
                'initial_power': round(P_inf_fit - delta_P_fit, 3),
                'final_power': round(P_inf_fit - delta_P_fit * np.exp(-decode_time_for_energy/tau_fit), 3),
                'decode_time': round(decode_time_for_energy, 2),
                'monitoring_time': round(monitoring_time, 2),
                'tot_gpu_energy': round(tot_gpu_energy_calc, 2),
                'decode_energy': round(decode_energy_calc, 2),
                'avg_decode_energy': round(avg_decode_energy_calc, 2),
                'e_total_less_e_decode': round(tot_gpu_energy_calc - decode_energy_calc, 2),
                'e_decode_vs_e_avg_diff': round((decode_energy_calc - avg_decode_energy_calc) / avg_decode_energy_calc * 100, 2) if avg_decode_energy_calc != 0 else None
            }
            
        except Exception as e:
            if self.verbose:
                print(f"       ❌ Exponential fit failed: {str(e)[:100]}")
                
                # Try to extract basic info for debugging
                try:
                    if 'df' in locals() and 'sec' in locals() and 'powers_w' in locals():
                        print(f"       Data: {len(df)} points, {sec[-1]:.1f}s duration")
                        print(f"       Power: {powers_w.min():.1f}W to {powers_w.max():.1f}W")
                        
                        # Check if this could be a simple linear or constant case
                        power_trend = powers_w[-1] - powers_w[0]
                        print(f"       Power trend: {power_trend:+.1f}W")
                        
                        # For very short sequences with small output tokens, this is expected
                        if 'output_tokens' in locals() and output_tokens and output_tokens <= 16:
                            print(f"       Note: Very few output tokens ({output_tokens}), exponential fit often fails")
                except:
                    pass
            
            return None
    
    def process_all_models(self) -> Dict:
        """Process all model directories and extract exponential parameters"""
        results = {}
        
        # Find all model directories (skip processed_results)
        for model_dir in self.data_root.iterdir():
            if not model_dir.is_dir() or model_dir.name == 'processed_results':
                continue
                
            fallback_model_name = f"Model_{model_dir.name}"  # e.g., "Model_1.5B", "Model_14B"
            print(f"\n=== Processing {model_dir.name} ===")
            
            model_results = []
            failed_configs = []
            actual_model_name = None  # Will be determined from first performance file
            
            # Find all energy CSV files in this model directory
            energy_files = list(model_dir.glob("energy_synthetic_analysis_*.csv"))
            print(f"  Found {len(energy_files)} energy CSV files")
            
            for i, energy_file in enumerate(energy_files):
                if self.verbose:
                    print(f"  Processing: {energy_file.name}")
                elif i > 0 and i % 20 == 0:  # Progress indicator every 20 files
                    print(f"  Progress: {i}/{len(energy_files)} files processed...")
                
                result = self.fit_exponential_to_file(energy_file, fallback_model_name)
                if result:
                    # Use the first actual model name we find
                    if actual_model_name is None and 'model_name' in result:
                        actual_model_name = result['model_name']
                        if actual_model_name != fallback_model_name:
                            print(f"  ✓ Using model name: {actual_model_name}")
                    
                    # Remove redundant model_name field from individual results
                    result.pop('model_name', None)
                    model_results.append(result)
                    
                    # Format output differently for fitted vs fallback cases
                    if self.verbose:
                        if result['P_inf'] is not None:
                            print(f"    ✅ P_∞={result['P_inf']}W, ΔP={result['delta_P']}W, τ={result['tau']}s, R²={result['r2']}")
                        else:
                            print(f"    ✅ Basic energy: {result['tot_gpu_energy']}J (no exponential fit)")
                else:
                    # Track failed configurations for analysis
                    input_tokens, output_tokens = self.extract_token_config(energy_file.name)
                    if input_tokens and output_tokens:
                        failed_configs.append(f"{input_tokens}→{output_tokens}")
                    if self.verbose:
                        print(f"    ❌ Failed to fit")
            
            if model_results:
                # Use actual model name as key, fallback to directory-based name
                final_model_name = actual_model_name if actual_model_name else fallback_model_name
                results[final_model_name] = model_results
                
                fitted_count = len([r for r in model_results if r['P_inf'] is not None])
                fallback_count = len([r for r in model_results if r['P_inf'] is None])
                
                print(f"  ✅ Successfully processed {len(model_results)}/{len(energy_files)} configurations")
                if fitted_count > 0 and fallback_count > 0:
                    print(f"    - Exponential fits: {fitted_count}")
                    print(f"    - Basic energy only: {fallback_count}")
                elif fitted_count > 0:
                    print(f"    - All exponential fits: {fitted_count}")
                elif fallback_count > 0:
                    print(f"    - All basic energy only: {fallback_count}")
                    
                if failed_configs:
                    print(f"  ❌ Failed configurations: {', '.join(failed_configs[:10])}")
                    if len(failed_configs) > 10:
                        print(f"      ... and {len(failed_configs)-10} more")
            else:
                print(f"  ❌ No successful processing for {fallback_model_name}")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "individual_exponential_models.json"):
        """Save results to JSON file with metadata header"""
        output_path = Path("output") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create JSON structure with metadata
        from datetime import datetime
        json_output = {
            "_metadata": {
                "description": "Individual Exponential Power Models for Each Configuration",
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "power_model": {
                    "equation": "P_decode(t) = P_inf - delta_P * exp(-t/tau)",
                    "description": "Exponential power ramp model for GPU decode process",
                    "parameters": {
                        "P_inf": "Steady-state power (W) - final power when fully ramped up",
                        "delta_P": "Initial power deficit (W) - how far below steady-state we start",
                        "tau": "Time constant (s) - how fast we ramp up (63% in tau seconds)",
                        "t": "Time since decode started (s)"
                    },
                    "physical_meaning": "At t=0: P(0) = P_inf - delta_P (initial power), At t=infinity: P(infinity) = P_inf (steady-state power)"
                },
                "energy_model": {
                    "equation": "E_decode(T) = P_inf*T - delta_P*tau*(1 - exp(-T/tau))",
                    "description": "Integrated energy from exponential power model",
                    "parameters": {
                        "T": "Total decode time (s)",
                        "E_decode": "Total energy consumed during decode (J)"
                    },
                    "usage": "Energy = P_inf*T - delta_P*tau*(1 - exp(-T/tau)) where T = actual decode time from performance file"
                },
                "structure": "Top-level keys are actual model names from performance files (e.g., DeepSeek-R1-Distill-Qwen-14B)",
                "data_fields": {
                    "input_tokens": "Number of input tokens in context",
                    "output_tokens": "Number of tokens to generate",
                    "P_inf": "Fitted steady-state power (W)",
                    "delta_P": "Fitted initial power deficit (W)",
                    "tau": "Fitted time constant (s)",
                    "r2": "R-squared goodness of fit (higher is better)",
                    "rmse": "Root mean square error (W)",
                    "initial_power": "Calculated initial power: P_inf - delta_P (W)",
                    "final_power": "Calculated final power at end of sequence (W)",
                    "decode_time": "Actual decode time from performance file (s) - used for energy calculation",
                    "monitoring_time": "Total energy monitoring window duration (s) - used for power fitting",
                    "tot_gpu_energy": "Measured total GPU energy from data (J)",
                    "decode_energy": "Predicted decode energy from exponential model (J)",
                    "avg_decode_energy": "Average power * decode time energy (J)",
                    "e_total_less_e_decode": "Difference between total measured and decode energy: e_total - e_decode (J)",
                    "e_decode_vs_e_avg_diff": "Difference between decode and average decode energy: (e_decode - e_avg)/e_avg * 100 (%)"
                },
                "limitations": [
                    "Exponential model works best for sequences >64 output tokens",
                    "Very short sequences (<3 data points) cannot be fitted",
                    "Step-function power changes don't fit exponential assumptions",
                    "Hardware power limits: P_inf <= 65W, delta_P <= 60W for this system",
                    "Power fitting uses full monitoring window, but energy calculation uses actual decode time"
                ]
            }
        }
        
        json_output.update(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate summary report of all models"""
        report = []
        report.append("=== INDIVIDUAL EXPONENTIAL MODELS SUMMARY ===\n")
        
        total_configs = 0
        all_r2_values = []
        failure_patterns = {}
        
        for model_name, configs in results.items():
            report.append(f"Model: {model_name}")
            report.append(f"Configurations: {len(configs)}")
            
            if configs:
                # Filter out None r2 values (from fallback cases)
                r2_values = [c['r2'] for c in configs if c['r2'] is not None]
                fitted_configs = [c for c in configs if c['r2'] is not None]
                fallback_configs = [c for c in configs if c['r2'] is None]
                
                if r2_values:
                    avg_r2 = np.mean(r2_values)
                    all_r2_values.extend(r2_values)
                    
                    report.append(f"Fitted configurations: {len(fitted_configs)}")
                    if fallback_configs:
                        report.append(f"Basic-only configurations (< 3 data points): {len(fallback_configs)}")
                    report.append(f"Average R²: {avg_r2:.3f}")
                    report.append(f"R² Range: {min(r2_values):.3f} - {max(r2_values):.3f}")
                    
                    # Show best and worst fits (only from fitted configs)
                    best_config = max(fitted_configs, key=lambda x: x['r2'])
                    worst_config = min(fitted_configs, key=lambda x: x['r2'])
                    
                    report.append(f"Best fit: {best_config['input_tokens']}→{best_config['output_tokens']} (R²={best_config['r2']:.3f})")
                    report.append(f"Worst fit: {worst_config['input_tokens']}→{worst_config['output_tokens']} (R²={worst_config['r2']:.3f})")
                    
                    # Analyze patterns in poor fits
                    poor_fits = [c for c in fitted_configs if c['r2'] < 0.7]
                    if poor_fits:
                        small_output = [c for c in poor_fits if c['output_tokens'] <= 32]
                        if small_output:
                            report.append(f"Poor fits with small output tokens: {len(small_output)}/{len(poor_fits)}")
                else:
                    report.append(f"No exponential fits possible (all configurations < 3 data points)")
                    report.append(f"Basic-only configurations: {len(fallback_configs)}")
                
                total_configs += len(configs)
            
            report.append("")
        
        # Overall summary
        if all_r2_values:
            overall_avg_r2 = np.mean(all_r2_values)
            good_fits = sum(1 for r2 in all_r2_values if r2 > 0.8)
            decent_fits = sum(1 for r2 in all_r2_values if r2 > 0.6)
            
            report.append(f"=== OVERALL SUMMARY ===")
            report.append(f"Total configurations: {total_configs}")
            report.append(f"Overall average R²: {overall_avg_r2:.3f}")
            report.append(f"Excellent fits (R²>0.8): {good_fits}/{len(all_r2_values)} ({good_fits/len(all_r2_values)*100:.1f}%)")
            report.append(f"Decent fits (R²>0.6): {decent_fits}/{len(all_r2_values)} ({decent_fits/len(all_r2_values)*100:.1f}%)")
            
            # Recommendations based on success rate
            if good_fits / len(all_r2_values) > 0.7:
                report.append(f"\n✅ RECOMMENDATION: Exponential model works well for most configurations")
            elif decent_fits / len(all_r2_values) > 0.6:
                report.append(f"\n⚠️  RECOMMENDATION: Exponential model has mixed success - consider filtering")
            else:
                report.append(f"\n❌ RECOMMENDATION: Exponential model struggles - may need alternative approaches")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract individual exponential models for each configuration')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed progress for each file')
    parser.add_argument('--data-root', default="../datasets/synthetic/gpu/decode/fine/",
                       help='Root directory containing energy data')
    
    args = parser.parse_args()
    
    print("🔥 INDIVIDUAL EXPONENTIAL MODELS EXTRACTION")
    print("=" * 50)
    
    # Initialize extractor
    extractor = IndividualExponentialModels(data_root=args.data_root, verbose=args.verbose)
    
    # Process all models
    results = extractor.process_all_models()
    
    if not results:
        print("❌ No models found or processed successfully")
        return
    
    # Save results
    extractor.save_results(results)
    
    # Generate and display summary
    summary = extractor.generate_summary_report(results)
    print("\n" + summary)
    
    # Save summary to file
    summary_path = Path("output/individual_models_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\n📊 Summary report saved to: {summary_path}")
    print("\n🎯 Next steps:")
    print("1. Review output/individual_exponential_models.json")
    print("2. Check R² values for model quality")
    print("3. Use exponential energy formula instead of P×T")


if __name__ == "__main__":
    main()
