"""
Energy analysis module to load energy monitoring CSV files, compute metrics, and visualize data.
"""
import os
from typing import Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import PathManager, save_dataframe, save_figure

sns.set_palette('colorblind')


class EnergyProcessor:
    """
    Processes multiple energy CSVs, computing summary metrics per model.
    """

    def __init__(self, dataset_paths: Dict[str, str] = None) -> None:  # type: ignore
        """
        Args:
            dataset_paths: Mapping from model name to energy CSV filepath.
        """
        self.dataset_paths = dataset_paths or {}
        PathManager.ensure_dirs()

    def process_energy_csv(self, csv_path: str, model_name: str = "unknown", prefer_gpu: bool = True) -> Dict[str, float]:  # type: ignore
        """
        Process a single energy CSV file and compute metrics.
        
        Args:
            csv_path: Path to energy CSV file
            model_name: Name of the model for logging
            prefer_gpu: Prioritize GPU power metrics over CPU (default: False for decode)
            
        Returns:
            Dictionary with energy metrics
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"Energy file not found for {model_name}: {csv_path}")
                return {}
                
            df = pd.read_csv(csv_path, comment='/')
                
            power_columns = [
                ('vdd_gpu_soc_current_mw', 1000) if prefer_gpu else ('vdd_cpu_cv_current_mw', 1000),
                ('vdd_cpu_cv_current_mw', 1000) if prefer_gpu else ('vdd_gpu_soc_current_mw', 1000),
                ('power_w', 1),
                ('vin_sys_5v0_current_mw', 1000)
            ]
            
            power_found = False
            power_source = 'Unknown'
            for col_name, divisor in power_columns:
                if col_name in df.columns:
                    df['Power_W'] = df[col_name] / divisor
                    if 'gpu' in col_name.lower():
                        power_source = 'GPU_SOC'
                    elif 'cpu' in col_name.lower():
                        power_source = 'CPU'
                    elif 'power_w' == col_name:
                        power_source = 'Direct'
                    elif 'sys' in col_name.lower():
                        power_source = 'SYS5V'
                    power_found = True
                    break
            
            if not power_found:
                print(f"Warning: No power column found in {csv_path}")
                print(f"Available columns: {df.columns.tolist()}")
                return {}

            # Always capture System 5V rail (if present) for additional metrics
            if 'vin_sys_5v0_current_mw' in df.columns:
                df['Power_SYS5V_W'] = df['vin_sys_5v0_current_mw'] / 1000
            
            # Compute elapsed seconds
            if 'timestamp' in df.columns:
                times = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S', errors='coerce')
                if times.isna().all():
                    times = pd.to_datetime(df['timestamp'], errors='coerce')
                sec = (times - times.iloc[0]).dt.total_seconds().fillna(0)
            else:
                sec = pd.Series(range(len(df)), dtype=float)
            
            td = sec.diff().fillna(1.0)
            df['Energy_Increment'] = df['Power_W'] * td
            df['Cumulative_Energy'] = df['Energy_Increment'].cumsum()
            
            output_tokens = self._extract_output_tokens(csv_path)
            
            avg_energy_per_token = None
            if output_tokens and output_tokens > 0:
                total_energy_j = df['Cumulative_Energy'].iloc[-1]
                avg_energy_per_token = round(total_energy_j / output_tokens, 6)
            
            metrics = {  # type: ignore
                'avg_power_w': round(df['Power_W'].mean(), 3),
                'peak_power_w': round(df['Power_W'].max(), 3),
                'min_power_w': round(df['Power_W'].min(), 3),
                'total_energy_j': round(df['Cumulative_Energy'].iloc[-1], 3),
                'duration_s': round(sec.iloc[-1], 2),
                'avg_energy_per_second': round(df['Energy_Increment'].mean(), 3),
                'power_source': power_source
            }

            if 'Power_SYS5V_W' in df.columns:
                df['Energy_Increment_SYS5V'] = df['Power_SYS5V_W'] * td
                metrics.update({
                    'avg_power_w_sys5v': round(df['Power_SYS5V_W'].mean(), 3),
                    'peak_power_w_sys5v': round(df['Power_SYS5V_W'].max(), 3),
                    'min_power_w_sys5v': round(df['Power_SYS5V_W'].min(), 3),
                    'total_energy_j_sys5v': round(df['Energy_Increment_SYS5V'].cumsum().iloc[-1], 3),
                    'avg_energy_per_second_sys5v': round(df['Energy_Increment_SYS5V'].mean(), 3)
                })
            
            if output_tokens is not None:
                metrics['output_tokens'] = output_tokens  # type: ignore
            if avg_energy_per_token is not None:
                metrics['avg_energy_per_token'] = avg_energy_per_token  # type: ignore
            
            return metrics
            
        except Exception as e:
            print(f"Error processing energy data for {model_name}: {e}")
            return {}

    def _extract_output_tokens(self, csv_path: str) -> Optional[int]:
        """
        Extract output token count from energy CSV filename.
        
        Supports multiple patterns:
        - Tegra: energy_SUBJECT_qNUMBER_inTOKENS_outTOKENS_TIMESTAMP.csv
        - CPU decode: energy_decode_synthetic_SUBJECT_qNUMBER_inTOKENS_NUMBERtokens_TIMESTAMP.csv

        """
        import re
        from pathlib import Path
        
        filename = Path(csv_path).stem
        
        # CPU decode pattern: _NUMBERtokens_
        cpu_pattern = r'_(\d+)tokens_'
        cpu_match = re.search(cpu_pattern, filename)
        
        if cpu_match:
            try:
                return int(cpu_match.group(1))
            except ValueError:
                pass
        
        # Tegra pattern: _outNUMBER_
        tegra_pattern = r'_out(\d+)_'
        tegra_match = re.search(tegra_pattern, filename)
        
        if tegra_match:
            try:
                return int(tegra_match.group(1))
            except ValueError:
                pass
        
        return None

    def summarize_all(self, prefer_gpu: bool = True) -> pd.DataFrame:  # type: ignore
        """
        Compute energy summary for each model in dataset_paths.

        Args:
            prefer_gpu: Prioritize GPU power metrics over CPU (default: True for decode)

        Returns:
            DataFrame with energy metrics per model
        """
        rows = []
        processed_count = 0
        
        for model, csv_path in self.dataset_paths.items():
            metrics = self.process_energy_csv(csv_path, model, prefer_gpu=prefer_gpu)
            if metrics:
                metrics['Model'] = model  # type: ignore
                rows.append(metrics)
                processed_count += 1
        
        print(f"Processed {processed_count}/{len(self.dataset_paths)} energy files")
        
        if rows:
            df = pd.DataFrame(rows)
            cols = ['Model'] + [col for col in df.columns if col != 'Model']
            return df[cols]  # type: ignore
        
        return pd.DataFrame()

    def create_energy_visualizations(self, summary_df: pd.DataFrame) -> None:
        """
        Create energy model-comparison charts only.
        """
        if summary_df.empty:
            print("No energy data available")
            return
        self._create_comparison_visualizations(summary_df)

    def _create_comparison_visualizations(self, df: pd.DataFrame) -> None:
        """Create visualizations for model comparison energy analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(4, 4))
        
        # Plot 1: Power Comparison
        sns.barplot(data=df, x='Model', y='avg_power_w', ax=axes[0,0])
        axes[0,0].set_title('Average Power Consumption by Model')
        axes[0,0].set_ylabel('Average Power (W)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Total Energy Comparison
        sns.barplot(data=df, x='Model', y='total_energy_j', ax=axes[0,1])
        axes[0,1].set_title('Total Energy Consumption by Model')
        axes[0,1].set_ylabel('Total Energy (J)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Duration Comparison
        sns.barplot(data=df, x='Model', y='duration_s', ax=axes[1,0])
        axes[1,0].set_title('Execution Duration by Model')
        axes[1,0].set_ylabel('Duration (s)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Power Range (min to max)
        x_pos = range(len(df))
        axes[1,1].bar(x_pos, df['peak_power_w'], alpha=0.7, label='Peak Power')
        axes[1,1].bar(x_pos, df['avg_power_w'], alpha=0.9, label='Average Power')
        axes[1,1].bar(x_pos, df['min_power_w'], alpha=0.5, label='Minimum Power')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(df['Model'], rotation=45)
        axes[1,1].set_title('Power Range by Model')
        axes[1,1].set_ylabel('Power (W)')
        axes[1,1].legend(loc='lower right')
        
        plt.tight_layout()
        save_figure(fig, 'energy_model_comparison.png', chart_type="profiling")
        plt.close(fig)

        # If System 5V rail metrics are available, generate additional comparison plots
        if {'avg_power_w_sys5v', 'total_energy_j_sys5v'}.issubset(df.columns):
            fig2, axes2 = plt.subplots(2, 2, figsize=(4, 4))

            sns.barplot(data=df, x='Model', y='avg_power_w_sys5v', ax=axes2[0, 0])
            axes2[0, 0].set_title('Average SYS5V Power by Model')
            axes2[0, 0].set_ylabel('Average Power (W)')
            axes2[0, 0].tick_params(axis='x', rotation=45)

            sns.barplot(data=df, x='Model', y='total_energy_j_sys5v', ax=axes2[0, 1])
            axes2[0, 1].set_title('Total SYS5V Energy by Model')
            axes2[0, 1].set_ylabel('Total Energy (J)')
            axes2[0, 1].tick_params(axis='x', rotation=45)

            sns.barplot(data=df, x='Model', y='duration_s', ax=axes2[1, 0])
            axes2[1, 0].set_title('Execution Duration by Model')
            axes2[1, 0].set_ylabel('Duration (s)')
            axes2[1, 0].tick_params(axis='x', rotation=45)

            x_pos = range(len(df))
            axes2[1, 1].bar(x_pos, df['peak_power_w_sys5v'], alpha=0.7, label='Peak Power')
            axes2[1, 1].bar(x_pos, df['avg_power_w_sys5v'], alpha=0.9, label='Average Power')
            axes2[1, 1].bar(x_pos, df['min_power_w_sys5v'], alpha=0.5, label='Minimum Power')
            axes2[1, 1].set_xticks(x_pos)
            axes2[1, 1].set_xticklabels(df['Model'], rotation=45)
            axes2[1, 1].set_title('SYS5V Power Range by Model')
            axes2[1, 1].set_ylabel('Power (W)')
            axes2[1, 1].legend(loc='lower right')

            plt.tight_layout()
            save_figure(fig2, 'energy_model_comparison_sys5v.png', chart_type="profiling")
            plt.close(fig2)


def main(dataset_paths: Dict[str, str], prefer_gpu: bool = True) -> None:
    """
    Energy model comparison only.
    
    Args:
        dataset_paths: Mapping from model name to energy CSV path
        prefer_gpu: Prioritize GPU power metrics over CPU (default: False for decode)
    """
    PathManager.ensure_dirs()
    processor = EnergyProcessor(dataset_paths)

    print("=== Energy Model Comparison ===")
    summary_df = processor.summarize_all(prefer_gpu=prefer_gpu)
    save_dataframe(summary_df, 'energy_comparison_summary.xlsx')
    print(f"Saved to {PathManager.get_data_path('energy_comparison_summary.xlsx')}")
    print(summary_df.to_string(index=False))

    processor.create_energy_visualizations(summary_df)




