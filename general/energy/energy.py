"""
Energy analysis module to load energy monitoring CSV files, compute metrics, and visualize data.
"""
import os
from typing import Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from energy.utils import PathManager, save_dataframe, save_figure

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

    def process_energy_csv(self, csv_path: str, model_name: str = "unknown") -> Dict[str, float]:  # type: ignore
        """
        Process an energy CSV file and compute energy metrics.
        
        Args:
            csv_path: Path to energy CSV file
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with energy metrics
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"Energy file not found for {model_name}: {csv_path}")
                return {}
                
            df = pd.read_csv(csv_path)
            
            # Check if required columns exist
            power_columns = ['vdd_cpu_cv_current_mw', 'vdd_gpu_soc_current_mw', 'vin_sys_5v0_current_mw']
            missing_columns = [col for col in power_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing power columns in {csv_path}: {missing_columns}")
                return {}
            
            if df.empty:
                print(f"Empty energy file for {model_name}: {csv_path}")
                return {}
            
            # Calculate total power (sum of all power sources)
            df['Total_Power_mW'] = (df['vdd_cpu_cv_current_mw'] + 
                                   df['vdd_gpu_soc_current_mw'] + 
                                   df['vin_sys_5v0_current_mw'])
            
            # Convert to watts
            df['Power_W'] = df['Total_Power_mW'] / 1000.0
            
            # Calculate time differences (assuming 1 second intervals)
            df['Time_Diff_s'] = 1.0  # Default 1 second interval
            
            # Calculate energy increments (Power * Time)
            df['Energy_Increment'] = df['Power_W'] * df['Time_Diff_s']
            
            # Calculate cumulative energy
            df['Cumulative_Energy'] = df['Energy_Increment'].cumsum()
            
            # Create time column for duration calculation
            df['Time_s'] = range(len(df))
            
            # Calculate metrics
            metrics = {  # type: ignore
                'avg_power_w': round(df['Power_W'].mean(), 3),
                'peak_power_w': round(df['Power_W'].max(), 3),
                'min_power_w': round(df['Power_W'].min(), 3),
                'total_energy_kj': round(df['Cumulative_Energy'].iloc[-1] / 1000, 3),
                'duration_s': round(len(df), 2),
            }
            
            # Add GPU utilization metrics if available
            if 'gpu_usage_pct' in df.columns:
                metrics['avg_gpu_usage_pct'] = round(df['gpu_usage_pct'].mean(), 2)
                metrics['peak_gpu_usage_pct'] = round(df['gpu_usage_pct'].max(), 2)
                metrics['min_gpu_usage_pct'] = round(df['gpu_usage_pct'].min(), 2)
            
            # Add memory usage metrics if available
            if 'ram_total_mb' in df.columns and 'ram_used_mb' in df.columns:
                metrics['avg_ram_used_mb'] = round(df['ram_used_mb'].mean(), 1)
                metrics['peak_ram_used_mb'] = round(df['ram_used_mb'].max(), 1)
                metrics['avg_ram_total_mb'] = round(df['ram_total_mb'].mean(), 1)
                # Calculate memory utilization percentage
                ram_utilization_pct = (df['ram_used_mb'] / df['ram_total_mb']) * 100
                metrics['avg_ram_utilization_pct'] = round(ram_utilization_pct.mean(), 2)
                metrics['peak_ram_utilization_pct'] = round(ram_utilization_pct.max(), 2)
            
            # Add GPU temperature metrics if available
            if 'temp_gpu_c' in df.columns:
                metrics['avg_temp_gpu_c'] = round(df['temp_gpu_c'].mean(), 1)
                metrics['peak_temp_gpu_c'] = round(df['temp_gpu_c'].max(), 1)
                metrics['min_temp_gpu_c'] = round(df['temp_gpu_c'].min(), 1)
            
            return metrics
            
        except Exception as e:
            print(f"Error processing energy file {csv_path}: {e}")
            return {}

    def _extract_input_tokens(self, csv_path: str) -> Optional[int]:
        """
        Extract input token count from energy CSV filename.
        
        Expected pattern: energy_SUBJECT_qNUMBER_inTOKENS_outTOKENS_TIMESTAMP.csv
        Example: energy_high_school_us_history_q64_in510_out1024_20250626_153610.csv
        
        Args:
            csv_path: Path to energy CSV file
            
        Returns:
            Input token count or None if not found
        """
        import re
        from pathlib import Path
        
        filename = Path(csv_path).stem
        
        # Pattern to match input tokens: _inNUMBER_
        pattern = r'_in(\d+)_'
        match = re.search(pattern, filename)
        
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None

    def summarize_all(self) -> pd.DataFrame:  # type: ignore
        """
        Compute energy summary for each model in dataset_paths.

        Returns:
            DataFrame with energy metrics per model
        """
        rows = []
        processed_count = 0
        
        for model, csv_path in self.dataset_paths.items():
            metrics = self.process_energy_csv(csv_path, model)
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Power Comparison
        sns.barplot(data=df, x='Model', y='avg_power_w', ax=axes[0,0])
        axes[0,0].set_title('Average Power Consumption by Model')
        axes[0,0].set_ylabel('Average Power (W)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Total Energy Comparison
        sns.barplot(data=df, x='Model', y='total_energy_kj', ax=axes[0,1])
        axes[0,1].set_title('Total Energy Consumption by Model')
        axes[0,1].set_ylabel('Total Energy (kJ)')
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


def main(dataset_paths: Dict[str, str]) -> None:
    """
    Energy model comparison only.
    
    Args:
        dataset_paths: Mapping from model name to energy CSV path
    """
    PathManager.ensure_dirs()
    processor = EnergyProcessor(dataset_paths)

    print("=== Energy Model Comparison ===")
    summary_df = processor.summarize_all()
    save_dataframe(summary_df, 'energy_comparison_summary.xlsx')
    print(f"Saved to {PathManager.get_data_path('energy_comparison_summary.xlsx')}")
    print(summary_df.to_string(index=False))

    processor.create_energy_visualizations(summary_df)




