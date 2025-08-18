"""
Comprehensive summary table creation with performance and energy metrics.
"""
import pandas as pd
from typing import List, Dict, Any

from plotHandler import ENERGY_DATASETS, SECTION_MAPPING
from .energy import EnergyProcessor


def create_comprehensive_summary(all_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comprehensive summary table with all requested metrics.
    
    Args:
        all_summaries: List of dictionaries containing performance summaries
        
    Returns:
        DataFrame with columns: Model, Config, total_questions, accuracy, 
        avg_inference_time, avg_tokens, total_tokens, avg_power_w, peak_power_w, 
        tokens_per_s, perf_per_watt, joules_per_question
    """
    print(f"\n>>>>Creating comprehensive summary table...")
    
    performance_df = pd.DataFrame(all_summaries)
    
    try:
        all_energy_summaries = []
        for section_name, energy_dict in ENERGY_DATASETS.items():
            energy_processor = EnergyProcessor(energy_dict)
            energy_summary = energy_processor.summarize_all()
            
            if not energy_summary.empty:
                energy_summary['Section'] = section_name
                all_energy_summaries.append(energy_summary)
        
        energy_df = pd.concat(all_energy_summaries, ignore_index=True) if all_energy_summaries else pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load energy data: {e}")
        energy_df = pd.DataFrame()
    
    comprehensive_data = []
    
    for _, row in performance_df.iterrows():
        section = row['Section']
        model_name = row['Model']
        
        config = SECTION_MAPPING.get(section, section)
        if section == 'Direct2Text':
            config = 'Direct'
        
        base_model_name = (model_name.replace('-128t_NC', '')
                          .replace('-256t_NC', '')
                          .replace('-128t', '')
                          .replace('-256t', '')
                          .replace('-no-cut', '')
                          .replace('-NR', '')
                          .strip())
        
        energy_row = None
        if not energy_df.empty:
            energy_matches = energy_df[
                (energy_df['Model'] == model_name) & 
                (energy_df['Section'] == section)
            ]
            if energy_matches.empty:
                energy_matches = energy_df[
                    (energy_df['Model'].str.contains(base_model_name.replace('DSR1-', ''), na=False)) & 
                    (energy_df['Section'] == section)
                ]
            
            if not energy_matches.empty:
                energy_row = energy_matches.iloc[0]
        
        summary_row = {
            'Model': base_model_name,
            'Config': config,
            'total_questions': int(row.get('sample_size', 0)),
            'accuracy': float(row.get('accuracy', 0)),
            'avg_inference_time': float(row.get('avg_inference_time', 0)),
            'avg_tokens': float(row.get('avg_tokens', row.get('avg_output_tokens', 0))),
            'total_tokens': int(row.get('total_tokens', 0)),
            'tokens_per_s': float(row.get('avg_tokens_per_second', 0))
        }
        
        if energy_row is not None:
            summary_row.update({
                'avg_power_w': float(energy_row.get('avg_power_w', 0)),
                'peak_power_w': float(energy_row.get('peak_power_w', 0)),
                'total_energy_kj': float(energy_row.get('total_energy_kj', 0)),
                'duration_s': float(energy_row.get('duration_s', 0))
            })
            
            avg_power = summary_row['avg_power_w']
            tokens_per_s = summary_row['tokens_per_s']
            total_questions = summary_row['total_questions']
            total_energy_j = summary_row['total_energy_kj'] * 1000
            
            summary_row['perf_per_watt'] = tokens_per_s / avg_power if avg_power > 0 else 0
            summary_row['joules_per_question'] = total_energy_j / total_questions if total_questions > 0 else 0
            
        else:
            summary_row.update({
                'avg_power_w': 0.0,
                'peak_power_w': 0.0,
                'total_energy_kj': 0.0,
                'duration_s': 0.0,
                'perf_per_watt': 0.0,
                'joules_per_question': 0.0
            })
        
        comprehensive_data.append(summary_row)
    
    columns_order = [
        'Model', 'Config', 'total_questions', 'accuracy', 'avg_inference_time',
        'avg_tokens', 'total_tokens', 'avg_power_w', 'peak_power_w', 
        'tokens_per_s', 'perf_per_watt', 'joules_per_question'
    ]
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    
    for col in columns_order:
        if col not in comprehensive_df.columns:
            comprehensive_df[col] = 0.0
    
    comprehensive_df = comprehensive_df[columns_order]
    
    rounding_rules = {
        'accuracy': 1,
        'avg_inference_time': 3,
        'avg_tokens': 1,
        'avg_power_w': 2,
        'peak_power_w': 2,
        'tokens_per_s': 3,
        'perf_per_watt': 4,
        'joules_per_question': 2
    }
    
    for col, decimals in rounding_rules.items():
        if col in comprehensive_df.columns:
            comprehensive_df[col] = comprehensive_df[col].round(decimals)
    
    return comprehensive_df



