#!/usr/bin/env python3
"""
Figure 6 Data Inspector

"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from processors.core_parser import ScalingResultsProcessor

def inspect_available_data():
    
    processor = ScalingResultsProcessor("../tegra/scaling")
    results = processor.parse_all_results()
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"✅ Found {len(results)} results")
    
    first_result = results[0]
    print(f"\n📋 Sample result from: {first_result.metadata.model_name}")
    print(f"   Samples: {first_result.metadata.num_samples}")
    print(f"   Token budget: {first_result.metadata.token_budget}")
    print(f"   Seed: {first_result.metadata.seed}")
    
    print(f"\n📊 Available metrics:")
    metrics = first_result.metrics
    print(f"   Accuracy: {metrics.accuracy}")
    print(f"   Total questions: {metrics.total_questions}")
    print(f"   Avg time per question: {metrics.avg_time_per_question}")
    print(f"   Avg tokens per second: {metrics.avg_tokens_per_second}")
    print(f"   Avg decode time: {metrics.avg_decode_time}")
    print(f"   Avg power consumption: {metrics.avg_power_consumption}")
    print(f"   Total energy consumed: {metrics.total_energy_consumed}")
    
    models = set(r.metadata.model_name for r in results)
    ps_factors = set(r.metadata.num_samples for r in results)
    token_budgets = set(r.metadata.token_budget for r in results)
    seeds = set(r.metadata.seed for r in results)
    
    print(f"\n📈 Data summary:")
    print(f"   Models: {sorted(models)}")
    print(f"   PS factors (sample counts): {sorted(ps_factors)}")
    print(f"   Token budgets: {sorted(token_budgets)}")
    print(f"   Seeds: {sorted(seeds)}")
    
    energy_results = [r for r in results if r.metrics.total_energy_consumed is not None]
    decode_results = [r for r in results if r.metrics.avg_decode_time is not None]
    
    print(f"\n🔋 Energy data availability:")
    print(f"   Results with energy data: {len(energy_results)}/{len(results)}")
    print(f"   Results with decode time: {len(decode_results)}/{len(results)}")
    
    if decode_results:
        print(f"\n⏱️  Sample decode times:")
        for i, r in enumerate(decode_results[:5]):
            print(f"   {r.metadata.model_name} ({r.metadata.num_samples} samples): {r.metrics.avg_decode_time:.3f}s")

if __name__ == '__main__':
    inspect_available_data() 