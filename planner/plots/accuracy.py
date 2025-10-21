#!/usr/bin/env python3

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
Accuracy plotting functionality for Natural-Plan evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from ..analysis import EvaluationResult


def plot_accuracy(results: List[EvaluationResult], 
                                 task: str = None, 
                                 models: List[str] = None, 
                                 output_dir: Path = None):
    """Create accuracy vs tokens/question plots"""
    if not results:
        print("✗ No results provided for plotting")
        return
        
    # Filter results
    filtered_results = results
    if task:
        filtered_results = [r for r in filtered_results if r.task == task]
    if models:
        filtered_results = [r for r in filtered_results if r.model_size in models]
    
    if not filtered_results:
        print(f"✗ No results found for task={task}, models={models}")
        return
    
    # Convert to DataFrame for easier plotting
    df_data = []
    for result in filtered_results:
        df_data.append({
            'Task': result.task,
            'Model': result.model_size,
            'Eval Type': result.eval_type,
            'Token Limit': result.token_limit,
            'Accuracy': result.accuracy,
            'Tokens per Question': result.avg_tokens_per_question,
            'Timestamp': result.timestamp
        })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("✗ No data to plot")
        return
    
    # Create plots for each task
    tasks = df['Task'].unique()
    
    for task_name in tasks:
        task_df = df[df['Task'] == task_name]
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with different colors for eval types and shapes for models
        eval_types = task_df['Eval Type'].unique()
        models = task_df['Model'].unique()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(eval_types)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        for i, eval_type in enumerate(eval_types):
            for j, model in enumerate(models):
                subset = task_df[(task_df['Eval Type'] == eval_type) & (task_df['Model'] == model)]
                if not subset.empty:
                    plt.scatter(subset['Tokens per Question'], subset['Accuracy'],
                              c=[colors[i]], marker=markers[j % len(markers)], s=100,
                              label=f"{model} ({eval_type})", alpha=0.7)
        
        plt.xlabel('Average Tokens per Question', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Accuracy vs Tokens per Question - {task_name.title()} Task', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            plot_path = output_dir / f"accuracy_vs_tokens_{task_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"* Plot saved: {plot_path}")
        else:
            plt.show()
        
        plt.close()
