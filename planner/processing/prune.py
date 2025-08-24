#!/usr/bin/env python3
"""
Text cleaning utilities for Natural-Plan model predictions.

Removes instruction noise and formatting artifacts from model outputs.
"""

import re
from typing import List


# Patterns to remove from model predictions
NOISE_PATTERNS = [
    re.compile(r"^Be concise and direct\.", re.I),
    re.compile(r"Okay, so I'm trying to figure out the best way to meet as many friends as possible in San Francisco today", re.I),
    re.compile(r"the best way to meet as many friends as possible in San Francisco today", re.I),
    re.compile(r"Each solution must be concise", re.I),
    re.compile(r"Each solution should be concise", re.I),
    re.compile(r"So, each solution should be concise", re.I),
    re.compile(r"Each solution must be concise and fit within the token limit", re.I),
    re.compile(r"Each step is a separate line, so each line is a token", re.I),
    re.compile(r"followed by the steps, each step being a concise sentence, and the entire thing under 512 tokens", re.I),
    re.compile(r"Keep your response concise", re.I),
    re.compile(r"Be direct and concise", re.I),
    re.compile(r"Please be concise", re.I),
    re.compile(r"You must be concise and direct", re.I),
    re.compile(r"You must be concise and clear", re.I),
    re.compile(r"You must limit your answer to exactly", re.I),
    re.compile(r"You must limit your answer to", re.I),
    re.compile(r"You can use any combination", re.I),
    re.compile(r"You can use any abbreviations", re.I),
    re.compile(r"Use only standard abbreviations", re.I),
    re.compile(r"You must be careful with your", re.I),
    
    # Solution format instructions
    re.compile(r"So, the solution should be", re.I),
    re.compile(r"Each solution is a separate", re.I),
    re.compile(r"Each problem is", re.I),
    re.compile(r"Each solution is a", re.I),
    re.compile(r"and follow the same solution format", re.I),
    re.compile(r"follow the same format", re.I),
    
    # Thinking/planning noise
    re.compile(r"Alright, so I'm trying to figure out", re.I),
    re.compile(r"Alright, I need to figure out", re.I),
    re.compile(r"Alright, let's tackle this problem", re.I),
    re.compile(r"Okay, so I'm trying to figure out", re.I),
    re.compile(r"First, I need to", re.I),
    re.compile(r"First, I'll", re.I),
    re.compile(r"First, I should", re.I),
    re.compile(r"Now, let's", re.I),
    re.compile(r"Now, I need to", re.I),
    re.compile(r"Wait, no, the user", re.I),
    re.compile(r"Wait, but", re.I),
    re.compile(r"But wait, the user's instruction says", re.I),
    re.compile(r"I start at", re.I),
    re.compile(r"I arrive at", re.I),
    re.compile(r"I also have", re.I),
    re.compile(r"I should probably", re.I),
    
    # Format markers
    re.compile(r"```", re.I),
    re.compile(r"^\"$", re.I),  # standalone quote marks
    
    # Template fragments
    re.compile(r"', followed by", re.I),
    re.compile(r"', then the", re.I),
    re.compile(r"', and", re.I),
    re.compile(r"and then the plan", re.I),
    re.compile(r"Start at Downtown", re.I),
    re.compile(r"followed by the schedule", re.I),
    re.compile(r"followed by the plan", re.I),
    re.compile(r"followed by the steps", re.I),
    re.compile(r"ollowed by the steps", re.I),  # catches "followed" with missing "f"
    re.compile(r"then the steps", re.I),
    re.compile(r"and following the same format", re.I),
    re.compile(r"</think>", re.I),
    re.compile(r"SOLUTION:", re.I),
    re.compile(r"Now, the problem you need to solve is:", re.I),
    re.compile(r"You are visiting San Francisco for the day", re.I),
    re.compile(r"' and end with", re.I),
    re.compile(r"Wait, the constraints are:", re.I),
    re.compile(r"Travel distances \\(in minutes\\):", re.I),
    re.compile(r"You start at", re.I),
    re.compile(r"You arrive at", re.I),
    re.compile(r"Use the", re.I),
    re.compile(r"You'll be able to", re.I),
]


def clean_prediction(text: str, task: str = None) -> str:
    """
    Remove instruction noise from generated text, keeping everything else intact.
    
    Args:
        text: Raw model prediction text
        task: Task name (unused currently, for future task-specific cleaning)
        
    Returns:
        Cleaned prediction text
    """
    cleaned_text = text
    for pattern in NOISE_PATTERNS:
        cleaned_text = pattern.sub('', cleaned_text)
    
    lines = [line.strip() for line in cleaned_text.splitlines()]
    lines = [l for l in lines if l]  
    
    return "\n".join(lines)


def clean_results(results_data: dict, task: str = None) -> dict:
    """
    Clean predictions in a results JSON structure.
    
    Args:
        results_data: Results JSON data with question_results
        task: Task name for task-specific cleaning
        
    Returns:
        Modified results data with cleaned predictions
    """
    for question_result in results_data.get("question_results", []):
        original_text = question_result.get("generated_text", "")
        if original_text:
            question_result["generated_text"] = clean_prediction(original_text, task)
    
    return results_data
