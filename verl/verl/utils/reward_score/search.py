import re
import random
import ast
import operator
import json
import signal
import contextlib
import numpy as np
import math
import copy
import traceback

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    #print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 0),
        'think_end': ('</think>', 0),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        #print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            #print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        #print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        #print("  Tag sequence validation passed")
        pass

    return validation_passed

def extract_solution(solution_str):
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.S)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1)
        return final_answer.strip()
    else:
        return None


def compute_score(solution_str, ground_truth) -> float:
    predicted_arrangement = None

    format_correct = validate_response_structure(solution_str)
    format_score = 0 if format_correct else -1

    if format_score == 0:
        predicted_arrangement = extract_solution(solution_str.strip())
    else:
        predicted_arrangement = None

    if predicted_arrangement is None:
        score = -1
    elif isinstance(ground_truth, str):
        score = int(predicted_arrangement == ground_truth)
    else:
        score = int(predicted_arrangement in ground_truth)

    score = score if score >= 0 else 0

    return score