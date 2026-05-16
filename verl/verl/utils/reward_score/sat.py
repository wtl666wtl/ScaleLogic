import torch
import re
import sys

def calc_sat_value(clause, solution):
    def parse_literals(clause_str):
        literals = []
        i = 0
        while i < len(clause_str):
            if clause_str[i] == '!':
                literals.append(clause_str[i:i+2])
                i += 2
            else:
                literals.append(clause_str[i])
                i += 1
        return literals
    
    for subclause in clause.split(' & '):
        satisfied = False
        for lit in parse_literals(subclause):
            neg = False
            if lit.startswith('!'):
                var = lit[1]
                neg = True
            else:
                var = lit
            
            idx = ord(var) - ord('A')
            if idx >= len(solution):
                val = '0'
            else:
                val = solution[idx]
            
            if (neg and val == '0') or (not neg and val == '1'):
                satisfied = True
                break
        
        if not satisfied:
            return 0
    return 1



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


def compute_score(solution_str: str, ground_truth: str, extra_info: dict):
    predicted_arrangement = None

    format_correct = validate_response_structure(solution_str)
    format_score = 0 if format_correct else -1

    if format_score == 0:
        predicted_arrangement = extract_solution(solution_str.strip())
    else:
        predicted_arrangement = None

    if predicted_arrangement is None or len(predicted_arrangement) != len(ground_truth):
        score = -1
    else:
        score = calc_sat_value(extra_info["clause"], predicted_arrangement)

    score = score if score >= 0 else 0

    return score
