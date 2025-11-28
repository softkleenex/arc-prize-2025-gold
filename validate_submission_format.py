#!/usr/bin/env python3
"""
Submission.json 형식 검증 스크립트
이전 10번 실패 원인 중 하나가 형식 오류일 가능성 체크
"""
import json
import sys
from pathlib import Path

def validate_submission_format(submission_path):
    """
    Kaggle ARC Prize 2025 submission.json 형식 검증
    
    올바른 형식:
    {
        "task_id_1": [
            [[0,1,2], [3,4,5]],  # attempt_1 (2D array)
            [[0,1,2], [3,4,5]]   # attempt_2 (2D array)
        ],
        "task_id_2": [...],
        ...
    }
    """
    print("="*60)
    print("Submission Format Validator")
    print("="*60)
    
    if not Path(submission_path).exists():
        print(f"❌ File not found: {submission_path}")
        return False
    
    print(f"✓ File found: {submission_path}")
    
    try:
        with open(submission_path) as f:
            submission = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    
    print(f"✓ Valid JSON")
    
    # Check top-level structure
    if not isinstance(submission, dict):
        print(f"❌ Submission must be a dict, got {type(submission)}")
        return False
    
    print(f"✓ Submission is dict")
    print(f"  Total tasks: {len(submission)}")
    
    errors = []
    warnings = []
    
    for task_id, attempts in submission.items():
        # Check task_id format (should be 8-char hex string)
        if not isinstance(task_id, str) or len(task_id) != 8:
            errors.append(f"{task_id}: Invalid task_id format (should be 8 chars)")
            continue
        
        # Check attempts is list
        if not isinstance(attempts, list):
            errors.append(f"{task_id}: attempts must be list, got {type(attempts)}")
            continue
        
        # Check exactly 2 attempts
        if len(attempts) != 2:
            errors.append(f"{task_id}: Need exactly 2 attempts, got {len(attempts)}")
            continue
        
        # Check each attempt
        for i, attempt in enumerate(attempts):
            attempt_name = f"{task_id}[attempt_{i+1}]"
            
            # Check attempt is 2D list
            if not isinstance(attempt, list):
                errors.append(f"{attempt_name}: Must be list, got {type(attempt)}")
                continue
            
            if len(attempt) == 0:
                errors.append(f"{attempt_name}: Empty array")
                continue
            
            if not isinstance(attempt[0], list):
                errors.append(f"{attempt_name}: Must be 2D array (list of lists)")
                continue
            
            # Check all rows have same length
            row_lengths = [len(row) for row in attempt]
            if len(set(row_lengths)) > 1:
                warnings.append(f"{attempt_name}: Inconsistent row lengths {row_lengths}")
            
            # Check values are 0-9
            for row_idx, row in enumerate(attempt):
                for col_idx, val in enumerate(row):
                    if not isinstance(val, int) or not (0 <= val <= 9):
                        errors.append(
                            f"{attempt_name}[{row_idx},{col_idx}]: "
                            f"Value must be int 0-9, got {val}"
                        )
    
    # Print results
    print(f"\n{'='*60}")
    print("Validation Results")
    print("="*60)
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    print(f"✓ No errors found!")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings[:5]:
            print(f"  - {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
    
    # Print sample
    sample_task = list(submission.keys())[0]
    sample_attempts = submission[sample_task]
    print(f"\n{'='*60}")
    print(f"Sample Task: {sample_task}")
    print("="*60)
    print(f"Attempt 1 shape: {len(sample_attempts[0])} x {len(sample_attempts[0][0])}")
    print(f"Attempt 2 shape: {len(sample_attempts[1])} x {len(sample_attempts[1][0])}")
    print(f"Attempt 1 preview:")
    for row in sample_attempts[0][:3]:
        print(f"  {row[:10]}{'...' if len(row) > 10 else ''}")
    
    print(f"\n{'='*60}")
    print("✅ SUBMISSION FORMAT VALID")
    print("="*60)
    return True


if __name__ == "__main__":
    # Test with existing submission files
    test_files = [
        "submission.json",
        "CompressARC/submission.json",
        "submissions/submission.json",
    ]
    
    print("Searching for submission.json files...\n")
    
    found = False
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"Found: {test_file}\n")
            validate_submission_format(test_file)
            found = True
            print("\n")
    
    if not found:
        print("No submission.json files found.")
        print("\nTo validate a specific file, run:")
        print("  python3 validate_submission_format.py <path_to_submission.json>")
