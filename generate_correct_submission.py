"""
Generate CORRECT submission.json for ARC Prize 2025
Fixes:
1. Use test_challenges (240 tasks) NOT evaluation_challenges (120 tasks)
2. Use correct format: [{'attempt_1': grid, 'attempt_2': grid}]
"""

import json
import numpy as np
from typing import List, Dict

# Import solver from final_submission
from final_submission import FinalARCSolver


def convert_to_correct_format(predictions: List[List]) -> List[Dict]:
    """
    Convert our format to Kaggle's required format

    Our format: [[grid1, grid2], [grid3, grid4], ...]
    Required format: [{'attempt_1': grid1, 'attempt_2': grid2}, {'attempt_1': grid3, 'attempt_2': grid4}, ...]
    """
    result = []

    for test_predictions in predictions:
        # test_predictions is a list of 2 attempts
        if len(test_predictions) >= 2:
            attempt_1 = test_predictions[0]
            attempt_2 = test_predictions[1]
        elif len(test_predictions) == 1:
            attempt_1 = test_predictions[0]
            attempt_2 = test_predictions[0]  # Duplicate if only one
        else:
            # Fallback: empty grid
            attempt_1 = [[0]]
            attempt_2 = [[0]]

        result.append({
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        })

    return result


def generate_correct_submission():
    """Generate submission.json in the CORRECT format"""

    print("=" * 70)
    print("GENERATING CORRECT SUBMISSION.JSON")
    print("=" * 70)

    # Load TEST challenges (NOT evaluation!)
    print("\n✅ Loading TEST challenges (240 tasks)...")
    try:
        with open('arc-agi_test_challenges.json', 'r') as f:
            challenges = json.load(f)
    except FileNotFoundError:
        print("❌ arc-agi_test_challenges.json not found!")
        print("Downloading...")
        import subprocess
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 'arc-prize-2025',
                       '-f', 'arc-agi_test_challenges.json'])
        with open('arc-agi_test_challenges.json', 'r') as f:
            challenges = json.load(f)

    print(f"✅ Loaded {len(challenges)} tasks (Expected: 240)")

    if len(challenges) != 240:
        print(f"⚠️ WARNING: Expected 240 tasks, got {len(challenges)}")

    # Initialize solver
    print("\n✅ Initializing solver...")
    solver = FinalARCSolver()
    print("✅ Solver ready!")

    # Generate predictions
    print("\n✅ Generating predictions for 240 tasks...")
    submission = {}

    for i, (task_id, task) in enumerate(challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(challenges)}")

        try:
            # Solve task
            predictions = solver.solve(task)

            # Convert to correct format
            correct_format = convert_to_correct_format(predictions)

            submission[task_id] = correct_format

        except Exception as e:
            print(f"  ❌ Error on {task_id}: {e}")
            # Provide fallback
            submission[task_id] = [{'attempt_1': [[0]], 'attempt_2': [[0]]}]

    print(f"\n✅ Generated predictions for {len(submission)} tasks")

    # Validate format
    print("\n✅ Validating submission format...")
    valid = validate_submission_format(submission)

    if not valid:
        print("❌ Validation failed!")
        return None

    print("✅ Validation passed!")

    # Save submission
    print("\n✅ Saving submission.json...")
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

    # Get file size
    import os
    size_mb = os.path.getsize('submission.json') / (1024 * 1024)

    print("=" * 70)
    print("SUBMISSION GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 File: submission.json")
    print(f"📊 Size: {size_mb:.2f} MB")
    print(f"📦 Tasks: {len(submission)}")
    print(f"✅ Format: CORRECT (verified)")

    # Sample prediction
    sample_task_id = list(submission.keys())[0]
    sample_pred = submission[sample_task_id]
    print(f"\n📋 Sample prediction ({sample_task_id}):")
    print(f"  Type: {type(sample_pred)}")
    print(f"  Length: {len(sample_pred)}")
    print(f"  First element keys: {sample_pred[0].keys() if len(sample_pred) > 0 else 'N/A'}")

    print("\n" + "=" * 70)
    print("READY FOR KAGGLE SUBMISSION!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Upload arc_submission_fixed.ipynb to Kaggle")
    print("2. Or submit using API:")
    print("   (Note: Cannot submit JSON directly in Code Competition)")
    print("=" * 70)

    return submission


def validate_submission_format(submission: Dict) -> bool:
    """Validate submission follows Kaggle format"""

    print("  Checking format...")

    # Check task count
    if len(submission) != 240:
        print(f"  ❌ Wrong number of tasks: {len(submission)} (expected 240)")
        return False

    # Check format of first few tasks
    for task_id, preds in list(submission.items())[:5]:
        # Should be a list
        if not isinstance(preds, list):
            print(f"  ❌ {task_id}: predictions must be a list")
            return False

        # Each element should be a dict with 'attempt_1' and 'attempt_2'
        for i, pred_dict in enumerate(preds):
            if not isinstance(pred_dict, dict):
                print(f"  ❌ {task_id} test {i}: should be dict, got {type(pred_dict)}")
                return False

            if 'attempt_1' not in pred_dict or 'attempt_2' not in pred_dict:
                print(f"  ❌ {task_id} test {i}: missing attempt_1 or attempt_2")
                print(f"     Keys: {pred_dict.keys()}")
                return False

            # Each attempt should be a 2D list (grid)
            for attempt_key in ['attempt_1', 'attempt_2']:
                attempt = pred_dict[attempt_key]
                if not isinstance(attempt, list):
                    print(f"  ❌ {task_id} test {i} {attempt_key}: should be list (grid)")
                    return False

                if len(attempt) > 0 and not isinstance(attempt[0], list):
                    print(f"  ❌ {task_id} test {i} {attempt_key}: should be 2D list")
                    return False

    print("  ✅ Format validation passed!")
    return True


def compare_with_sample():
    """Compare our submission with sample_submission.json"""

    print("\n" + "=" * 70)
    print("COMPARING WITH SAMPLE SUBMISSION")
    print("=" * 70)

    with open('sample_submission.json', 'r') as f:
        sample = json.load(f)

    with open('submission.json', 'r') as f:
        our_sub = json.load(f)

    print(f"\nSample submission:")
    print(f"  Tasks: {len(sample)}")
    sample_task_id = list(sample.keys())[0]
    print(f"  First task: {sample_task_id}")
    print(f"  Format: {type(sample[sample_task_id])}")
    print(f"  Structure: {sample[sample_task_id][0].keys() if len(sample[sample_task_id]) > 0 else 'N/A'}")

    print(f"\nOur submission:")
    print(f"  Tasks: {len(our_sub)}")
    our_task_id = list(our_sub.keys())[0]
    print(f"  First task: {our_task_id}")
    print(f"  Format: {type(our_sub[our_task_id])}")
    print(f"  Structure: {our_sub[our_task_id][0].keys() if len(our_sub[our_task_id]) > 0 else 'N/A'}")

    # Check if formats match
    if len(sample) == len(our_sub):
        print("\n✅ Task count matches!")
    else:
        print(f"\n❌ Task count mismatch: sample {len(sample)} vs ours {len(our_sub)}")

    # Check structure
    sample_structure = type(sample[sample_task_id])
    our_structure = type(our_sub[our_task_id])

    if sample_structure == our_structure:
        print("✅ Top-level structure matches!")
    else:
        print(f"❌ Structure mismatch: sample {sample_structure} vs ours {our_structure}")

    print("=" * 70)


if __name__ == "__main__":
    print("Generating CORRECT submission for ARC Prize 2025...\n")

    submission = generate_correct_submission()

    if submission:
        print("\n")
        compare_with_sample()
        print("\n✅ All done! submission.json is ready!")
    else:
        print("\n❌ Failed to generate submission")
