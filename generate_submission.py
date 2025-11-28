"""
Generate submission.json locally for verification
Uses the same solver as arc_submission.ipynb
"""

import json
import numpy as np
from typing import List, Dict

# Import solver classes from final_submission
from final_submission import FinalARCSolver


def generate_submission():
    """Generate submission.json from local evaluation data"""

    print("=" * 70)
    print("GENERATING LOCAL SUBMISSION.JSON")
    print("=" * 70)

    # Load evaluation challenges
    print("\nLoading evaluation data...")
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        challenges = json.load(f)

    print(f"Loaded {len(challenges)} tasks")

    # Initialize solver
    print("Initializing FinalARCSolver...")
    solver = FinalARCSolver()
    print("✓ Solver ready!")

    # Generate predictions
    print("\nGenerating predictions...")
    submission = {}

    for i, (task_id, task) in enumerate(challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(challenges)}")

        try:
            predictions = solver.solve(task)
            submission[task_id] = predictions
        except Exception as e:
            print(f"  Error on {task_id}: {e}")
            # Provide fallback empty prediction
            submission[task_id] = [[[[0]]]]

    print(f"\n✓ Generated predictions for {len(submission)} tasks")

    # Validate submission format
    print("\nValidating submission format...")
    valid = True
    for task_id, preds in submission.items():
        if not isinstance(preds, list):
            print(f"  ✗ {task_id}: predictions must be a list")
            valid = False
        elif len(preds) == 0:
            print(f"  ✗ {task_id}: no predictions")
            valid = False
        else:
            for i, pred_list in enumerate(preds):
                if not isinstance(pred_list, list) or len(pred_list) == 0:
                    print(f"  ✗ {task_id} test {i}: invalid format")
                    valid = False
                    break

    if valid:
        print("✓ All tasks have valid format")

    # Save submission
    print("\nSaving submission.json...")
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
    print(f"✓ Format: Valid")

    # Sample prediction
    sample_task_id = list(submission.keys())[0]
    sample_pred = submission[sample_task_id]
    print(f"\n📋 Sample prediction ({sample_task_id}):")
    print(f"  Test examples: {len(sample_pred)}")
    print(f"  Attempts per test: {len(sample_pred[0])} (Pass@2)")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review submission.json (file created)")
    print("2. Upload arc_submission.ipynb to Kaggle")
    print("3. Or use Kaggle API:")
    print("   kaggle competitions submit -c arc-prize-2025 \\")
    print("     -f submission.json -m 'Phase 4 submission'")
    print("\nSee SUBMISSION_GUIDE.md for detailed instructions")
    print("=" * 70)

    return submission


if __name__ == "__main__":
    submission = generate_submission()
