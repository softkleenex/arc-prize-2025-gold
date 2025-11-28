"""V8 Hybrid 평가"""

import json
import numpy as np
from pathlib import Path
import sys

# V8 import
sys.path.insert(0, str(Path(__file__).parent))
from kaggle_notebook_v8 import HybridSolver

def evaluate_v8():
    print("=" * 70)
    print("V8 Hybrid Ultimate - Local Evaluation")
    print("=" * 70)

    # Load evaluation data
    eval_path = Path('data/arc-agi_evaluation_challenges.json')
    with open(eval_path, 'r') as f:
        eval_challenges = json.load(f)

    eval_sol_path = Path('data/arc-agi_evaluation_solutions.json')
    with open(eval_sol_path, 'r') as f:
        eval_solutions = json.load(f)

    solver = HybridSolver()

    correct = 0
    partial_matches = []
    total_tasks = len(eval_challenges)

    print(f"\nEvaluating {total_tasks} tasks...")

    for i, (task_id, task) in enumerate(eval_challenges.items()):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{total_tasks} tasks...")

        try:
            # Solve task
            predictions = solver.solve(task, max_attempts=2)

            # Check each test case
            for test_idx, test_pred in enumerate(predictions):
                expected = eval_solutions[task_id][test_idx]

                # Check both attempts
                for attempt_idx, attempt in enumerate(test_pred):
                    if attempt == expected:
                        correct += 1
                        break
                    else:
                        # Calculate partial match
                        try:
                            pred_arr = np.array(attempt)
                            exp_arr = np.array(expected)

                            if pred_arr.shape == exp_arr.shape:
                                match_ratio = np.sum(pred_arr == exp_arr) / pred_arr.size
                                if match_ratio > 0.8:  # >80% match
                                    partial_matches.append((task_id, match_ratio * 100))
                        except:
                            pass
        except Exception as e:
            print(f"  Error on task {task_id}: {e}")

    # Results
    print("\n" + "=" * 70)
    print("V8 Results")
    print("=" * 70)

    accuracy = correct / total_tasks * 100
    print(f"\nTotal tasks: {total_tasks}")
    print(f"Correct: {correct} ({accuracy:.2f}%)")
    print(f"Partial matches (>80%): {len(partial_matches)}")

    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 partial matches:")
        for task_id, ratio in partial_matches[:10]:
            print(f"  - {task_id}: {ratio:.2f}%")

    print("\n" + "=" * 70)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial_count': len(partial_matches),
        'top_partial': partial_matches[:5] if partial_matches else []
    }

if __name__ == "__main__":
    results = evaluate_v8()

    # Save results
    with open('v8_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to: v8_evaluation.json")
