"""
Evaluate DSL V1 on evaluation set
"""

import json
import numpy as np
from program_search_v1 import DSLSolver
import time


def evaluate():
    print("=" * 70)
    print("DSL V1 Evaluation on 120 tasks")
    print("=" * 70)

    # Load evaluation data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_challenges = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = DSLSolver()

    correct_count = 0
    partial_matches = []
    total_tasks = len(eval_challenges)

    print(f"\nEvaluating {total_tasks} tasks...")
    start_time = time.time()

    for i, (task_id, task) in enumerate(eval_challenges.items()):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{total_tasks} tasks ({elapsed:.1f}s)...")

        try:
            predictions = solver.solve(task)

            # Check correctness
            solutions = eval_solutions[task_id]
            task_correct = 0

            for pred_attempts, solution in zip(predictions, solutions):
                for attempt in pred_attempts:
                    if attempt == solution:
                        task_correct += 1
                        break

                    # Calculate partial match
                    try:
                        pred_arr = np.array(attempt)
                        sol_arr = np.array(solution)

                        if pred_arr.shape == sol_arr.shape:
                            match_ratio = np.sum(pred_arr == sol_arr) / pred_arr.size
                            if match_ratio > 0.8:
                                partial_matches.append((task_id, match_ratio * 100))
                    except:
                        pass

            if task_correct == len(solutions):
                correct_count += 1
                print(f"  ✓ {task_id} CORRECT!")

        except Exception as e:
            print(f"  Error on {task_id}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DSL V1 Results")
    print("=" * 70)

    accuracy = correct_count / total_tasks * 100
    print(f"\nTotal tasks: {total_tasks}")
    print(f"Correct: {correct_count} ({accuracy:.2f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/total_tasks:.1f}s per task)")

    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        print(f"\nPartial matches (>80%): {len(partial_matches)}")
        print("Top 5:")
        for task_id, ratio in partial_matches[:5]:
            print(f"  - {task_id}: {ratio:.2f}%")

    print("\n" + "=" * 70)

    # Save results
    results = {
        'accuracy': accuracy,
        'correct': correct_count,
        'partial_count': len(partial_matches),
        'time': elapsed
    }

    with open('dsl_v1_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to dsl_v1_evaluation.json")


if __name__ == "__main__":
    evaluate()