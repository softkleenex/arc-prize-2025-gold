"""
Evaluate DSL V3 Pattern-Based on evaluation set
"""

import json
import numpy as np
from dsl_v3_pattern import AdvancedProgramSearcher
import time


def evaluate_v3():
    print("=" * 70)
    print("DSL V3 Pattern-Based Evaluation - First 30 tasks")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    searcher = AdvancedProgramSearcher()

    correct = 0
    partial_matches = []
    perfect_matches = []

    # Test first 30 tasks for better coverage
    test_tasks = list(eval_data.items())[:30]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks...")
    print("Pattern-based approach with:")
    print("• Advanced color learning")
    print("• Object replication/multiplication")
    print("• Pattern completion")
    print("• Smart composition\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        print(f"\n[{i+1}/{total}] Task {task_id}:", end=' ')

        try:
            predictions = searcher.search(task, max_programs=2)
            solutions = eval_solutions[task_id]

            task_correct = False
            best_match = 0

            for pred_list, sol in zip(predictions, solutions):
                for attempt in pred_list:
                    try:
                        # Convert to arrays for comparison
                        pred_arr = np.array(attempt)
                        sol_arr = np.array(sol)

                        # Check exact match
                        if pred_arr.shape == sol_arr.shape and np.array_equal(pred_arr, sol_arr):
                            correct += 1
                            task_correct = True
                            perfect_matches.append(task_id)
                            print("✓ CORRECT!")
                            break

                        # Check partial match
                        if pred_arr.shape == sol_arr.shape:
                            match = np.sum(pred_arr == sol_arr) / pred_arr.size
                            best_match = max(best_match, match)
                    except Exception as e:
                        pass

                if task_correct:
                    break

            if not task_correct:
                if best_match > 0.8:
                    partial_matches.append((task_id, best_match * 100))
                    print(f"~ Partial ({best_match*100:.1f}%)")
                else:
                    print("✗ Incorrect")

        except Exception as e:
            print(f"Error: {e}")

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("DSL V3 Pattern-Based Results")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nTotal: {total}")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Partial (>80%): {len(partial_matches)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s per task)")

    if perfect_matches:
        print(f"\nPerfect matches: {perfect_matches[:5]}")

    if partial_matches:
        print(f"\nBest partial matches:")
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in partial_matches[:5]:
            print(f"  {task_id}: {pct:.1f}%")

    # Analyze pattern types that worked
    if correct > 0:
        print("\n✓ Pattern-based approach showing improvement!")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_matches
    }


if __name__ == "__main__":
    results = evaluate_v3()

    with open('dsl_v3_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved to dsl_v3_results.json")