"""
Evaluate DSL V2 on evaluation set
"""

import json
import numpy as np
from dsl_v2_improved import SmartProgramSearcher
import time


def evaluate_v2():
    print("=" * 70)
    print("DSL V2 Evaluation - First 20 tasks")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    searcher = SmartProgramSearcher()

    correct = 0
    partial_matches = []

    # Test first 20 tasks for speed
    test_tasks = list(eval_data.items())[:20]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks...")
    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        print(f"\n[{i+1}/{total}] Task {task_id}:")

        try:
            predictions = searcher.search_with_learning(task)
            solutions = eval_solutions[task_id]

            task_correct = False
            for pred, sol in zip(predictions, solutions):
                for attempt in pred:
                    if attempt == sol:
                        correct += 1
                        task_correct = True
                        print(f"  ✓ CORRECT!")
                        break

                    # Check partial match
                    try:
                        pred_arr = np.array(attempt)
                        sol_arr = np.array(sol)
                        if pred_arr.shape == sol_arr.shape:
                            match = np.sum(pred_arr == sol_arr) / pred_arr.size
                            if match > 0.8:
                                partial_matches.append((task_id, match * 100))
                    except:
                        pass

            if not task_correct:
                print(f"  ✗ Incorrect")

        except Exception as e:
            print(f"  Error: {e}")

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("DSL V2 Results")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nTotal: {total}")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s per task)")

    if partial_matches:
        print(f"\nPartial matches (>80%): {len(partial_matches)}")
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in partial_matches[:5]:
            print(f"  {task_id}: {pct:.1f}%")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches)
    }


if __name__ == "__main__":
    results = evaluate_v2()

    with open('dsl_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved to dsl_v2_results.json")