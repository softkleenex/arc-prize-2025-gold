"""
Evaluate 3-Step Program Search
"""

import json
import numpy as np
from three_step_search import SmartSearchSolver
import time


def evaluate_search():
    print("=" * 70)
    print("🔍 3-STEP PROGRAM SEARCH EVALUATION")
    print("Searching for combinations of DSL primitives")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = SmartSearchSolver()

    correct = 0
    partial_matches = []
    perfect_tasks = []

    # Test first 100 tasks
    test_tasks = list(eval_data.items())[:100]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks with program search...")
    print("Search depth: 1-step, 2-step, 3-step programs\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"\n[{i+1}/{total}] Status:")
            if correct > 0:
                print(f"  ✓ {correct} CORRECT ({correct/(i+1)*100:.1f}%)")
            print(f"  Time: {elapsed:.1f}s")

        try:
            predictions = solver.solve(task)
            solutions = eval_solutions[task_id]

            task_correct = False
            best_match = 0

            for pred_list, sol in zip(predictions, solutions):
                for attempt in pred_list:
                    try:
                        pred_arr = np.array(attempt)
                        sol_arr = np.array(sol)

                        # Check exact match
                        if pred_arr.shape == sol_arr.shape and np.array_equal(pred_arr, sol_arr):
                            correct += 1
                            task_correct = True
                            perfect_tasks.append(task_id)
                            print("✓", end='', flush=True)
                            break

                        # Track partial match
                        if pred_arr.shape == sol_arr.shape:
                            match = np.sum(pred_arr == sol_arr) / pred_arr.size
                            best_match = max(best_match, match)
                    except:
                        pass

                if task_correct:
                    break

            if not task_correct:
                if best_match > 0.95:
                    partial_matches.append((task_id, best_match * 100))
                    print("~", end='', flush=True)
                elif best_match > 0.8:
                    print(".", end='', flush=True)
                else:
                    print("x", end='', flush=True)

        except Exception as e:
            print("E", end='', flush=True)

    elapsed = time.time() - start
    print("\n")

    print("=" * 70)
    print("📊 3-STEP SEARCH RESULTS")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n🎯 Performance:")
    print(f"  Tasks evaluated: {total}")
    print(f"  ✓ Correct: {correct} ({accuracy:.2f}%)")
    print(f"  ~ Near (>95%): {len([p for p in partial_matches if p[1] > 95])}")
    print(f"  Time: {elapsed:.1f}s")

    if correct > 0:
        print(f"\n🎉 BREAKTHROUGH!")
        print(f"  Achieved {correct} correct with program search!")
        print(f"  Tasks solved:")
        for task in perfect_tasks[:10]:
            print(f"    • {task}")

    print("\n📈 COMPLETE SOLVER COMPARISON:")
    print(f"  1. DSL V1:          0.00%")
    print(f"  2. DSL V2:          0.00% (98.9% partial)")
    print(f"  3. DSL V3:          0.00% (95.0% partial)")
    print(f"  4. Hybrid:          0.00% (96.8% partial)")
    print(f"  5. Direct Match:    0.00% (98.9% partial)")
    print(f"  6. Perfect Solver:  0.00%")
    print(f"  7. Program Search: {accuracy:.2f}% ← CURRENT")

    if accuracy >= 5:
        print(f"\n🏆 WEEK 1 GOAL ACHIEVED!")
        print(f"  Target: 5-10%")
        print(f"  Achieved: {accuracy:.2f}%")
        print(f"  READY FOR SUBMISSION!")
    elif accuracy > 0:
        print(f"\n📍 Progress:")
        print(f"  Current: {accuracy:.2f}%")
        print(f"  Gap to goal: {5 - accuracy:.2f}%")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    results = evaluate_search()

    with open('search_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved to search_results.json")