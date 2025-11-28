"""
Evaluate Direct Matcher on evaluation set
"""

import json
import numpy as np
from direct_matcher import DirectMatcher
import time


def evaluate_direct():
    print("=" * 70)
    print("DIRECT MATCHER EVALUATION - First 60 tasks")
    print("Target: 100% pixel-perfect accuracy")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = DirectMatcher()

    correct = 0
    partial_matches = []
    perfect_tasks = []

    # Test first 60 tasks
    test_tasks = list(eval_data.items())[:60]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks with direct matching...")
    print("Strategies: Exact replication, Deterministic, Consensus, Template\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            if correct > 0:
                print(f"\n[{i+1}/{total}] ✓ {correct} correct so far! ({elapsed:.1f}s)")
            else:
                print(f"\n[{i+1}/{total}] Searching... ({elapsed:.1f}s)")

        try:
            predictions = solver.solve(task)
            solutions = eval_solutions[task_id]

            task_correct = False
            best_match = 0

            for pred_list, sol in zip(predictions, solutions):
                for attempt in pred_list:
                    try:
                        # Convert to arrays
                        pred_arr = np.array(attempt)
                        sol_arr = np.array(sol)

                        # Check exact match
                        if pred_arr.shape == sol_arr.shape and np.array_equal(pred_arr, sol_arr):
                            correct += 1
                            task_correct = True
                            perfect_tasks.append(task_id)
                            print(f"✓", end='')
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
                    print("~", end='')
                elif best_match > 0.8:
                    print(".", end='')
                else:
                    print("x", end='')

        except Exception as e:
            print("E", end='')

    elapsed = time.time() - start
    print("\n")

    print("=" * 70)
    print("DIRECT MATCHER RESULTS")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n🎯 PERFORMANCE:")
    print(f"  Total tasks: {total}")
    print(f"  ✓ CORRECT: {correct} ({accuracy:.1f}%)")
    print(f"  ~ Near miss (>95%): {len([p for p in partial_matches if p[1] > 95])}")
    print(f"  . Partial (>80%): {len(partial_matches)}")
    print(f"  Time: {elapsed:.1f}s")

    if correct > 0:
        print(f"\n🎉 BREAKTHROUGH! Achieved {correct} correct answers!")
        print(f"  Perfect tasks: {', '.join(t[:8] for t in perfect_tasks[:10])}")

    if partial_matches:
        print(f"\n📊 Closest attempts (>95%):")
        very_close = [(t, p) for t, p in partial_matches if p > 95]
        very_close.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in very_close[:5]:
            print(f"  {task_id[:12]}: {pct:.2f}%")

    print("\n📈 PROGRESS TRACKER:")
    print(f"  DSL V1:        0% (baseline)")
    print(f"  DSL V2:        0% (98.9% partial)")
    print(f"  DSL V3:        0% (95.0% partial)")
    print(f"  Hybrid:        0% (96.8% partial)")
    print(f"  Direct Match: {accuracy:.1f}% ← Current")

    if accuracy >= 5:
        print(f"\n🏆 ACHIEVEMENT UNLOCKED!")
        print(f"  Week 1 Goal (5-10%): ACHIEVED with {accuracy:.1f}%!")
    elif accuracy > 0:
        print(f"\n📍 PROGRESS TO GOAL:")
        print(f"  Current: {accuracy:.1f}%")
        print(f"  Week 1 target: 5.0%")
        print(f"  Gap: {5.0 - accuracy:.1f}%")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    results = evaluate_direct()

    with open('direct_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to direct_results.json")