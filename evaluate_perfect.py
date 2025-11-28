"""
Evaluate Perfect Solver - Context-aware approach
"""

import json
import numpy as np
from perfect_solver import PerfectSolver
import time


def evaluate_perfect():
    print("=" * 70)
    print("🎯 PERFECT SOLVER EVALUATION")
    print("Context-aware color learning for 100% accuracy")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = PerfectSolver()

    correct = 0
    partial_matches = []
    perfect_tasks = []
    near_perfect = []  # 99%+ matches

    # Test all 120 tasks for comprehensive evaluation
    test_tasks = list(eval_data.items())[:120]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks...")
    print("Methods: Context-aware, Pattern-exact, Neighbor-based, Object-aware\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"\n[{i+1}/{total}] Progress check:")
            if correct > 0:
                print(f"  ✓ {correct} CORRECT so far! ({correct/i*100:.1f}% accuracy)")
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
                if best_match >= 0.99:
                    near_perfect.append((task_id, best_match * 100))
                    print("!", end='', flush=True)  # Very close!
                elif best_match > 0.95:
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
    print("🏁 FINAL RESULTS - PERFECT SOLVER")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Total evaluated: {total} tasks")
    print(f"  ✓ CORRECT: {correct} ({accuracy:.2f}%)")
    print(f"  ! Near perfect (≥99%): {len(near_perfect)}")
    print(f"  ~ Close (>95%): {len(partial_matches)}")
    print(f"  Execution time: {elapsed:.1f}s ({elapsed/total:.2f}s per task)")

    if correct > 0:
        print(f"\n🎊 SUCCESS! ACHIEVED {correct} CORRECT SOLUTIONS!")
        print(f"\n✓ Perfect solutions on:")
        for task in perfect_tasks[:10]:
            print(f"    • {task}")

    if near_perfect:
        print(f"\n⚠️ Almost perfect (≥99% accuracy):")
        near_perfect.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in near_perfect[:5]:
            print(f"    {task_id}: {pct:.2f}%")

    print("\n📈 COMPLETE PROGRESS SUMMARY:")
    print(f"  1. DSL V1:        0.00%")
    print(f"  2. DSL V2:        0.00% (best partial: 98.9%)")
    print(f"  3. DSL V3:        0.00% (best partial: 95.0%)")
    print(f"  4. Hybrid:        0.00% (best partial: 96.8%)")
    print(f"  5. Direct Match:  0.00% (best partial: 98.9%)")
    print(f"  6. Perfect Solver: {accuracy:.2f}% ← CURRENT")

    if accuracy >= 5:
        print(f"\n🏆 MILESTONE ACHIEVED!")
        print(f"  ✓ Week 1 Goal (5-10%): COMPLETED with {accuracy:.2f}%!")
        print(f"  Next target: 10-15% for strong position")
    elif accuracy > 0:
        print(f"\n🎯 BREAKTHROUGH!")
        print(f"  First correct solutions achieved: {accuracy:.2f}%")
        print(f"  Distance to Week 1 goal: {5.0 - accuracy:.2f}%")
    else:
        print(f"\n💡 ANALYSIS:")
        if near_perfect:
            print(f"  Very close with {len(near_perfect)} tasks at ≥99%")
            print(f"  Need final pixel-perfect adjustments")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'near_perfect': len(near_perfect),
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    results = evaluate_perfect()

    with open('perfect_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to perfect_results.json")

    if results['accuracy'] > 0:
        print(f"\n🎉 READY FOR SUBMISSION with {results['accuracy']:.2f}% accuracy!")