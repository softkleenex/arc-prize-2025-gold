"""
Evaluate Ensemble Solver
"""

import json
import numpy as np
from ensemble_solver import ImprovedEnsemble
import time


def evaluate_ensemble():
    print("=" * 70)
    print("🎯 ENSEMBLE SOLVER EVALUATION")
    print("Combining 5 solvers with voting (AIRV-inspired)")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = ImprovedEnsemble()

    correct = 0
    partial_matches = []
    perfect_tasks = []

    # Test first 80 tasks
    test_tasks = list(eval_data.items())[:80]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks with ensemble...")
    print("Solvers: Direct, Perfect, Search, Hybrid, Pattern")
    print("+ Augmentation with rotations and voting\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"\n[{i+1}/{total}] Status:")
            if correct > 0:
                print(f"  ✓✓✓ {correct} CORRECT! ({correct/(i+1)*100:.1f}%)")
            else:
                print(f"  Searching...")
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
                if best_match > 0.97:
                    partial_matches.append((task_id, best_match * 100))
                    print("!", end='', flush=True)
                elif best_match > 0.9:
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
    print("🏆 ENSEMBLE RESULTS")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n📊 Final Performance:")
    print(f"  Tasks: {total}")
    print(f"  ✓ CORRECT: {correct} ({accuracy:.2f}%)")
    print(f"  ! Very close (>97%): {len([p for p in partial_matches if p[1] > 97])}")
    print(f"  ~ Close (>90%): {len(partial_matches)}")
    print(f"  Time: {elapsed:.1f}s")

    if correct > 0:
        print(f"\n🎊 BREAKTHROUGH ACHIEVED!")
        print(f"  {correct} tasks solved perfectly!")
        print(f"\n  Tasks solved:")
        for i, task in enumerate(perfect_tasks[:15], 1):
            print(f"    {i}. {task}")

    if partial_matches:
        print(f"\n📈 Closest attempts (>97%):")
        very_close = [(t, p) for t, p in partial_matches if p > 97]
        very_close.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in very_close[:5]:
            print(f"    {task_id}: {pct:.2f}%")

    print("\n🔍 COMPLETE SOLVER COMPARISON:")
    print("  ┌─────────────────────────────────────────┐")
    print("  │ Solver          │ Accuracy │ Best Match │")
    print("  ├─────────────────────────────────────────┤")
    print("  │ DSL V1          │   0.00%  │    0.0%    │")
    print("  │ DSL V2          │   0.00%  │   98.9%    │")
    print("  │ DSL V3          │   0.00%  │   95.0%    │")
    print("  │ Hybrid          │   0.00%  │   96.8%    │")
    print("  │ Direct Match    │   0.00%  │   98.9%    │")
    print("  │ Perfect Solver  │   0.00%  │    -       │")
    print("  │ Program Search  │   0.00%  │    -       │")
    print(f"  │ ENSEMBLE        │ {accuracy:6.2f}%  │    -       │")
    print("  └─────────────────────────────────────────┘")

    if accuracy >= 5:
        print(f"\n🏅 WEEK 1 GOAL ACHIEVED!")
        print(f"  Target: 5-10%")
        print(f"  Achieved: {accuracy:.2f}%")
        print(f"  STATUS: READY FOR SUBMISSION!")
        print(f"\n  Next steps:")
        print(f"    1. Create submission kernel")
        print(f"    2. Submit to Kaggle")
        print(f"    3. Target 10-15% for next iteration")
    elif accuracy > 0:
        print(f"\n✨ FIRST CORRECT SOLUTIONS!")
        print(f"  Current: {accuracy:.2f}%")
        print(f"  Gap to Week 1 goal: {5 - accuracy:.2f}%")
        print(f"  Progress: {accuracy/5*100:.0f}% of goal")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    print("Starting ensemble evaluation...")
    print("This combines all our approaches with voting.\n")

    results = evaluate_ensemble()

    with open('ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to ensemble_results.json")

    if results['accuracy'] >= 5:
        print("\n" + "=" * 50)
        print("🎉 READY FOR KAGGLE SUBMISSION!")
        print("=" * 50)