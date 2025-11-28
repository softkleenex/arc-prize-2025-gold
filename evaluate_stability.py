"""
Evaluate Stability-Based Solver
Testing ARChitects' key innovation
"""

import json
import numpy as np
from stability_solver import StabilityBasedSolver, EnhancedAugmentationSolver
from direct_matcher import DirectMatcher
import time


def evaluate_stability():
    print("=" * 70)
    print("STABILITY-BASED SOLVER EVALUATION")
    print("ARChitects' Winning Approach (53.5%)")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    # Initialize solvers
    base_solver = DirectMatcher()
    stability_solver = StabilityBasedSolver(stability_threshold=0.6)

    correct = 0
    partial_matches = []
    perfect_tasks = []

    # Test first 40 tasks
    test_tasks = list(eval_data.items())[:40]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks...")
    print("Stability criterion: 60% augmentation agreement\n")
    print("Augmentations tested:")
    print("  • Identity, Rotations (90°, 180°, 270°)")
    print("  • Flips (H, V, Both)")
    print("  • Transpose\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"\n[{i+1}/{total}] Status:")
            if correct > 0:
                print(f"  ✓✓✓ {correct} CORRECT! ({correct/(i+1)*100:.1f}%)")
            else:
                print(f"  Searching for stable solutions...")
            print(f"  Time: {elapsed:.1f}s")

        try:
            # Use stability-based solver
            predictions = stability_solver.solve_with_stability(task, base_solver)
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
                else:
                    print(".", end='', flush=True)

        except Exception as e:
            print("E", end='', flush=True)

    elapsed = time.time() - start
    print("\n")

    print("=" * 70)
    print("STABILITY-BASED RESULTS")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n🎯 Performance:")
    print(f"  Tasks: {total}")
    print(f"  ✓ CORRECT: {correct} ({accuracy:.2f}%)")
    print(f"  ! Very close (>97%): {len([p for p in partial_matches if p[1] > 97])}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s per task)")

    if correct > 0:
        print(f"\n🎊 BREAKTHROUGH WITH STABILITY!")
        print(f"  {correct} tasks solved perfectly!")
        print(f"\n  Perfect solutions:")
        for task in perfect_tasks[:10]:
            print(f"    • {task}")

    if partial_matches:
        print(f"\n📊 Near perfect (>97%):")
        very_close = [(t, p) for t, p in partial_matches if p > 97]
        very_close.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in very_close[:5]:
            print(f"    {task_id}: {pct:.2f}%")

    print("\n📈 COMPLETE SOLVER COMPARISON:")
    print("  ┌────────────────────────────────────────┐")
    print("  │ Approach              │ Accuracy      │")
    print("  ├────────────────────────────────────────┤")
    print("  │ DSL V1                │   0.00%       │")
    print("  │ DSL V2                │   0.00%       │")
    print("  │ Direct Match          │   0.00%       │")
    print("  │ Ensemble              │   0.00%       │")
    print("  │ Competition Pipeline  │   0.00%       │")
    print(f"  │ Stability-Based       │ {accuracy:6.2f}%      │")
    print("  ├────────────────────────────────────────┤")
    print("  │ ARChitects (2024)     │  53.50% ⭐    │")
    print("  └────────────────────────────────────────┘")

    if accuracy >= 5:
        print(f"\n🏆 WEEK 1 GOAL ACHIEVED!")
        print(f"  Target: 5-10%")
        print(f"  Achieved: {accuracy:.2f}%")
        print(f"  READY FOR SUBMISSION!")
    elif accuracy > 0:
        print(f"\n🎯 FIRST CORRECT SOLUTIONS!")
        print(f"  Current: {accuracy:.2f}%")
        print(f"  Progress: {accuracy/5*100:.0f}% toward 5% goal")
        print(f"  Gap: {5-accuracy:.2f}%")
    else:
        print(f"\n💡 Stability criterion applied but no perfect matches yet")
        print(f"  Very close predictions suggest approach is promising")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    print("Testing ARChitects' stability-based approach...\n")

    results = evaluate_stability()

    with open('stability_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to stability_results.json")

    if results['accuracy'] > 0:
        print(f"\n🎉 STABILITY-BASED APPROACH WORKS!")
        print(f"Achieved {results['accuracy']:.2f}% accuracy")
    else:
        print(f"\n📊 Analysis: {results['partial']} partial matches")
        print("Stability filtering reduces noise but needs base solver improvement")