"""
Evaluate Hybrid Solver on evaluation set
"""

import json
import numpy as np
from hybrid_solver import HybridSolver
import time


def evaluate_hybrid():
    print("=" * 70)
    print("Hybrid Solver Evaluation - First 50 tasks")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    solver = HybridSolver()

    correct = 0
    partial_matches = []
    perfect_tasks = []
    strategy_success = {'direct': 0, 'transform': 0, 'interpolation': 0, 'rule': 0}

    # Test first 50 tasks for comprehensive evaluation
    test_tasks = list(eval_data.items())[:50]
    total = len(test_tasks)

    print(f"\nEvaluating {total} tasks...")
    print("Strategies:")
    print("• Direct pattern matching")
    print("• Transform-based analysis")
    print("• Example interpolation")
    print("• Rule extraction\n")

    start = time.time()

    for i, (task_id, task) in enumerate(test_tasks):
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i+1}/{total}")

        print(f"[{i+1}] {task_id[:8]}...: ", end='')

        try:
            predictions = solver.solve(task)
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
                            perfect_tasks.append(task_id)
                            print("✓", end=' ')
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
                if best_match > 0.9:
                    partial_matches.append((task_id, best_match * 100))
                    print(f"~{int(best_match*100)}%", end=' ')
                elif best_match > 0.8:
                    partial_matches.append((task_id, best_match * 100))
                    print(".", end=' ')
                else:
                    print("x", end=' ')

        except Exception as e:
            print("E", end=' ')

    print()  # New line after progress

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("HYBRID SOLVER RESULTS")
    print("=" * 70)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n📊 Performance:")
    print(f"  Total tasks: {total}")
    print(f"  Correct: {correct} ({accuracy:.1f}%)")
    print(f"  Partial (>80%): {len(partial_matches)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.1f}s per task)")

    if correct > 0:
        print(f"\n✓ SUCCESS! Achieved {accuracy:.1f}% accuracy")
        print(f"  Perfect matches: {', '.join(t[:8] for t in perfect_tasks[:5])}")

    if partial_matches:
        print(f"\n📈 Near misses (top 5):")
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        for task_id, pct in partial_matches[:5]:
            print(f"  {task_id[:12]}: {pct:.1f}%")

    # Progress summary
    print("\n📋 Progress Summary:")
    print(f"  DSL V1: 0% correct, 0 partial")
    print(f"  DSL V2: 0% correct, 23 partial (best 98.9%)")
    print(f"  DSL V3: 0% correct, 12 partial (best 95.0%)")
    print(f"  Hybrid: {accuracy:.1f}% correct, {len(partial_matches)} partial")

    if accuracy > 0:
        print(f"\n🎯 Goal Progress:")
        print(f"  Week 1 target (5-10%): {'ACHIEVED!' if accuracy >= 5 else f'{accuracy:.1f}/5.0%'}")
        print(f"  Top 30-40 target (25-35%): {accuracy:.1f}/25.0%")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'partial': len(partial_matches),
        'perfect_tasks': perfect_tasks,
        'time': elapsed
    }


if __name__ == "__main__":
    results = evaluate_hybrid()

    with open('hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to hybrid_results.json")