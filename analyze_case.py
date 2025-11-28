"""
Analyze specific case that's getting 98.93% match
Understand what's preventing 100% accuracy
"""

import json
import numpy as np
from direct_matcher import DirectMatcher


def analyze_case():
    print("=" * 70)
    print("CASE ANALYSIS: Task 135a2760 (98.93% match)")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    # Get specific task
    task_id = "135a2760"
    task = eval_data[task_id]
    solution = eval_solutions[task_id]

    print(f"\n📋 Task Analysis:")
    print(f"  Training examples: {len(task['train'])}")
    print(f"  Test examples: {len(task['test'])}")

    # Analyze training examples
    print(f"\n🔍 Training Pattern Analysis:")
    for idx, example in enumerate(task['train']):
        in_grid = example['input']
        out_grid = example['output']

        in_h, in_w = len(in_grid), len(in_grid[0])
        out_h, out_w = len(out_grid), len(out_grid[0])

        in_colors = set(v for row in in_grid for v in row)
        out_colors = set(v for row in out_grid for v in row)

        print(f"\n  Example {idx+1}:")
        print(f"    Input:  {in_h}x{in_w}, colors: {sorted(in_colors)}")
        print(f"    Output: {out_h}x{out_w}, colors: {sorted(out_colors)}")

        # Check transformation type
        if in_h == out_h and in_w == out_w:
            # Check if it's color mapping
            color_changes = 0
            for i in range(in_h):
                for j in range(in_w):
                    if in_grid[i][j] != out_grid[i][j]:
                        color_changes += 1

            if color_changes == 0:
                print(f"    Transform: Identity (no change)")
            else:
                print(f"    Transform: Color mapping ({color_changes} pixels changed)")

                # Show color mapping
                color_map = {}
                for i in range(in_h):
                    for j in range(in_w):
                        if in_grid[i][j] != 0:
                            if in_grid[i][j] not in color_map:
                                color_map[in_grid[i][j]] = out_grid[i][j]

                print(f"    Color map: {color_map}")
        else:
            scale_h = out_h / in_h
            scale_w = out_w / in_w
            print(f"    Transform: Resize {scale_h:.1f}x{scale_w:.1f}")

    # Run solver and compare
    solver = DirectMatcher()
    predictions = solver.solve(task)

    print(f"\n🎯 Prediction vs Solution:")
    for test_idx, (pred_list, sol) in enumerate(zip(predictions, solution)):
        print(f"\n  Test {test_idx+1}:")

        for attempt_idx, attempt in enumerate(pred_list):
            pred_arr = np.array(attempt)
            sol_arr = np.array(sol)

            if pred_arr.shape != sol_arr.shape:
                print(f"    Attempt {attempt_idx+1}: Wrong shape! {pred_arr.shape} vs {sol_arr.shape}")
                continue

            # Calculate match
            matches = np.sum(pred_arr == sol_arr)
            total = pred_arr.size
            accuracy = matches / total * 100

            print(f"    Attempt {attempt_idx+1}: {accuracy:.2f}% match ({matches}/{total} pixels)")

            if accuracy < 100:
                # Show differences
                diff_positions = []
                for i in range(pred_arr.shape[0]):
                    for j in range(pred_arr.shape[1]):
                        if pred_arr[i, j] != sol_arr[i, j]:
                            diff_positions.append((i, j, pred_arr[i, j], sol_arr[i, j]))

                print(f"    Differences (first 5):")
                for i, j, pred_val, sol_val in diff_positions[:5]:
                    print(f"      Position ({i},{j}): predicted {pred_val}, actual {sol_val}")

                # Analyze pattern in differences
                wrong_colors = {}
                for _, _, pred_val, sol_val in diff_positions:
                    key = (pred_val, sol_val)
                    wrong_colors[key] = wrong_colors.get(key, 0) + 1

                print(f"    Color mismatch pattern:")
                for (pred_c, sol_c), count in sorted(wrong_colors.items(), key=lambda x: -x[1])[:3]:
                    print(f"      {pred_c} → {sol_c}: {count} times")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")

    # Try to identify the exact issue
    print("\n🔑 The 1.07% error suggests:")
    print("  • Likely a systematic color mapping issue")
    print("  • Or edge case handling problem")
    print("  • Need exact pixel-by-pixel rule learning")

    return task_id


def deep_analyze():
    """Deep analysis of multiple high-scoring tasks"""
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS: All >95% matches")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)

    # High scoring tasks from our results
    high_scoring = ["135a2760", "3e6067c3", "409aa875", "4c3d4a41"]

    patterns = {
        'same_size': 0,
        'scale': 0,
        'color_only': 0,
        'complex': 0
    }

    for task_id in high_scoring:
        if task_id in eval_data:
            task = eval_data[task_id]
            print(f"\n📌 {task_id}:")

            for example in task['train']:
                in_h, in_w = len(example['input']), len(example['input'][0])
                out_h, out_w = len(example['output']), len(example['output'][0])

                if in_h == out_h and in_w == out_w:
                    patterns['same_size'] += 1

                    # Check if only colors change
                    shape_same = True
                    for i in range(in_h):
                        for j in range(in_w):
                            if (example['input'][i][j] == 0) != (example['output'][i][j] == 0):
                                shape_same = False
                                break
                        if not shape_same:
                            break

                    if shape_same:
                        patterns['color_only'] += 1
                        print(f"  → Color-only transformation detected")

                elif out_h == 2*in_h and out_w == 2*in_w:
                    patterns['scale'] += 1
                else:
                    patterns['complex'] += 1

    print(f"\n📊 Pattern Distribution in high-scoring tasks:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}")

    print("\n💡 RECOMMENDATION:")
    print("  Focus on exact color mapping rules")
    print("  Need pixel-perfect transformation learning")


if __name__ == "__main__":
    task_id = analyze_case()
    deep_analyze()

    print("\n✓ Analysis complete")