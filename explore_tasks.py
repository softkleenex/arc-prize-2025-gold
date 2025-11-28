"""
Explore and visualize specific ARC tasks to understand patterns.
"""

import json
import numpy as np
from pathlib import Path


def load_task(task_id: str, split: str = "training"):
    """Load a specific task."""
    data_dir = Path("data")

    with open(data_dir / f"arc-agi_{split}_challenges.json", 'r') as f:
        challenges = json.load(f)

    if split != "test":
        with open(data_dir / f"arc-agi_{split}_solutions.json", 'r') as f:
            solutions = json.load(f)
    else:
        solutions = {}

    return challenges[task_id], solutions.get(task_id, [])


def print_grid(grid, label="Grid"):
    """Print a grid in a readable format."""
    print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(cell) for cell in row))


def analyze_transformation(input_grid, output_grid):
    """Analyze the transformation between input and output."""
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)

    print("\n" + "="*70)
    print("TRANSFORMATION ANALYSIS")
    print("="*70)

    # Size analysis
    print(f"\nSize change: {input_arr.shape} → {output_arr.shape}")

    if input_arr.shape == output_arr.shape:
        print("  Type: In-place transformation (same size)")

        # Check for simple transformations
        if np.array_equal(input_arr, output_arr):
            print("  ⚠ Input and output are identical!")
        elif np.array_equal(input_arr, np.flipud(output_arr)):
            print("  ✓ Vertical flip detected")
        elif np.array_equal(input_arr, np.fliplr(output_arr)):
            print("  ✓ Horizontal flip detected")
        elif np.array_equal(input_arr, np.rot90(output_arr, 1)):
            print("  ✓ 90° rotation detected")
        elif np.array_equal(input_arr, np.rot90(output_arr, 2)):
            print("  ✓ 180° rotation detected")
        elif np.array_equal(input_arr, np.rot90(output_arr, 3)):
            print("  ✓ 270° rotation detected")
        else:
            print("  ✓ Complex in-place transformation")

            # Check cell-by-cell changes
            changed_cells = np.sum(input_arr != output_arr)
            total_cells = input_arr.size
            print(f"  Changed cells: {changed_cells}/{total_cells} ({changed_cells/total_cells*100:.1f}%)")

    elif output_arr.shape[0] > input_arr.shape[0] or output_arr.shape[1] > input_arr.shape[1]:
        print("  Type: Expansion/Scaling up")

        # Check if it's simple tiling
        h_ratio = output_arr.shape[0] / input_arr.shape[0]
        w_ratio = output_arr.shape[1] / input_arr.shape[1]
        print(f"  Scale factors: height={h_ratio:.2f}x, width={w_ratio:.2f}x")

        if h_ratio.is_integer() and w_ratio.is_integer():
            # Check if it's exact tiling
            h_ratio, w_ratio = int(h_ratio), int(w_ratio)
            is_tiled = True

            for i in range(h_ratio):
                for j in range(w_ratio):
                    tile = output_arr[i*input_arr.shape[0]:(i+1)*input_arr.shape[0],
                                     j*input_arr.shape[1]:(j+1)*input_arr.shape[1]]
                    if not np.array_equal(tile, input_arr):
                        is_tiled = False
                        break

            if is_tiled:
                print(f"  ✓ Simple {h_ratio}×{w_ratio} tiling detected")
            else:
                print(f"  Complex {h_ratio}×{w_ratio} scaling transformation")

    else:
        print("  Type: Contraction/Scaling down")

    # Color analysis
    input_colors = set(input_arr.flatten())
    output_colors = set(output_arr.flatten())

    print(f"\nColor analysis:")
    print(f"  Input colors:  {sorted(input_colors)} ({len(input_colors)} unique)")
    print(f"  Output colors: {sorted(output_colors)} ({len(output_colors)} unique)")

    if input_colors == output_colors:
        print("  ✓ Same colors (rearrangement only)")
    elif output_colors.issubset(input_colors):
        removed = input_colors - output_colors
        print(f"  ✓ Color filtering (removed: {sorted(removed)})")
    elif input_colors.issubset(output_colors):
        added = output_colors - input_colors
        print(f"  ✓ Color addition (added: {sorted(added)})")
    else:
        removed = input_colors - output_colors
        added = output_colors - input_colors
        print(f"  ✓ Color replacement (removed: {sorted(removed)}, added: {sorted(added)})")


def explore_task(task_id: str, split: str = "training"):
    """Explore a specific task in detail."""
    print("\n" + "="*70)
    print(f"TASK: {task_id}")
    print("="*70)

    challenge, solution = load_task(task_id, split)

    print(f"\nTraining examples: {len(challenge['train'])}")
    print(f"Test examples: {len(challenge['test'])}")

    # Analyze each training example
    for i, example in enumerate(challenge['train']):
        print(f"\n{'─'*70}")
        print(f"TRAINING EXAMPLE {i+1}")
        print('─'*70)

        print_grid(example['input'], "Input")
        print_grid(example['output'], "Output")

        analyze_transformation(example['input'], example['output'])

    # Show test examples
    print(f"\n{'─'*70}")
    print("TEST EXAMPLES")
    print('─'*70)

    for i, example in enumerate(challenge['test']):
        print(f"\nTest {i+1}:")
        print_grid(example['input'], "Input")

        if solution and i < len(solution):
            print_grid(solution[i], "Expected Output")


def explore_multiple_tasks(task_ids: list):
    """Explore multiple tasks."""
    for task_id in task_ids:
        explore_task(task_id)
        print("\n" + "="*70)
        input("\nPress Enter to continue to next task...")


if __name__ == "__main__":
    # Load some interesting tasks
    data_dir = Path("data")
    with open(data_dir / "arc-agi_training_challenges.json", 'r') as f:
        challenges = json.load(f)

    # Select diverse tasks
    task_ids = list(challenges.keys())[:5]

    print("="*70)
    print("ARC PRIZE 2025 - TASK EXPLORATION")
    print("="*70)
    print(f"\nExploring {len(task_ids)} sample tasks...")
    print("This will help understand the patterns and transformations.")
    print("\nNote: Run with matplotlib for visual representations")

    for task_id in task_ids:
        explore_task(task_id)
        print("\n")
