"""
ARC Prize 2025 - Quick Data Exploration Script

Run this script to get a quick overview of the dataset.
"""

import json
from pathlib import Path
from collections import Counter


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_dataset(split="training"):
    """
    Analyze a dataset split and print statistics.

    Args:
        split: 'training' or 'evaluation'
    """
    print(f"\n{'='*70}")
    print(f" {split.upper()} DATASET ANALYSIS")
    print(f"{'='*70}")

    # Load data
    data_dir = Path("data")
    challenges = load_json(data_dir / f"arc-agi_{split}_challenges.json")
    solutions = load_json(data_dir / f"arc-agi_{split}_solutions.json")

    print(f"\nNumber of tasks: {len(challenges)}")

    # Statistics
    n_train_examples = []
    n_test_examples = []
    input_shapes = []
    output_shapes = []
    input_colors_count = []
    output_colors_count = []

    for task_id, task in challenges.items():
        n_train_examples.append(len(task['train']))
        n_test_examples.append(len(task['test']))

        # Analyze training examples
        for example in task['train']:
            input_grid = example['input']
            output_grid = example['output']

            input_shapes.append((len(input_grid), len(input_grid[0]) if input_grid else 0))
            output_shapes.append((len(output_grid), len(output_grid[0]) if output_grid else 0))

            # Count unique colors
            input_colors = set()
            output_colors = set()
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)

            input_colors_count.append(len(input_colors))
            output_colors_count.append(len(output_colors))

    # Print statistics
    print(f"\n{'─'*70}")
    print("TASK STRUCTURE:")
    print(f"  Training examples per task: {min(n_train_examples)}-{max(n_train_examples)}")
    print(f"  Test examples per task: {min(n_test_examples)}-{max(n_test_examples)}")

    print(f"\n{'─'*70}")
    print("GRID SIZES:")
    print(f"  Input shapes: {min(input_shapes)} to {max(input_shapes)}")
    print(f"  Output shapes: {min(output_shapes)} to {max(output_shapes)}")

    # Most common sizes
    input_size_counter = Counter(input_shapes)
    output_size_counter = Counter(output_shapes)
    print(f"\n  Most common input sizes:")
    for size, count in input_size_counter.most_common(5):
        print(f"    {size[0]}×{size[1]}: {count} times")
    print(f"\n  Most common output sizes:")
    for size, count in output_size_counter.most_common(5):
        print(f"    {size[0]}×{size[1]}: {count} times")

    print(f"\n{'─'*70}")
    print("COLOR USAGE:")
    print(f"  Unique colors in inputs: {min(input_colors_count)}-{max(input_colors_count)}")
    print(f"  Unique colors in outputs: {min(output_colors_count)}-{max(output_colors_count)}")
    print(f"  Average colors per input: {sum(input_colors_count)/len(input_colors_count):.1f}")
    print(f"  Average colors per output: {sum(output_colors_count)/len(output_colors_count):.1f}")

    # Color distribution
    print(f"\n  Color usage frequency (0-9):")
    all_input_colors = []
    for task_id, task in challenges.items():
        for example in task['train']:
            for row in example['input']:
                all_input_colors.extend(row)

    color_counter = Counter(all_input_colors)
    for color in range(10):
        count = color_counter.get(color, 0)
        percentage = (count / len(all_input_colors) * 100) if all_input_colors else 0
        bar = '█' * int(percentage / 2)
        print(f"    Color {color}: {bar} {percentage:.1f}% ({count:,} cells)")


def show_sample_tasks(n=3):
    """Show a few sample task IDs."""
    print(f"\n{'='*70}")
    print(f" SAMPLE TASKS")
    print(f"{'='*70}\n")

    data_dir = Path("data")
    challenges = load_json(data_dir / "arc-agi_training_challenges.json")

    task_ids = list(challenges.keys())[:n]

    print(f"Showing {n} sample tasks:\n")
    for i, task_id in enumerate(task_ids, 1):
        task = challenges[task_id]
        print(f"{i}. Task ID: {task_id}")
        print(f"   Training examples: {len(task['train'])}")
        print(f"   Test examples: {len(task['test'])}")

        # First example dimensions
        first_train = task['train'][0]
        input_h = len(first_train['input'])
        input_w = len(first_train['input'][0]) if first_train['input'] else 0
        output_h = len(first_train['output'])
        output_w = len(first_train['output'][0]) if first_train['output'] else 0

        print(f"   First example: {input_h}×{input_w} → {output_h}×{output_w}\n")


def main():
    """Main exploration function."""
    print("\n" + "="*70)
    print(" ARC PRIZE 2025 - DATASET OVERVIEW")
    print("="*70)

    # Check if data exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("\n❌ Error: data/ directory not found!")
        print("Please run: python3 download_data.py")
        return

    # Analyze training set
    analyze_dataset("training")

    # Analyze evaluation set
    analyze_dataset("evaluation")

    # Show sample tasks
    show_sample_tasks(5)

    print(f"\n{'='*70}")
    print(" NEXT STEPS")
    print(f"{'='*70}\n")
    print("1. Explore individual tasks:")
    print("   python3 src/data_loader.py")
    print()
    print("2. Visualize tasks (requires matplotlib):")
    print("   python3 src/visualizer.py")
    print()
    print("3. Start with Jupyter notebooks:")
    print("   jupyter notebook notebooks/")
    print()
    print("4. Read the competition details:")
    print("   https://www.kaggle.com/competitions/arc-prize-2025")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
