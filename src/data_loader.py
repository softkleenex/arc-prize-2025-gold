"""
ARC Prize 2025 - Data Loader Module

This module provides utilities for loading and parsing ARC-AGI data files.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class ARCDataLoader:
    """Loader for ARC-AGI dataset files."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the ARC data files
        """
        self.data_dir = Path(data_dir)

    def load_challenges(self, split: str = "training") -> Dict:
        """
        Load challenges from specified split.

        Args:
            split: One of 'training', 'evaluation', or 'test'

        Returns:
            Dictionary of challenges
        """
        file_path = self.data_dir / f"arc-agi_{split}_challenges.json"
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_solutions(self, split: str = "training") -> Dict:
        """
        Load solutions from specified split.

        Args:
            split: One of 'training' or 'evaluation'

        Returns:
            Dictionary of solutions
        """
        file_path = self.data_dir / f"arc-agi_{split}_solutions.json"
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_task(self, task_id: str, split: str = "training") -> Tuple[Dict, Dict]:
        """
        Load a specific task with its solution.

        Args:
            task_id: Task identifier
            split: One of 'training' or 'evaluation'

        Returns:
            Tuple of (challenge, solution)
        """
        challenges = self.load_challenges(split)
        solutions = self.load_solutions(split)
        return challenges[task_id], solutions[task_id]

    def get_task_ids(self, split: str = "training") -> List[str]:
        """
        Get all task IDs from specified split.

        Args:
            split: One of 'training', 'evaluation', or 'test'

        Returns:
            List of task IDs
        """
        challenges = self.load_challenges(split)
        return list(challenges.keys())

    def get_grid_shape(self, grid: List[List[int]]) -> Tuple[int, int]:
        """
        Get the shape of a grid.

        Args:
            grid: 2D list representing the grid

        Returns:
            Tuple of (height, width)
        """
        return len(grid), len(grid[0]) if grid else 0

    def get_unique_colors(self, grid: List[List[int]]) -> set:
        """
        Get unique colors in a grid.

        Args:
            grid: 2D list representing the grid

        Returns:
            Set of unique color values
        """
        colors = set()
        for row in grid:
            colors.update(row)
        return colors


def print_task_info(task_id: str, challenge: Dict, solution: Dict = None):
    """
    Print information about a task.

    Args:
        task_id: Task identifier
        challenge: Challenge data
        solution: Solution data (optional)
    """
    print(f"\n{'='*60}")
    print(f"Task ID: {task_id}")
    print(f"{'='*60}")

    # Training examples
    print(f"\nTraining Examples: {len(challenge['train'])}")
    for i, example in enumerate(challenge['train']):
        input_grid = example['input']
        output_grid = example['output']
        print(f"\n  Example {i+1}:")
        print(f"    Input shape:  {len(input_grid)}x{len(input_grid[0])}")
        print(f"    Output shape: {len(output_grid)}x{len(output_grid[0])}")

        loader = ARCDataLoader()
        input_colors = loader.get_unique_colors(input_grid)
        output_colors = loader.get_unique_colors(output_grid)
        print(f"    Input colors:  {sorted(input_colors)}")
        print(f"    Output colors: {sorted(output_colors)}")

    # Test examples
    print(f"\nTest Examples: {len(challenge['test'])}")
    for i, example in enumerate(challenge['test']):
        input_grid = example['input']
        print(f"\n  Test {i+1}:")
        print(f"    Input shape: {len(input_grid)}x{len(input_grid[0])}")

        if solution:
            output_grid = solution[i]
            print(f"    Output shape: {len(output_grid)}x{len(output_grid[0])}")


if __name__ == "__main__":
    # Example usage
    loader = ARCDataLoader()

    # Load training data
    print("Loading training data...")
    training_ids = loader.get_task_ids("training")
    print(f"Number of training tasks: {len(training_ids)}")

    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_ids = loader.get_task_ids("evaluation")
    print(f"Number of evaluation tasks: {len(eval_ids)}")

    # Show example task
    if training_ids:
        example_id = training_ids[0]
        challenge, solution = loader.load_task(example_id, "training")
        print_task_info(example_id, challenge, solution)
