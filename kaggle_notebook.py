"""
ARC Prize 2025 - Kaggle Notebook Submission

This script is designed to run on Kaggle as a notebook submission.
It must work offline (no internet access) and generate predictions for all test tasks.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Callable


# ============================================================================
# BASELINE SOLVER
# ============================================================================

class BaselineSolver:
    """Simple rule-based solver for ARC tasks."""

    def __init__(self):
        self.transformations = [
            self.identity,
            self.flip_horizontal,
            self.flip_vertical,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.transpose,
            self.scale_2x,
            self.scale_3x,
            self.tile_2x2,
            self.tile_3x3,
            self.tile_pattern_intelligent,
        ]

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """Solve a task and return predictions."""
        predictions = []

        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            # Find best transformations from training examples
            best_transforms = self.find_best_transformations(task['train'], max_attempts)

            attempts = []
            for transform_func in best_transforms[:max_attempts]:
                try:
                    output = transform_func(test_input)
                    attempts.append(output.tolist())
                except Exception:
                    attempts.append(test_input.tolist())

            # Pad with identity if not enough attempts
            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def find_best_transformations(self, train_examples: List[dict],
                                   n_transforms: int = 2) -> List[Callable]:
        """Find the best n transformations based on training examples."""
        transform_scores = {}

        for transform_func in self.transformations:
            score = 0
            for example in train_examples:
                input_grid = np.array(example['input'])
                expected_output = np.array(example['output'])

                try:
                    predicted_output = transform_func(input_grid)

                    if predicted_output.shape == expected_output.shape:
                        if np.array_equal(predicted_output, expected_output):
                            score += 100
                        else:
                            matching_cells = np.sum(predicted_output == expected_output)
                            total_cells = expected_output.size
                            score += (matching_cells / total_cells) * 50
                except:
                    pass

            transform_scores[transform_func] = score

        # Sort by score and return top n
        sorted_transforms = sorted(transform_scores.items(), key=lambda x: -x[1])
        return [func for func, score in sorted_transforms[:n_transforms]]

    # Transformation functions
    def identity(self, grid: np.ndarray) -> np.ndarray:
        return grid.copy()

    def flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    def flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    def rotate_90(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=-1)

    def rotate_180(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    def rotate_270(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=-3)

    def transpose(self, grid: np.ndarray) -> np.ndarray:
        return grid.T

    def scale_2x(self, grid: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

    def scale_3x(self, grid: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

    def tile_2x2(self, grid: np.ndarray) -> np.ndarray:
        return np.block([[grid, grid], [grid, grid]])

    def tile_3x3(self, grid: np.ndarray) -> np.ndarray:
        return np.block([[grid, grid, grid],
                        [grid, grid, grid],
                        [grid, grid, grid]])

    def tile_pattern_intelligent(self, grid: np.ndarray) -> np.ndarray:
        """Intelligently tile based on grid size."""
        h, w = grid.shape
        if h <= 3 and w <= 3:
            return self.tile_3x3(grid)
        elif h <= 5 and w <= 5:
            return self.tile_2x2(grid)
        return grid.copy()


# ============================================================================
# MAIN SUBMISSION CODE
# ============================================================================

def main():
    """Main function to generate submission."""
    print("ARC Prize 2025 - Generating Predictions")
    print("=" * 70)

    # Load test challenges
    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    # Initialize solver
    solver = BaselineSolver()

    # Generate predictions
    submission = {}

    print("\nGenerating predictions...")
    for i, (task_id, task) in enumerate(test_challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_challenges)} tasks...")

        # Get predictions
        predictions = solver.solve(task, max_attempts=2)

        # Format: [{"attempt_1": [[...]], "attempt_2": [[...]]}]
        task_submission = []
        for test_predictions in predictions:
            test_dict = {
                "attempt_1": test_predictions[0],
                "attempt_2": test_predictions[1]
            }
            task_submission.append(test_dict)

        submission[task_id] = task_submission

    print(f"\n✓ Generated predictions for {len(submission)} tasks")

    # Save submission
    output_path = Path('/kaggle/working/submission.json')
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"✓ Saved submission to: {output_path}")

    # Verify file size
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    print("\n" + "=" * 70)
    print("Submission generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
