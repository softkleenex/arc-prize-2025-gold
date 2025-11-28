"""
ARC Prize 2025 - Baseline Solver

This module implements simple rule-based transformations for ARC tasks.
"""

import numpy as np
from typing import List, Tuple, Callable
import copy


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
            self.fill_background,
            self.scale_2x,
            self.scale_3x,
            self.tile_2x2,
            self.tile_3x3,
            self.extract_pattern,
            self.color_swap_most_common,
        ]

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """
        Solve a task and return predictions.

        Args:
            task: Task dictionary with 'train' and 'test' keys
            max_attempts: Number of attempts (Pass@2 = 2 attempts)

        Returns:
            List of predictions for each test example
        """
        predictions = []

        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            # Try to find a matching transformation from training examples
            best_transforms = self.find_best_transformations(task['train'], max_attempts)

            attempts = []
            for transform_func in best_transforms[:max_attempts]:
                try:
                    output = transform_func(test_input)
                    attempts.append(output.tolist())
                except Exception as e:
                    # If transformation fails, use identity
                    attempts.append(test_input.tolist())

            # Pad with identity if not enough attempts
            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def find_best_transformations(self, train_examples: List[dict],
                                   n_transforms: int = 2) -> List[Callable]:
        """
        Find the best n transformations based on training examples.

        Args:
            train_examples: List of training examples
            n_transforms: Number of transformations to return

        Returns:
            List of transformation functions
        """
        transform_scores = {}

        for transform_func in self.transformations:
            score = 0
            for example in train_examples:
                input_grid = np.array(example['input'])
                expected_output = np.array(example['output'])

                try:
                    predicted_output = transform_func(input_grid)

                    # Check if transformation matches
                    if predicted_output.shape == expected_output.shape:
                        if np.array_equal(predicted_output, expected_output):
                            score += 100  # Perfect match
                        else:
                            # Partial match: count matching cells
                            matching_cells = np.sum(predicted_output == expected_output)
                            total_cells = expected_output.size
                            score += (matching_cells / total_cells) * 50
                except:
                    pass

            transform_scores[transform_func] = score

        # Sort by score and return top n
        sorted_transforms = sorted(transform_scores.items(), key=lambda x: -x[1])
        return [func for func, score in sorted_transforms[:n_transforms]]

    # ============ TRANSFORMATION FUNCTIONS ============

    def identity(self, grid: np.ndarray) -> np.ndarray:
        """Return the grid unchanged."""
        return grid.copy()

    def flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally."""
        return np.fliplr(grid)

    def flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically."""
        return np.flipud(grid)

    def rotate_90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise."""
        return np.rot90(grid, k=-1)

    def rotate_180(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 180 degrees."""
        return np.rot90(grid, k=2)

    def rotate_270(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 270 degrees clockwise."""
        return np.rot90(grid, k=-3)

    def transpose(self, grid: np.ndarray) -> np.ndarray:
        """Transpose the grid."""
        return grid.T

    def fill_background(self, grid: np.ndarray) -> np.ndarray:
        """Fill all cells with background color."""
        background = self.get_background_color(grid)
        return np.full_like(grid, background)

    def scale_2x(self, grid: np.ndarray) -> np.ndarray:
        """Scale grid 2x using nearest neighbor."""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

    def scale_3x(self, grid: np.ndarray) -> np.ndarray:
        """Scale grid 3x using nearest neighbor."""
        return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

    def tile_2x2(self, grid: np.ndarray) -> np.ndarray:
        """Tile the grid in a 2x2 pattern."""
        return np.block([[grid, grid],
                        [grid, grid]])

    def tile_3x3(self, grid: np.ndarray) -> np.ndarray:
        """Tile the grid in a 3x3 pattern."""
        return np.block([[grid, grid, grid],
                        [grid, grid, grid],
                        [grid, grid, grid]])

    def extract_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Extract non-background pattern."""
        background = self.get_background_color(grid)
        mask = grid != background

        if not mask.any():
            return grid.copy()

        # Find bounding box of pattern
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return grid.copy()

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Extract pattern
        pattern = grid[rmin:rmax+1, cmin:cmax+1]
        return pattern

    def color_swap_most_common(self, grid: np.ndarray) -> np.ndarray:
        """Swap the two most common colors."""
        unique, counts = np.unique(grid, return_counts=True)

        if len(unique) < 2:
            return grid.copy()

        # Get two most common colors
        sorted_idx = np.argsort(-counts)
        color1, color2 = unique[sorted_idx[0]], unique[sorted_idx[1]]

        # Swap colors
        result = grid.copy()
        result[grid == color1] = -1  # Temporary marker
        result[grid == color2] = color1
        result[result == -1] = color2

        return result

    def get_background_color(self, grid: np.ndarray) -> int:
        """Get the most common color (assumed to be background)."""
        unique, counts = np.unique(grid, return_counts=True)
        return unique[np.argmax(counts)]


class AdvancedSolver(BaselineSolver):
    """More advanced solver with additional transformations."""

    def __init__(self):
        super().__init__()
        # Add more sophisticated transformations
        self.transformations.extend([
            self.detect_and_tile_pattern,
            self.fill_symmetric,
            self.connect_objects,
            self.color_by_position,
        ])

    def detect_and_tile_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Detect pattern and tile intelligently."""
        h, w = grid.shape

        # Try to detect if input is small pattern
        if h <= 3 and w <= 3:
            # Tile 3x3
            result = self.tile_3x3(grid)
            return result
        elif h <= 5 and w <= 5:
            # Tile 2x2
            result = self.tile_2x2(grid)
            return result

        return grid.copy()

    def fill_symmetric(self, grid: np.ndarray) -> np.ndarray:
        """Fill grid to make it symmetric."""
        h, w = grid.shape

        if h != w:
            return grid.copy()

        # Make it symmetric along diagonal
        result = grid.copy()
        for i in range(h):
            for j in range(i+1, w):
                if result[i, j] == 0:
                    result[i, j] = result[j, i]
                elif result[j, i] == 0:
                    result[j, i] = result[i, j]

        return result

    def connect_objects(self, grid: np.ndarray) -> np.ndarray:
        """Connect non-background objects."""
        background = self.get_background_color(grid)
        result = grid.copy()

        # Find all non-background cells
        objects = np.argwhere(grid != background)

        if len(objects) < 2:
            return result

        # Connect first two objects with a line
        if len(objects) >= 2:
            y1, x1 = objects[0]
            y2, x2 = objects[1]

            # Draw line
            for t in np.linspace(0, 1, max(abs(y2-y1), abs(x2-x1))+1):
                y = int(y1 + t * (y2 - y1))
                x = int(x1 + t * (x2 - x1))
                if result[y, x] == background:
                    result[y, x] = grid[y1, x1]

        return result

    def color_by_position(self, grid: np.ndarray) -> np.ndarray:
        """Color cells based on their position."""
        result = grid.copy()
        h, w = result.shape

        # Color in a checkerboard pattern
        for i in range(h):
            for j in range(w):
                result[i, j] = (i + j) % 10

        return result


def test_solver():
    """Test the solver on training data."""
    from data_loader import ARCDataLoader

    loader = ARCDataLoader()
    solver = BaselineSolver()

    # Load a few training tasks
    train_ids = loader.get_task_ids("training")[:10]

    print("="*70)
    print("TESTING BASELINE SOLVER")
    print("="*70)

    correct = 0
    total = 0

    for task_id in train_ids:
        challenge, solution = loader.load_task(task_id, "training")

        predictions = solver.solve(challenge, max_attempts=1)

        for i, (pred_attempts, expected) in enumerate(zip(predictions, solution)):
            pred = pred_attempts[0]  # First attempt
            is_correct = pred == expected

            total += 1
            if is_correct:
                correct += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} Task {task_id} - Test {i+1}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_solver()
