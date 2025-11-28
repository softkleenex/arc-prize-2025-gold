"""
ARC Prize 2025 - Version 7: DSL-Based Program Search

핵심 아이디어:
- Training examples에서 "프로그램" 합성
- Domain-Specific Language (DSL) 정의
- Shortest program 선호 (Occam's Razor)
- Exhaustive search within time limit
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Tuple, Optional
from itertools import product, combinations


class DSLOperations:
    """Domain-Specific Language Operations"""

    @staticmethod
    def identity(grid):
        return grid.copy()

    # Geometric transforms
    @staticmethod
    def rot90(grid):
        return np.rot90(grid, k=-1)

    @staticmethod
    def rot180(grid):
        return np.rot90(grid, k=2)

    @staticmethod
    def rot270(grid):
        return np.rot90(grid, k=-3)

    @staticmethod
    def flip_h(grid):
        return np.fliplr(grid)

    @staticmethod
    def flip_v(grid):
        return np.flipud(grid)

    @staticmethod
    def transpose(grid):
        return grid.T

    # Size transforms
    @staticmethod
    def scale_2x(grid):
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

    @staticmethod
    def scale_3x(grid):
        return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

    @staticmethod
    def downscale_2x(grid):
        if grid.shape[0] % 2 == 0 and grid.shape[1] % 2 == 0:
            return grid[::2, ::2]
        return grid.copy()

    # Grid operations
    @staticmethod
    def crop_nonzero(grid):
        """Crop to non-zero content"""
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return grid.copy()
        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)
        return grid[min_row:max_row+1, min_col:max_col+1]

    @staticmethod
    def pad_to_square(grid):
        """Pad to square shape"""
        h, w = grid.shape
        if h == w:
            return grid.copy()
        size = max(h, w)
        result = np.zeros((size, size), dtype=grid.dtype)
        result[:h, :w] = grid
        return result

    # Color operations
    @staticmethod
    def invert_colors(grid):
        """Invert colors (0-9)"""
        return 9 - grid

    @staticmethod
    def swap_colors_01(grid):
        """Swap colors 0 and 1"""
        result = grid.copy()
        result[grid == 0] = -1
        result[grid == 1] = 0
        result[result == -1] = 1
        return result

    @staticmethod
    def map_to_most_common(grid):
        """Map all non-zero to most common non-zero color"""
        non_zero = grid[grid != 0]
        if len(non_zero) == 0:
            return grid.copy()
        most_common = np.bincount(non_zero).argmax()
        result = grid.copy()
        result[grid != 0] = most_common
        return result

    # Pattern operations
    @staticmethod
    def fill_diagonal(grid):
        """Fill main diagonal"""
        result = grid.copy()
        non_zero = grid[grid != 0]
        if len(non_zero) > 0:
            fill_color = np.bincount(non_zero).argmax()
            for i in range(min(grid.shape)):
                if result[i, i] == 0:
                    result[i, i] = fill_color
        return result

    @staticmethod
    def fill_border(grid):
        """Fill border with most common color"""
        result = grid.copy()
        non_zero = grid[grid != 0]
        if len(non_zero) > 0:
            fill_color = np.bincount(non_zero).argmax()
            result[0, :] = fill_color
            result[-1, :] = fill_color
            result[:, 0] = fill_color
            result[:, -1] = fill_color
        return result


class ProgramSynthesizer:
    """Synthesize programs from training examples"""

    def __init__(self):
        self.ops = DSLOperations()

        # Define atomic operations
        self.atomic_ops = {
            'identity': self.ops.identity,
            'rot90': self.ops.rot90,
            'rot180': self.ops.rot180,
            'rot270': self.ops.rot270,
            'flip_h': self.ops.flip_h,
            'flip_v': self.ops.flip_v,
            'transpose': self.ops.transpose,
            'scale_2x': self.ops.scale_2x,
            'scale_3x': self.ops.scale_3x,
            'downscale_2x': self.ops.downscale_2x,
            'crop_nonzero': self.ops.crop_nonzero,
            'pad_to_square': self.ops.pad_to_square,
            'invert_colors': self.ops.invert_colors,
            'swap_colors_01': self.ops.swap_colors_01,
            'map_to_most_common': self.ops.map_to_most_common,
            'fill_diagonal': self.ops.fill_diagonal,
            'fill_border': self.ops.fill_border,
        }

    def synthesize(self, train_examples: List[dict], max_length: int = 3) -> List[List[str]]:
        """Synthesize programs that solve training examples"""
        successful_programs = []

        # Try programs of increasing length
        for length in range(1, max_length + 1):
            programs = self.generate_programs(length)

            for program in programs:
                if self.verify_program(program, train_examples):
                    successful_programs.append(program)
                    # Prefer shorter programs
                    if length == 1:
                        return [program]

        return successful_programs[:10]  # Return top 10

    def generate_programs(self, length: int) -> List[List[str]]:
        """Generate all possible programs of given length"""
        if length == 1:
            return [[op] for op in self.atomic_ops.keys()]

        # Generate combinations
        programs = []
        for ops_combo in product(self.atomic_ops.keys(), repeat=length):
            programs.append(list(ops_combo))

            # Limit to prevent explosion
            if len(programs) >= 1000:
                break

        return programs

    def verify_program(self, program: List[str], train_examples: List[dict]) -> bool:
        """Verify if program solves all training examples"""
        for example in train_examples:
            input_grid = np.array(example['input'])
            expected = np.array(example['output'])

            try:
                result = self.execute_program(input_grid, program)

                if result.shape != expected.shape:
                    return False

                if not np.array_equal(result, expected):
                    return False
            except:
                return False

        return True

    def execute_program(self, grid: np.ndarray, program: List[str]) -> np.ndarray:
        """Execute a program on a grid"""
        result = grid.copy()

        for op_name in program:
            op_func = self.atomic_ops[op_name]
            result = op_func(result)

        return result


class DSLSolver:
    """DSL-based solver with program synthesis"""

    def __init__(self):
        self.synthesizer = ProgramSynthesizer()

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """Solve task using program synthesis"""
        predictions = []

        # Synthesize programs from training examples
        programs = self.synthesizer.synthesize(task['train'], max_length=2)

        # If no perfect program found, use heuristics
        if not programs:
            programs = self.fallback_heuristics(task['train'])

        # Apply programs to test inputs
        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            attempts = []
            for program in programs[:max_attempts]:
                try:
                    output = self.synthesizer.execute_program(test_input, program)
                    attempts.append(output.tolist())
                except:
                    attempts.append(test_input.tolist())

            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def fallback_heuristics(self, train_examples: List[dict]) -> List[List[str]]:
        """Fallback heuristic programs"""
        heuristics = []

        # Check for simple patterns
        all_same_size = all(
            np.array(ex['input']).shape == np.array(ex['output']).shape
            for ex in train_examples
        )

        if all_same_size:
            # Same size: try simple transforms
            heuristics = [
                ['identity'],
                ['rot90'],
                ['flip_h'],
                ['flip_v'],
                ['invert_colors'],
            ]
        else:
            # Size change: try scaling/cropping
            heuristics = [
                ['crop_nonzero'],
                ['scale_2x'],
                ['downscale_2x'],
                ['crop_nonzero', 'scale_2x'],
                ['downscale_2x', 'crop_nonzero'],
            ]

        return heuristics


def main():
    print("ARC Prize 2025 - DSL-Based Program Search V7")
    print("=" * 70)

    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    solver = DSLSolver()
    submission = {}

    print("\nGenerating predictions with DSL program synthesis...")
    print("(Searching for shortest programs that solve training examples)")

    for i, (task_id, task) in enumerate(test_challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_challenges)} tasks...")

        predictions = solver.solve(task, max_attempts=2)

        task_submission = []
        for test_predictions in predictions:
            test_dict = {
                "attempt_1": test_predictions[0],
                "attempt_2": test_predictions[1]
            }
            task_submission.append(test_dict)

        submission[task_id] = task_submission

    print(f"\n✓ Generated predictions for {len(submission)} tasks")

    output_path = Path('/kaggle/working/submission.json')
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"✓ Saved submission to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")

    print("\n" + "=" * 70)
    print("V7 DSL-based submission complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
