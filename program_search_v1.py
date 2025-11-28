"""
ARC Prize 2025 - Program Search V1
Search for DSL programs that solve training examples
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from arc_dsl_v1 import ARCDSL, Grid, grid_from_list, grid_to_list
import time
from multiprocessing import Pool, cpu_count
import itertools


class ProgramSearcher:
    """Search for DSL programs that solve ARC tasks"""

    def __init__(self):
        self.dsl = ARCDSL()
        self.primitives = self._get_all_primitives()
        print(f"Initialized with {len(self.primitives)} primitives")

    def _get_all_primitives(self) -> List[Tuple[str, Callable]]:
        """Get all DSL primitives as (name, function) pairs"""
        primitives = []

        # Grid transformations
        primitives.append(('rot90', self.dsl.rot90))
        primitives.append(('rot180', self.dsl.rot180))
        primitives.append(('rot270', self.dsl.rot270))
        primitives.append(('hmirror', self.dsl.hmirror))
        primitives.append(('vmirror', self.dsl.vmirror))
        primitives.append(('dmirror', self.dsl.dmirror))

        # Special handling for parameterized functions
        for factor in [2, 3]:
            primitives.append((f'upscale_{factor}', lambda g, f=factor: self.dsl.upscale(g, f)))
            primitives.append((f'downscale_{factor}', lambda g, f=factor: self.dsl.downscale(g, f)))

        # Color operations (simplified for grid->grid)
        primitives.append(('color_invert', self._color_invert))
        primitives.append(('color_shift_1', self._color_shift_1))
        primitives.append(('color_shift_2', self._color_shift_2))
        primitives.append(('color_mod_3', self._color_mod_3))
        primitives.append(('color_mod_5', self._color_mod_5))
        primitives.append(('binarize', self._binarize))

        # Pattern operations
        primitives.append(('fill_holes', self._fill_holes))
        primitives.append(('extend_pattern', self._extend_pattern))

        # Composite operations
        primitives.append(('rot90_mirror', lambda g: self.dsl.hmirror(self.dsl.rot90(g))))
        primitives.append(('double_mirror', lambda g: self.dsl.hmirror(self.dsl.vmirror(g))))

        return primitives

    # =====================================
    # Color transform helpers
    # =====================================

    def _color_invert(self, grid: Grid) -> Grid:
        """Invert colors (9 - value)"""
        return tuple(tuple(9 - v if v > 0 else 0 for v in row) for row in grid)

    def _color_shift_1(self, grid: Grid) -> Grid:
        """Shift colors by 1"""
        return tuple(tuple((v + 1) % 10 for v in row) for row in grid)

    def _color_shift_2(self, grid: Grid) -> Grid:
        """Shift colors by 2"""
        return tuple(tuple((v + 2) % 10 for v in row) for row in grid)

    def _color_mod_3(self, grid: Grid) -> Grid:
        """Colors modulo 3"""
        return tuple(tuple(v % 3 if v > 0 else 0 for v in row) for row in grid)

    def _color_mod_5(self, grid: Grid) -> Grid:
        """Colors modulo 5"""
        return tuple(tuple(v % 5 if v > 0 else 0 for v in row) for row in grid)

    def _binarize(self, grid: Grid) -> Grid:
        """Convert to binary (0 or 1)"""
        return tuple(tuple(1 if v > 0 else 0 for v in row) for row in grid)

    def _fill_holes(self, grid: Grid) -> Grid:
        """Fill 0-valued cells surrounded by non-zero"""
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        for i in range(1, h-1):
            for j in range(1, w-1):
                if grid[i][j] == 0:
                    neighbors = [
                        grid[i-1][j], grid[i+1][j],
                        grid[i][j-1], grid[i][j+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    if len(non_zero) >= 3:
                        result[i][j] = max(set(non_zero), key=non_zero.count)

        return tuple(tuple(row) for row in result)

    def _extend_pattern(self, grid: Grid) -> Grid:
        """Try to extend repeating patterns"""
        # Simple implementation: if grid is small, tile it 2x2
        h, w = len(grid), len(grid[0])
        if h <= 15 and w <= 15:
            return self.dsl.vconcat(
                self.dsl.hconcat(grid, grid),
                self.dsl.hconcat(grid, grid)
            )
        return grid

    # =====================================
    # Program verification
    # =====================================

    def verify_program(self, program: List[Callable], train_examples: List[Dict]) -> bool:
        """Check if program solves all training examples"""
        for example in train_examples:
            input_grid = grid_from_list(example['input'])
            expected = grid_from_list(example['output'])

            try:
                result = self.execute_program(input_grid, program)

                # Check if result matches expected
                if result != expected:
                    return False
            except Exception:
                return False

        return True

    def execute_program(self, grid: Grid, program: List[Callable]) -> Grid:
        """Execute a program (list of functions) on a grid"""
        result = grid
        for func in program:
            result = func(result)
        return result

    def match_score(self, output: Grid, expected: Grid) -> float:
        """Calculate match score between two grids"""
        if len(output) != len(expected) or len(output[0]) != len(expected[0]):
            return 0.0

        total = len(output) * len(output[0])
        matches = sum(1 for i in range(len(output))
                     for j in range(len(output[0]))
                     if output[i][j] == expected[i][j])

        return matches / total if total > 0 else 0.0

    # =====================================
    # Single-step search
    # =====================================

    def single_step_search(self, train_examples: List[Dict]) -> List[Tuple[List[str], float]]:
        """Try all single primitives"""
        candidates = []

        print("\nSingle-step search...")
        for name, func in self.primitives:
            total_score = 0
            valid = True

            for example in train_examples:
                input_grid = grid_from_list(example['input'])
                expected = grid_from_list(example['output'])

                try:
                    result = func(input_grid)
                    score = self.match_score(result, expected)
                    total_score += score

                    if score < 1.0:
                        valid = False
                except:
                    valid = False
                    break

            if valid:
                print(f"  ✓ Found: {name} (perfect match!)")
                return [([name], 1.0)]
            elif total_score > 0:
                avg_score = total_score / len(train_examples)
                if avg_score > 0.5:
                    candidates.append(([name], avg_score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    # =====================================
    # Two-step search
    # =====================================

    def two_step_search(self, train_examples: List[Dict]) -> List[Tuple[List[str], float]]:
        """Try two-primitive combinations"""
        candidates = []

        print("\nTwo-step search...")
        total_combos = len(self.primitives) ** 2
        checked = 0

        # Focus on promising combinations
        promising_first = ['rot90', 'rot180', 'rot270', 'hmirror', 'vmirror',
                          'upscale_2', 'upscale_3', 'downscale_2']
        promising_second = ['rot90', 'hmirror', 'vmirror', 'color_shift_1',
                           'color_shift_2', 'binarize', 'fill_holes']

        for name1, func1 in self.primitives:
            # Prioritize promising combinations
            if name1 not in promising_first and checked > 200:
                continue

            for name2, func2 in self.primitives:
                if name2 not in promising_second and checked > 200:
                    continue

                checked += 1
                if checked % 100 == 0:
                    print(f"  Checked {checked}/{min(total_combos, 500)} combinations...")

                total_score = 0
                valid = True

                for example in train_examples:
                    input_grid = grid_from_list(example['input'])
                    expected = grid_from_list(example['output'])

                    try:
                        # Apply two operations
                        temp = func1(input_grid)
                        result = func2(temp)
                        score = self.match_score(result, expected)
                        total_score += score

                        if score < 1.0:
                            valid = False
                    except:
                        valid = False
                        break

                if valid:
                    print(f"  ✓ Found: {name1} → {name2} (perfect match!)")
                    return [([name1, name2], 1.0)]
                elif total_score > 0:
                    avg_score = total_score / len(train_examples)
                    if avg_score > 0.7:
                        candidates.append(([name1, name2], avg_score))

                # Limit search time
                if checked >= 500:
                    break

            if checked >= 500:
                break

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    # =====================================
    # Main search
    # =====================================

    def search(self, task: Dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """Search for programs that solve the task"""
        train_examples = task['train']

        print(f"\nSearching for solution ({len(train_examples)} training examples)...")

        # Try single-step first (fast)
        single_candidates = self.single_step_search(train_examples)

        # If no perfect single-step, try two-step
        if not single_candidates or single_candidates[0][1] < 1.0:
            two_candidates = self.two_step_search(train_examples)

            # Combine candidates
            all_candidates = single_candidates + two_candidates
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        else:
            all_candidates = single_candidates

        # Generate predictions for test examples
        predictions = []

        for test_example in task['test']:
            test_input = grid_from_list(test_example['input'])
            attempts = []

            # Use top candidates
            for program_names, score in all_candidates[:max_attempts]:
                try:
                    # Build program from names
                    program = []
                    for name in program_names:
                        for pname, pfunc in self.primitives:
                            if pname == name:
                                program.append(pfunc)
                                break

                    result = self.execute_program(test_input, program)
                    attempts.append(grid_to_list(result))

                    if len(attempts) >= max_attempts:
                        break
                except:
                    pass

            # Fill remaining attempts with input
            while len(attempts) < max_attempts:
                attempts.append(grid_to_list(test_input))

            predictions.append(attempts[:max_attempts])

        # Report results
        if all_candidates:
            print(f"\nTop candidates:")
            for i, (program_names, score) in enumerate(all_candidates[:3]):
                print(f"  {i+1}. {' → '.join(program_names)}: {score:.2%} match")
        else:
            print("\nNo good candidates found, using identity")

        return predictions


# =====================================
# Main solver
# =====================================

class DSLSolver:
    """Main solver using Program Search"""

    def __init__(self):
        self.searcher = ProgramSearcher()

    def solve(self, task: Dict) -> List[List[List[int]]]:
        """Solve an ARC task"""
        return self.searcher.search(task, max_attempts=2)


# =====================================
# Testing
# =====================================

def test_on_training_sample():
    """Test on a few training examples"""
    print("=" * 70)
    print("ARC Prize 2025 - Program Search V1")
    print("=" * 70)

    # Load training data
    with open('data/arc-agi_training_challenges.json', 'r') as f:
        train_data = json.load(f)
    with open('data/arc-agi_training_solutions.json', 'r') as f:
        train_solutions = json.load(f)

    solver = DSLSolver()

    # Test on first 3 tasks
    correct = 0
    for i, (task_id, task) in enumerate(list(train_data.items())[:3]):
        print(f"\n{'='*70}")
        print(f"Task {i+1}: {task_id}")
        print(f"Training examples: {len(task['train'])}")

        # Get predictions
        predictions = solver.solve(task)

        # Check against solutions
        solutions = train_solutions[task_id]
        match = False

        for pred_attempts, solution in zip(predictions, solutions):
            for attempt in pred_attempts:
                if attempt == solution:
                    match = True
                    break

        if match:
            print("✓ CORRECT!")
            correct += 1
        else:
            print("✗ Incorrect")

    print(f"\n{'='*70}")
    print(f"Results: {correct}/3 correct")
    print("=" * 70)


if __name__ == "__main__":
    test_on_training_sample()