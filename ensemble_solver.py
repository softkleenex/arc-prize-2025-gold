"""
Ensemble Solver: Combine multiple approaches with voting
Inspired by MindsAI's AIRV technique
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from direct_matcher import DirectMatcher
from perfect_solver import PerfectSolver
from three_step_search import SmartSearchSolver
from hybrid_solver import HybridSolver
from dsl_v3_pattern import AdvancedProgramSearcher


class EnsembleSolver:
    """Ensemble multiple solvers with voting"""

    def __init__(self):
        self.solvers = [
            DirectMatcher(),
            PerfectSolver(),
            SmartSearchSolver(),
            HybridSolver(),
            AdvancedProgramSearcher()
        ]

    def solve(self, task: Dict) -> List[List]:
        """Solve using ensemble of methods"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        predictions = []
        for test_example in test_examples:
            test_input = test_example['input']
            attempts = self.ensemble_solve(train_examples, test_input, task)
            predictions.append(attempts[:2])

        return predictions

    def ensemble_solve(self, train_examples: List[Dict], test_input: List[List], task: Dict) -> List:
        """Generate solutions using ensemble"""

        all_attempts = []

        # Collect predictions from all solvers
        for solver in self.solvers:
            try:
                # Get predictions from this solver
                if hasattr(solver, 'solve'):
                    # Full task solver
                    preds = solver.solve(task)
                    if preds and len(preds) > 0:
                        all_attempts.extend(preds[0])  # First test case predictions
                elif hasattr(solver, 'search'):
                    # Search-based solver
                    preds = solver.search(task)
                    if preds and len(preds) > 0:
                        all_attempts.extend(preds[0])
            except:
                continue

        # Also try augmentation-based approaches
        augmented_attempts = self.augmentation_solve(train_examples, test_input)
        all_attempts.extend(augmented_attempts)

        # Vote on best solutions
        final_attempts = self.vote_and_select(all_attempts, train_examples)

        # Ensure at least 2 attempts
        while len(final_attempts) < 2:
            if final_attempts:
                final_attempts.append(final_attempts[0])
            else:
                final_attempts.append(test_input)

        return final_attempts

    def augmentation_solve(self, train_examples: List[Dict], test_input: List[List]) -> List:
        """Apply augmentation-based solving (inspired by AIRV)"""
        attempts = []

        # Try with rotations
        for rotation in range(4):
            rotated_input = self.rotate_n_times(test_input, rotation)

            # Apply simple transformations
            for transform in [self.identity, self.mirror_h, self.mirror_v]:
                transformed = transform(rotated_input)

                # Apply learned color mapping
                color_map = self.learn_color_map(train_examples)
                if color_map:
                    result = self.apply_color_map(transformed, color_map)

                    # Reverse the rotation
                    result = self.rotate_n_times(result, (4 - rotation) % 4)

                    if self.is_valid_grid(result):
                        attempts.append(result)

        return attempts[:3]  # Return top 3 augmentation attempts

    def vote_and_select(self, all_attempts: List, train_examples: List[Dict]) -> List:
        """Vote on best solutions"""

        if not all_attempts:
            return []

        # Convert to hashable format for voting
        hashable_attempts = []
        for attempt in all_attempts:
            try:
                if isinstance(attempt, (list, tuple)):
                    # Convert to tuple
                    hashable = tuple(tuple(row) for row in attempt)
                    hashable_attempts.append(hashable)
            except:
                continue

        if not hashable_attempts:
            return []

        # Count votes
        votes = Counter(hashable_attempts)

        # Get top 2 most common
        top_solutions = []
        for solution, count in votes.most_common(2):
            # Convert back to list
            top_solutions.append([list(row) for row in solution])

        return top_solutions

    def rotate_n_times(self, grid: List[List], n: int) -> List[List]:
        """Rotate grid n times (90 degrees each)"""
        result = grid
        for _ in range(n % 4):
            result = [list(row) for row in zip(*result[::-1])]
        return result

    def identity(self, grid: List[List]) -> List[List]:
        """Return grid as-is"""
        return [list(row) for row in grid]

    def mirror_h(self, grid: List[List]) -> List[List]:
        """Mirror horizontally"""
        return grid[::-1]

    def mirror_v(self, grid: List[List]) -> List[List]:
        """Mirror vertically"""
        return [row[::-1] for row in grid]

    def learn_color_map(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn color mapping from examples"""
        all_mappings = []

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                mapping = {}
                for i in range(len(in_grid)):
                    for j in range(len(in_grid[0])):
                        if in_grid[i][j] != 0:
                            mapping[in_grid[i][j]] = out_grid[i][j]
                all_mappings.append(mapping)

        # Consensus mapping
        final_map = {}
        if all_mappings:
            all_keys = set()
            for m in all_mappings:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [m.get(key) for m in all_mappings if key in m]
                if values and all(v == values[0] for v in values):
                    final_map[key] = values[0]

        return final_map

    def apply_color_map(self, grid: List[List], color_map: Dict[int, int]) -> List[List]:
        """Apply color mapping"""
        result = []
        for row in grid:
            new_row = []
            for val in row:
                new_row.append(color_map.get(val, val))
            result.append(new_row)
        return result

    def is_valid_grid(self, grid: Any) -> bool:
        """Check if grid is valid"""
        try:
            if not isinstance(grid, (list, tuple)):
                return False
            if len(grid) == 0 or len(grid[0]) == 0:
                return False
            h, w = len(grid), len(grid[0])
            if h > 100 or w > 100:
                return False

            # Check all rows same length
            for row in grid:
                if len(row) != w:
                    return False

            # Check valid colors
            for row in grid:
                for val in row:
                    if not isinstance(val, int) or val < 0 or val > 9:
                        return False

            return True
        except:
            return False


class ImprovedEnsemble(EnsembleSolver):
    """Improved ensemble with better voting and validation"""

    def vote_and_select(self, all_attempts: List, train_examples: List[Dict]) -> List:
        """Improved voting with validation"""

        if not all_attempts:
            return []

        # Score each attempt
        scored_attempts = []
        for attempt in all_attempts:
            if not self.is_valid_grid(attempt):
                continue

            score = self.score_attempt(attempt, train_examples)
            scored_attempts.append((score, attempt))

        # Sort by score
        scored_attempts.sort(key=lambda x: x[0], reverse=True)

        # Return top 2
        final_attempts = []
        for score, attempt in scored_attempts[:2]:
            final_attempts.append(attempt)

        return final_attempts

    def score_attempt(self, attempt: List[List], train_examples: List[Dict]) -> float:
        """Score an attempt based on similarity to training outputs"""
        score = 0.0

        # Check shape similarity
        for ex in train_examples:
            out_grid = ex['output']

            # Shape match
            if len(attempt) == len(out_grid) and len(attempt[0]) == len(out_grid[0]):
                score += 1.0

            # Color distribution similarity
            attempt_colors = Counter(v for row in attempt for v in row)
            output_colors = Counter(v for row in out_grid for v in row)

            common_colors = set(attempt_colors.keys()) & set(output_colors.keys())
            if len(common_colors) > 0:
                score += len(common_colors) / max(len(attempt_colors), len(output_colors))

        return score / len(train_examples) if train_examples else 0