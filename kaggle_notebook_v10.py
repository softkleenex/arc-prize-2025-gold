"""
ARC Prize 2025 - Version 10: Enhanced Hybrid

V8 기반 개선:
- V2-V5의 50+ 변환 모두 통합
- 확장된 DSL operations
- 더 나은 앙상블 전략
- 중복 제거 및 우선순위 최적화
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Tuple
from collections import Counter


class EnhancedHybridSolver:
    """V8 개선 버전 - 더 많은 변환과 더 나은 앙상블"""

    def __init__(self):
        # DSL operations (확장)
        self.dsl_ops = self._init_dsl_ops()

        # V2-V5의 모든 변환
        self.all_transforms = self._init_all_transforms()

    def _init_dsl_ops(self):
        """확장된 DSL operations"""
        return {
            # Identity
            'identity': lambda g: g.copy(),

            # Geometric (rotation)
            'rot90': lambda g: np.rot90(g, k=-1),
            'rot180': lambda g: np.rot90(g, k=2),
            'rot270': lambda g: np.rot90(g, k=-3),

            # Geometric (flip)
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
            'transpose': lambda g: g.T,

            # Scaling
            'scale_2x': lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),
            'scale_3x': lambda g: np.repeat(np.repeat(g, 3, axis=0), 3, axis=1),
            'scale_4x': lambda g: np.repeat(np.repeat(g, 4, axis=0), 4, axis=1),
            'downscale_2x': lambda g: g[::2, ::2] if g.shape[0] >= 2 and g.shape[1] >= 2 else g.copy(),
            'downscale_3x': lambda g: g[::3, ::3] if g.shape[0] >= 3 and g.shape[1] >= 3 else g.copy(),

            # Cropping
            'crop_nonzero': lambda g: self._crop_nonzero(g),
            'pad_square': lambda g: self._pad_square(g),

            # Color operations
            'invert_colors': lambda g: 9 - g,
            'binarize': lambda g: (g > 0).astype(g.dtype),
            'color_mod_3': lambda g: g % 3,
            'color_mod_5': lambda g: g % 5,
            'color_shift_1': lambda g: (g + 1) % 10,
            'color_shift_2': lambda g: (g + 2) % 10,
        }

    def _init_all_transforms(self):
        """V2-V5의 모든 변환 통합"""
        transforms = []

        # Basic geometric (V2)
        transforms.extend([
            lambda g: g.copy(),  # identity
            lambda g: np.rot90(g, k=-1),  # rot90
            lambda g: np.rot90(g, k=2),   # rot180
            lambda g: np.rot90(g, k=-3),  # rot270
            lambda g: np.fliplr(g),       # flip_h
            lambda g: np.flipud(g),       # flip_v
            lambda g: g.T,                # transpose
        ])

        # Scaling (V2)
        transforms.extend([
            lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),
            lambda g: np.repeat(np.repeat(g, 3, axis=0), 3, axis=1),
            lambda g: g[::2, ::2] if g.shape[0] >= 2 and g.shape[1] >= 2 else g.copy(),
        ])

        # Color operations (V2-V3)
        transforms.extend([
            lambda g: 9 - g,  # invert
            lambda g: (g > 0).astype(g.dtype),  # binarize
            lambda g: (g + 1) % 10,  # color shift
            lambda g: (g + 2) % 10,
            lambda g: g % 3,
            lambda g: g % 5,
        ])

        # Pattern operations (V3)
        transforms.extend([
            self.pattern_fill,
            self.pattern_replicate,
            self.isolate_largest,
        ])

        # Grid operations (V3)
        transforms.extend([
            lambda g: np.hstack([g, np.fliplr(g)]),  # mirror_h
            lambda g: np.vstack([g, np.flipud(g)]),  # mirror_v
            self.extract_border,
        ])

        # Advanced operations (V4-V5)
        transforms.extend([
            self.color_distance_fill,
            self.symmetry_complete,
            self.detect_and_extend,
            self.object_separation,
        ])

        # Composite operations
        transforms.extend([
            lambda g: self._crop_nonzero(np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)),
            lambda g: np.rot90(np.fliplr(g), k=-1),
            lambda g: (9 - g) if g.max() > 0 else g.copy(),
        ])

        return transforms

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """Enhanced hybrid approach"""
        predictions = []

        # Step 1: DSL Program Search (fast, exact)
        dsl_programs = self.synthesize_programs(task['train'])

        # Step 2: Transform search (comprehensive)
        verified_transforms = self.find_verified_transforms(task['train'])

        # Step 3: Learning-based (pattern extraction)
        learned_rules = self.learn_rules(task['train'])

        # Combine with priority:
        # 1. DSL programs (exact match on training)
        # 2. Verified transforms (exact match on training)
        # 3. Learned rules (pattern-based)
        all_methods = []

        # Priority 1: DSL (highest confidence)
        for prog in dsl_programs[:2]:
            all_methods.append(('dsl', prog))

        # Priority 2: Verified transforms
        for trans in verified_transforms[:3]:
            all_methods.append(('transform', trans))

        # Priority 3: Learned rules
        for rule in learned_rules[:2]:
            all_methods.append(('learned', rule))

        # Apply to test
        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            attempts = []
            seen_outputs = set()  # Avoid duplicates

            for method_type, method in all_methods:
                if len(attempts) >= max_attempts:
                    break

                try:
                    if method_type == 'dsl':
                        output = self.execute_program(test_input, method)
                    elif method_type == 'transform':
                        output = method(test_input)
                    else:  # learned
                        output = self.apply_learned_rule(test_input, method)

                    # Check for duplicates
                    output_tuple = tuple(map(tuple, output.tolist()))
                    if output_tuple not in seen_outputs:
                        attempts.append(output.tolist())
                        seen_outputs.add(output_tuple)
                except:
                    pass

            # Fill remaining attempts with input if needed
            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts[:max_attempts])

        return predictions

    def synthesize_programs(self, train_examples: List[dict]) -> List[List[str]]:
        """DSL program synthesis"""
        successful = []

        # Single operations
        for op_name, op_func in self.dsl_ops.items():
            if self._verify_func(op_func, train_examples):
                successful.append([op_name])

        # Common 2-op combinations
        if len(successful) < 3:
            combos = [
                ['crop_nonzero', 'scale_2x'],
                ['crop_nonzero', 'scale_3x'],
                ['scale_2x', 'crop_nonzero'],
                ['scale_3x', 'crop_nonzero'],
                ['downscale_2x', 'crop_nonzero'],
                ['rot90', 'flip_h'],
                ['flip_h', 'rot90'],
                ['transpose', 'flip_v'],
            ]

            for combo in combos:
                if self._verify_program(combo, train_examples):
                    successful.append(combo)

        return successful if successful else [['identity']]

    def find_verified_transforms(self, train_examples: List[dict]) -> List[Callable]:
        """Find transforms that work on training data"""
        verified = []

        for transform in self.all_transforms:
            if self._verify_func(transform, train_examples):
                verified.append(transform)

        return verified if verified else [lambda g: g.copy()]

    def learn_rules(self, train_examples: List[dict]) -> List[Dict]:
        """Extract patterns from training data"""
        rules = []

        # Size rule
        size_ratios = []
        for ex in train_examples:
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            ratio = (out_arr.shape[0] / in_arr.shape[0],
                    out_arr.shape[1] / in_arr.shape[1])
            size_ratios.append(ratio)

        if len(set(size_ratios)) == 1 and size_ratios[0] != (1.0, 1.0):
            rules.append({'type': 'size', 'ratio': size_ratios[0]})

        # Color mapping rule
        color_map = self._learn_color_mapping(train_examples)
        if color_map and len(color_map) > 0:
            rules.append({'type': 'color_map', 'mapping': color_map})

        # Pattern rule (same structure, different values)
        if self._detect_pattern_rule(train_examples):
            rules.append({'type': 'pattern'})

        return rules if rules else [{'type': 'identity'}]

    def _learn_color_mapping(self, train_examples: List[dict]) -> Dict:
        """Learn consistent color mappings"""
        mappings = {}

        for ex in train_examples:
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])

            if in_arr.shape != out_arr.shape:
                continue

            for i in range(in_arr.shape[0]):
                for j in range(in_arr.shape[1]):
                    in_color = in_arr[i, j]
                    out_color = out_arr[i, j]

                    if in_color not in mappings:
                        mappings[in_color] = []
                    mappings[in_color].append(out_color)

        # Keep only consistent mappings (>80% agreement)
        consistent = {}
        for in_color, out_colors in mappings.items():
            counter = Counter(out_colors)
            most_common = counter.most_common(1)[0]
            if most_common[1] / len(out_colors) > 0.8:
                consistent[in_color] = most_common[0]

        return consistent

    def _detect_pattern_rule(self, train_examples: List[dict]) -> bool:
        """Detect if it's a pattern-based transformation"""
        # Check if all examples have same size transformation
        same_size = all(
            np.array(ex['input']).shape == np.array(ex['output']).shape
            for ex in train_examples
        )
        return same_size

    def apply_learned_rule(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply learned rule"""
        if rule['type'] == 'size':
            h_ratio, w_ratio = rule['ratio']

            if h_ratio == w_ratio == 2.0:
                return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
            elif h_ratio == w_ratio == 3.0:
                return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
            elif h_ratio == w_ratio == 0.5:
                return grid[::2, ::2]
            elif h_ratio == w_ratio == 0.33:
                return grid[::3, ::3]

        elif rule['type'] == 'color_map':
            result = grid.copy()
            for in_color, out_color in rule['mapping'].items():
                result[grid == in_color] = out_color
            return result

        return grid.copy()

    def execute_program(self, grid: np.ndarray, program: List[str]) -> np.ndarray:
        """Execute DSL program"""
        result = grid.copy()
        for op_name in program:
            if op_name in self.dsl_ops:
                result = self.dsl_ops[op_name](result)
        return result

    def _verify_func(self, func: Callable, train_examples: List[dict]) -> bool:
        """Verify function works on all training examples"""
        for ex in train_examples:
            try:
                in_arr = np.array(ex['input'])
                expected = np.array(ex['output'])
                result = func(in_arr)

                if result.shape != expected.shape:
                    return False
                if not np.array_equal(result, expected):
                    return False
            except:
                return False
        return True

    def _verify_program(self, program: List[str], train_examples: List[dict]) -> bool:
        """Verify DSL program works on all training examples"""
        for ex in train_examples:
            try:
                in_arr = np.array(ex['input'])
                expected = np.array(ex['output'])
                result = self.execute_program(in_arr, program)

                if result.shape != expected.shape:
                    return False
                if not np.array_equal(result, expected):
                    return False
            except:
                return False
        return True

    # Helper methods
    def _crop_nonzero(self, grid):
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return grid.copy()
        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)
        return grid[min_row:max_row+1, min_col:max_col+1]

    def _pad_square(self, grid):
        h, w = grid.shape
        if h == w:
            return grid.copy()
        size = max(h, w)
        result = np.zeros((size, size), dtype=grid.dtype)
        result[:h, :w] = grid
        return result

    # Pattern operations
    def pattern_fill(self, grid):
        result = grid.copy()
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                if grid[i, j] == 0:
                    neighbors = [grid[i-1,j], grid[i+1,j], grid[i,j-1], grid[i,j+1]]
                    neighbors = [n for n in neighbors if n != 0]
                    if neighbors:
                        result[i, j] = Counter(neighbors).most_common(1)[0][0]
        return result

    def pattern_replicate(self, grid):
        h, w = grid.shape
        if h <= 10 and w <= 10:
            return np.tile(grid, (2, 2))
        return grid.copy()

    def isolate_largest(self, grid):
        if grid.max() == 0:
            return grid.copy()

        result = np.zeros_like(grid)
        for color in range(1, 10):
            mask = (grid == color)
            if not mask.any():
                continue

            coords = np.argwhere(mask)
            if len(coords) > result.sum():
                result = mask.astype(grid.dtype) * color

        return result

    def extract_border(self, grid):
        result = np.zeros_like(grid)
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return grid.copy()

        result[0, :] = grid[0, :]
        result[-1, :] = grid[-1, :]
        result[:, 0] = grid[:, 0]
        result[:, -1] = grid[:, -1]
        return result

    def color_distance_fill(self, grid):
        result = grid.copy()
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return result

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0:
                    distances = np.sqrt((non_zero[:, 0] - i)**2 + (non_zero[:, 1] - j)**2)
                    nearest = non_zero[np.argmin(distances)]
                    result[i, j] = grid[nearest[0], nearest[1]]

        return result

    def symmetry_complete(self, grid):
        result = grid.copy()
        h, w = grid.shape

        # Horizontal symmetry
        mid = w // 2
        for i in range(h):
            for j in range(mid):
                if result[i, j] != 0 and result[i, w-1-j] == 0:
                    result[i, w-1-j] = result[i, j]
                elif result[i, w-1-j] != 0 and result[i, j] == 0:
                    result[i, j] = result[i, w-1-j]

        return result

    def detect_and_extend(self, grid):
        # Detect repeating patterns and extend
        h, w = grid.shape
        if h >= 6 and w >= 6:
            # Check for 3x3 pattern
            pattern = grid[:3, :3]
            if np.array_equal(grid[3:6, :3], pattern):
                return np.tile(pattern, (h//3 + 1, w//3 + 1))[:h, :w]

        return grid.copy()

    def object_separation(self, grid):
        # Separate different colored regions
        result = np.zeros_like(grid)
        for color in range(1, 10):
            mask = (grid == color)
            if mask.any():
                result += mask.astype(grid.dtype) * color

        return result


def main():
    print("ARC Prize 2025 - Enhanced Hybrid Solver V10")
    print("=" * 70)

    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    solver = EnhancedHybridSolver()
    submission = {}

    print("\nGenerating predictions with Enhanced Hybrid approach...")
    print("(Extended DSL + All V2-V5 Transforms + Learning)")

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
    print("V10 Enhanced Hybrid submission complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
