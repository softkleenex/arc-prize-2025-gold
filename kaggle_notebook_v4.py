"""
ARC Prize 2025 - Version 4: Advanced Multi-Step Solver

Added 40+ transformations with:
- Multi-step transformations
- Conditional logic
- Advanced pattern detection
- Boundary operations
- Object manipulation
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Callable


class AdvancedSolver:
    """Advanced solver with 40+ transformations including multi-step logic."""

    def __init__(self):
        self.transformations = [
            # Core basics
            self.identity,
            self.flip_horizontal,
            self.flip_vertical,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.transpose,

            # Advanced color operations
            self.swap_all_colors,
            self.replace_most_common,
            self.gradient_color_map,
            self.checkerboard_colors,
            self.invert_colors,
            self.color_by_distance_from_edge,
            self.color_by_row_col_sum,
            self.swap_background_foreground,

            # Grid manipulation
            self.crop_to_content,
            self.pad_to_double,
            self.pad_to_square,
            self.split_and_rearrange,
            self.mirror_horizontal,
            self.mirror_vertical,
            self.frame_with_border,
            self.remove_border,
            self.compress_empty_rows_cols,

            # Pattern operations
            self.fill_gaps,
            self.extend_all_lines,
            self.extend_vertical_lines,
            self.complete_grid_pattern,
            self.replicate_smallest_unit,
            self.replicate_by_thirds,
            self.detect_and_complete_rectangles,

            # Symmetry operations
            self.make_horizontally_symmetric,
            self.make_vertically_symmetric,
            self.make_diagonal_symmetric,
            self.complete_symmetry_all,

            # Multi-step combinations
            self.rotate_and_flip,
            self.scale_and_tile,
            self.color_and_rotate,
            self.crop_and_scale,
            self.mirror_and_color,

            # Scale variations
            self.scale_2x,
            self.scale_3x,
            self.scale_4x,
            self.scale_half,
            self.tile_2x2,
            self.tile_3x3,
        ]

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        predictions = []

        for test_example in task['test']:
            test_input = np.array(test_example['input'])
            best_transforms = self.find_best_transformations(task['train'], max_attempts)

            attempts = []
            for transform_func in best_transforms[:max_attempts]:
                try:
                    output = transform_func(test_input)
                    attempts.append(output.tolist())
                except:
                    attempts.append(test_input.tolist())

            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def find_best_transformations(self, train_examples: List[dict], n_transforms: int = 2) -> List[Callable]:
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

        sorted_transforms = sorted(transform_scores.items(), key=lambda x: -x[1])
        return [func for func, score in sorted_transforms[:n_transforms]]

    # ============ BASIC TRANSFORMATIONS ============
    def identity(self, grid): return grid.copy()
    def flip_horizontal(self, grid): return np.fliplr(grid)
    def flip_vertical(self, grid): return np.flipud(grid)
    def rotate_90(self, grid): return np.rot90(grid, k=-1)
    def rotate_180(self, grid): return np.rot90(grid, k=2)
    def rotate_270(self, grid): return np.rot90(grid, k=-3)
    def transpose(self, grid): return grid.T
    def scale_2x(self, grid): return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    def scale_3x(self, grid): return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
    def scale_4x(self, grid): return np.repeat(np.repeat(grid, 4, axis=0), 4, axis=1)
    def tile_2x2(self, grid): return np.block([[grid, grid], [grid, grid]])
    def tile_3x3(self, grid): return np.block([[grid]*3]*3)

    # ============ ADVANCED COLOR OPERATIONS ============
    def swap_all_colors(self, grid):
        result = grid.copy()
        for i in range(10):
            result[grid == i] = (i + 5) % 10
        return result

    def replace_most_common(self, grid):
        unique, counts = np.unique(grid, return_counts=True)
        if len(unique) < 2: return grid.copy()
        most_common = unique[np.argmax(counts)]
        second = unique[np.argsort(-counts)[1]]
        result = grid.copy()
        result[grid == most_common] = second
        return result

    def gradient_color_map(self, grid):
        result = grid.copy()
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0:
                    result[i, j] = int((i + j) / (h + w) * 9) + 1
        return result

    def checkerboard_colors(self, grid):
        result = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    result[i, j] = 1 if (i + j) % 2 == 0 else 2
        return result

    def invert_colors(self, grid):
        return 9 - grid

    def color_by_distance_from_edge(self, grid):
        result = grid.copy()
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0:
                    dist = min(i, j, h-1-i, w-1-j)
                    result[i, j] = min(dist + 1, 9)
        return result

    def color_by_row_col_sum(self, grid):
        result = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    result[i, j] = ((i + j) % 9) + 1
        return result

    def swap_background_foreground(self, grid):
        unique, counts = np.unique(grid, return_counts=True)
        if len(unique) < 2: return grid.copy()
        bg = unique[np.argmax(counts)]
        result = grid.copy()
        result[grid == bg] = -1
        result[grid != -1] = bg
        result[result == -1] = unique[0] if unique[0] != bg else unique[1]
        return result

    # ============ GRID MANIPULATION ============
    def crop_to_content(self, grid):
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0: return grid.copy()
        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)
        return grid[min_row:max_row+1, min_col:max_col+1]

    def pad_to_double(self, grid):
        h, w = grid.shape
        result = np.zeros((h*2, w*2), dtype=grid.dtype)
        result[:h, :w] = grid
        return result

    def pad_to_square(self, grid):
        h, w = grid.shape
        if h == w: return grid.copy()
        size = max(h, w)
        result = np.zeros((size, size), dtype=grid.dtype)
        result[:h, :w] = grid
        return result

    def split_and_rearrange(self, grid):
        h, w = grid.shape
        if h % 2 == 0 and w % 2 == 0:
            mh, mw = h // 2, w // 2
            tl = grid[:mh, :mw]
            tr = grid[:mh, mw:]
            bl = grid[mh:, :mw]
            br = grid[mh:, mw:]
            return np.block([[br, bl], [tr, tl]])
        return grid.copy()

    def mirror_horizontal(self, grid):
        return np.hstack([grid, np.fliplr(grid)])

    def mirror_vertical(self, grid):
        return np.vstack([grid, np.flipud(grid)])

    def frame_with_border(self, grid):
        h, w = grid.shape
        result = np.zeros((h+2, w+2), dtype=grid.dtype)
        result[1:-1, 1:-1] = grid
        non_zero = grid[grid != 0]
        if len(non_zero) > 0:
            border_color = np.bincount(non_zero).argmax()
            result[0, :] = border_color
            result[-1, :] = border_color
            result[:, 0] = border_color
            result[:, -1] = border_color
        return result

    def remove_border(self, grid):
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return grid.copy()
        return grid[1:-1, 1:-1]

    def compress_empty_rows_cols(self, grid):
        # Remove empty rows and columns
        non_zero_rows = np.any(grid != 0, axis=1)
        non_zero_cols = np.any(grid != 0, axis=0)
        if not np.any(non_zero_rows) or not np.any(non_zero_cols):
            return grid.copy()
        return grid[non_zero_rows][:, non_zero_cols]

    def scale_half(self, grid):
        h, w = grid.shape
        if h % 2 == 0 and w % 2 == 0:
            return grid[::2, ::2]
        return grid.copy()

    # ============ PATTERN OPERATIONS ============
    def fill_gaps(self, grid):
        result = grid.copy()
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                if grid[i, j] == 0:
                    neighbors = [grid[i-1,j], grid[i+1,j], grid[i,j-1], grid[i,j+1]]
                    neighbors = [n for n in neighbors if n != 0]
                    if neighbors:
                        result[i, j] = max(set(neighbors), key=neighbors.count)
        return result

    def extend_all_lines(self, grid):
        result = grid.copy()
        for i in range(grid.shape[0]):
            row_colors = grid[i][grid[i] != 0]
            if len(row_colors) > 0:
                result[i, :] = row_colors[0]
        return result

    def extend_vertical_lines(self, grid):
        result = grid.copy()
        for j in range(grid.shape[1]):
            col_colors = grid[:, j][grid[:, j] != 0]
            if len(col_colors) > 0:
                result[:, j] = col_colors[0]
        return result

    def complete_grid_pattern(self, grid):
        result = grid.copy()
        if grid.shape[0] >= 2 and grid.shape[1] >= 2:
            unit = grid[:2, :2]
            for i in range(0, grid.shape[0], 2):
                for j in range(0, grid.shape[1], 2):
                    if i+2 <= grid.shape[0] and j+2 <= grid.shape[1]:
                        result[i:i+2, j:j+2] = unit
        return result

    def replicate_smallest_unit(self, grid):
        if grid.shape[0] >= 2 and grid.shape[1] >= 2:
            unit = grid[:2, :2]
            reps_h = grid.shape[0] // 2
            reps_w = grid.shape[1] // 2
            result = np.tile(unit, (reps_h, reps_w))
            if result.shape[0] < grid.shape[0]:
                result = np.vstack([result, grid[-1:, :result.shape[1]]])
            if result.shape[1] < grid.shape[1]:
                result = np.hstack([result, grid[:result.shape[0], -1:]])
            return result[:grid.shape[0], :grid.shape[1]]
        return grid.copy()

    def replicate_by_thirds(self, grid):
        h, w = grid.shape
        if h % 3 == 0 and w % 3 == 0:
            unit = grid[:h//3, :w//3]
            return np.tile(unit, (3, 3))
        return grid.copy()

    def detect_and_complete_rectangles(self, grid):
        result = grid.copy()
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return result

        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)

        # Fill bounding rectangle with most common non-zero color
        colors = grid[non_zero[:, 0], non_zero[:, 1]]
        most_common_color = np.bincount(colors).argmax()

        result[min_row:max_row+1, min_col:max_col+1] = most_common_color
        return result

    # ============ SYMMETRY OPERATIONS ============
    def make_horizontally_symmetric(self, grid):
        result = grid.copy()
        h, w = grid.shape
        mid = w // 2
        for i in range(h):
            for j in range(mid):
                if result[i, j] != 0:
                    result[i, w-1-j] = result[i, j]
        return result

    def make_vertically_symmetric(self, grid):
        result = grid.copy()
        h, w = grid.shape
        mid = h // 2
        for i in range(mid):
            for j in range(w):
                if result[i, j] != 0:
                    result[h-1-i, j] = result[i, j]
        return result

    def make_diagonal_symmetric(self, grid):
        h, w = grid.shape
        if h == w:
            result = grid.copy()
            for i in range(h):
                for j in range(i+1, w):
                    if result[i, j] != 0 and result[j, i] == 0:
                        result[j, i] = result[i, j]
                    elif result[j, i] != 0 and result[i, j] == 0:
                        result[i, j] = result[j, i]
            return result
        return grid.copy()

    def complete_symmetry_all(self, grid):
        result = self.make_horizontally_symmetric(grid)
        result = self.make_vertically_symmetric(result)
        return result

    # ============ MULTI-STEP COMBINATIONS ============
    def rotate_and_flip(self, grid):
        return np.fliplr(np.rot90(grid, k=-1))

    def scale_and_tile(self, grid):
        scaled = self.scale_2x(grid)
        return self.tile_2x2(scaled)

    def color_and_rotate(self, grid):
        swapped = self.swap_all_colors(grid)
        return self.rotate_90(swapped)

    def crop_and_scale(self, grid):
        cropped = self.crop_to_content(grid)
        return self.scale_2x(cropped)

    def mirror_and_color(self, grid):
        mirrored = self.mirror_horizontal(grid)
        return self.gradient_color_map(mirrored)


def main():
    print("ARC Prize 2025 - Advanced Multi-Step Solver V4")
    print("=" * 70)

    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    solver = AdvancedSolver()
    submission = {}

    print("\nGenerating predictions with 40+ advanced transformations...")
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
    print("V4 Advanced submission complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
