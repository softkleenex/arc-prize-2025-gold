"""
ARC Prize 2025 - DSL V2 Improved
Enhanced with Object-based operations and pattern learning
"""

import json
import numpy as np
from typing import List, Dict, Tuple, FrozenSet, Set, Callable, Optional
from collections import Counter
from arc_dsl_v1 import ARCDSL, Grid, Object, Objects, grid_from_list, grid_to_list


class ImprovedDSL(ARCDSL):
    """Enhanced DSL with object-based reasoning"""

    def __init__(self):
        super().__init__()
        self.training_patterns = self._analyze_training_patterns()

    def _analyze_training_patterns(self) -> Dict:
        """Analyze common patterns from training data"""
        patterns = {
            'color_mappings': {},
            'size_transformations': [],
            'object_operations': []
        }

        # Will be populated during training
        return patterns

    # =====================================
    # Enhanced Object Operations
    # =====================================

    def extract_and_transform_objects(self, grid: Grid,
                                     transform: Callable) -> Grid:
        """Extract objects, transform each, and reassemble"""
        bg_color = self.mostcolor(grid)
        objects = self.objects(grid, diagonal=True, without_bg=True)

        # Start with background
        result = self.canvas(bg_color, (len(grid), len(grid[0])))

        # Transform and paint each object
        for obj in objects:
            transformed = transform(obj)
            result = self.paint(result, transformed)

        return result

    def color_by_size(self, grid: Grid) -> Grid:
        """Color objects based on their size"""
        objects = self.objects(grid, diagonal=True, without_bg=True)
        bg_color = self.mostcolor(grid)
        result = self.canvas(bg_color, (len(grid), len(grid[0])))

        # Sort objects by size
        obj_sizes = [(obj, len(obj)) for obj in objects]
        obj_sizes.sort(key=lambda x: x[1])

        # Assign colors based on size order
        for i, (obj, _) in enumerate(obj_sizes):
            new_color = (i + 1) % 10
            recolored = self.recolor(obj, new_color)
            result = self.paint(result, recolored)

        return result

    def connect_same_color(self, grid: Grid) -> Grid:
        """Connect objects of the same color"""
        objects = self.objects(grid, diagonal=False, without_bg=True)
        result = grid  # Start with original

        # Group by color
        color_groups = {}
        for obj in objects:
            if obj:
                color = next(iter(obj))[0]
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(obj)

        # Connect objects of same color
        for color, objs in color_groups.items():
            if len(objs) > 1:
                # Get centers
                centers = [self.center(obj) for obj in objs]

                # Connect each pair
                for i in range(len(centers) - 1):
                    c1, c2 = centers[i], centers[i + 1]
                    # Draw line between centers
                    line_indices = self._draw_line(c1, c2)
                    result = self.fill(result, color, line_indices)

        return result

    def _draw_line(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> FrozenSet:
        """Draw line between two points"""
        i1, j1 = p1
        i2, j2 = p2
        points = set()

        # Bresenham's line algorithm (simplified)
        di = abs(i2 - i1)
        dj = abs(j2 - j1)
        si = 1 if i2 > i1 else -1
        sj = 1 if j2 > j1 else -1

        if di > dj:
            for i in range(i1, i2 + si, si):
                j = j1 + (j2 - j1) * (i - i1) // (i2 - i1) if i2 != i1 else j1
                points.add((i, j))
        else:
            for j in range(j1, j2 + sj, sj):
                i = i1 + (i2 - i1) * (j - j1) // (j2 - j1) if j2 != j1 else i1
                points.add((i, j))

        return frozenset(points)

    def fill_between_objects(self, grid: Grid) -> Grid:
        """Fill space between objects with pattern"""
        objects = self.objects(grid, diagonal=True, without_bg=True)

        if len(objects) < 2:
            return grid

        bg_color = self.mostcolor(grid)
        result = grid

        # Get bounding boxes
        for obj in objects:
            # Find empty spaces near object
            bb = self._get_bounding_box(obj)
            expanded = self._expand_box(bb, 1)

            # Fill with pattern
            for i, j in expanded:
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
                    if result[i][j] == bg_color:
                        # Use a pattern color
                        pattern_color = (len(obj) % 8) + 1
                        result = self.fill(result, pattern_color, frozenset([(i, j)]))

        return result

    def _get_bounding_box(self, obj: Object) -> Set:
        """Get bounding box of object"""
        if not obj:
            return set()

        locs = [loc for _, loc in obj]
        min_i = min(i for i, j in locs)
        max_i = max(i for i, j in locs)
        min_j = min(j for i, j in locs)
        max_j = max(j for i, j in locs)

        box = set()
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                box.add((i, j))

        return box

    def _expand_box(self, box: Set, n: int) -> Set:
        """Expand bounding box by n pixels"""
        expanded = set()
        for i, j in box:
            for di in range(-n, n + 1):
                for dj in range(-n, n + 1):
                    expanded.add((i + di, j + dj))
        return expanded

    def mirror_objects(self, grid: Grid, axis: str = 'horizontal') -> Grid:
        """Mirror individual objects around their center"""
        objects = self.objects(grid, diagonal=True, without_bg=True)
        bg_color = self.mostcolor(grid)
        result = self.canvas(bg_color, (len(grid), len(grid[0])))

        for obj in objects:
            # Get center
            ci, cj = self.center(obj)

            # Mirror each cell
            mirrored = set()
            for val, (i, j) in obj:
                if axis == 'horizontal':
                    new_i = 2 * ci - i
                    new_j = j
                else:  # vertical
                    new_i = i
                    new_j = 2 * cj - j

                # Keep both original and mirrored
                mirrored.add((val, (i, j)))
                if 0 <= new_i < len(grid) and 0 <= new_j < len(grid[0]):
                    mirrored.add((val, (new_i, new_j)))

            result = self.paint(result, frozenset(mirrored))

        return result

    def complete_pattern(self, grid: Grid) -> Grid:
        """Try to complete partial patterns"""
        h, w = len(grid), len(grid[0])

        # Detect if grid has symmetry potential
        # Check horizontal symmetry
        half_h = h // 2
        if h > 2:
            top_half = grid[:half_h]
            bottom_half = grid[half_h:h]

            # If top is mostly filled and bottom mostly empty
            top_filled = sum(1 for row in top_half for v in row if v != 0)
            bottom_filled = sum(1 for row in bottom_half for v in row if v != 0)

            if top_filled > bottom_filled * 2:
                # Complete with horizontal mirror
                return self.vconcat(grid[:half_h], self.hmirror(grid[:half_h]))

        # Check vertical symmetry
        half_w = w // 2
        if w > 2:
            left_filled = sum(1 for row in grid for v in row[:half_w] if v != 0)
            right_filled = sum(1 for row in grid for v in row[half_w:] if v != 0)

            if left_filled > right_filled * 2:
                # Complete with vertical mirror
                left_part = self.crop(grid, (0, 0), (h, half_w))
                right_part = self.vmirror(left_part)
                return self.hconcat(left_part, right_part)

        return grid

    def propagate_pattern(self, grid: Grid) -> Grid:
        """Propagate small patterns across the grid"""
        h, w = len(grid), len(grid[0])

        # Find small repeating pattern (3x3 or smaller)
        for ph in [2, 3]:
            for pw in [2, 3]:
                if h >= ph * 2 and w >= pw * 2:
                    # Extract potential pattern
                    pattern = self.crop(grid, (0, 0), (ph, pw))

                    # Check if it appears elsewhere
                    pattern_obj = self.asobject(pattern)
                    occurrences = self.occurrences(grid, pattern_obj)

                    if len(occurrences) > 1:
                        # Tile the pattern
                        result = self.canvas(0, (h, w))
                        for i in range(0, h, ph):
                            for j in range(0, w, pw):
                                sub = self.crop(pattern, (0, 0),
                                              (min(ph, h - i), min(pw, w - j)))
                                for si, row in enumerate(sub):
                                    for sj, val in enumerate(row):
                                        if i + si < h and j + sj < w:
                                            result = self.fill(result, val,
                                                             frozenset([(i + si, j + sj)]))
                        return result

        return grid

    # =====================================
    # Learning-based transformations
    # =====================================

    def learn_color_mapping(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn consistent color mappings from examples"""
        mappings = {}

        for example in train_examples:
            in_grid = grid_from_list(example['input'])
            out_grid = grid_from_list(example['output'])

            if self.shape_matches(in_grid, out_grid):
                for i in range(len(in_grid)):
                    for j in range(len(in_grid[0])):
                        in_color = in_grid[i][j]
                        out_color = out_grid[i][j]

                        if in_color not in mappings:
                            mappings[in_color] = []
                        mappings[in_color].append(out_color)

        # Find consistent mappings
        consistent = {}
        for in_color, out_colors in mappings.items():
            if out_colors:
                most_common = Counter(out_colors).most_common(1)[0]
                if most_common[1] / len(out_colors) > 0.7:  # 70% consistency
                    consistent[in_color] = most_common[0]

        return consistent

    def apply_color_mapping(self, grid: Grid, mapping: Dict[int, int]) -> Grid:
        """Apply learned color mapping"""
        return tuple(
            tuple(mapping.get(v, v) for v in row)
            for row in grid
        )

    def shape_matches(self, g1: Grid, g2: Grid) -> bool:
        """Check if two grids have same shape"""
        return len(g1) == len(g2) and len(g1[0]) == len(g2[0])

    def detect_size_transform(self, train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Detect consistent size transformation"""
        ratios = []

        for example in train_examples:
            in_grid = grid_from_list(example['input'])
            out_grid = grid_from_list(example['output'])

            h_ratio = len(out_grid) / len(in_grid)
            w_ratio = len(out_grid[0]) / len(in_grid[0])

            ratios.append((h_ratio, w_ratio))

        # Check if all ratios are the same
        if len(set(ratios)) == 1:
            hr, wr = ratios[0]
            if hr == wr and hr in [2.0, 3.0, 0.5, 0.33]:
                return int(hr) if hr > 1 else None

        return None


class SmartProgramSearcher:
    """Smarter program search with learning"""

    def __init__(self):
        self.dsl = ImprovedDSL()
        self.primitives = self._get_all_primitives()
        print(f"Smart Searcher initialized with {len(self.primitives)} primitives")

    def _get_all_primitives(self) -> List[Tuple[str, Callable]]:
        """Get all primitives including new ones"""
        primitives = []

        # Basic transformations (from V1)
        primitives.extend([
            ('identity', lambda g: g),
            ('rot90', self.dsl.rot90),
            ('rot180', self.dsl.rot180),
            ('rot270', self.dsl.rot270),
            ('hmirror', self.dsl.hmirror),
            ('vmirror', self.dsl.vmirror),
            ('dmirror', self.dsl.dmirror),
        ])

        # Scaling
        for f in [2, 3]:
            primitives.append((f'upscale_{f}', lambda g, factor=f: self.dsl.upscale(g, factor)))
            primitives.append((f'downscale_{f}', lambda g, factor=f: self.dsl.downscale(g, factor)))

        # New object-based operations
        primitives.extend([
            ('color_by_size', self.dsl.color_by_size),
            ('connect_same_color', self.dsl.connect_same_color),
            ('fill_between_objects', self.dsl.fill_between_objects),
            ('mirror_objects_h', lambda g: self.dsl.mirror_objects(g, 'horizontal')),
            ('mirror_objects_v', lambda g: self.dsl.mirror_objects(g, 'vertical')),
            ('complete_pattern', self.dsl.complete_pattern),
            ('propagate_pattern', self.dsl.propagate_pattern),
        ])

        # Object extraction and transformation
        primitives.extend([
            ('extract_largest', self._extract_largest),
            ('extract_smallest', self._extract_smallest),
            ('remove_smallest', self._remove_smallest),
            ('isolate_colors', self._isolate_colors),
        ])

        return primitives

    def _extract_largest(self, grid: Grid) -> Grid:
        """Extract only the largest object"""
        objects = self.dsl.objects(grid, diagonal=True, without_bg=True)
        if not objects:
            return grid

        largest = max(objects, key=len)
        bg_color = self.dsl.mostcolor(grid)
        result = self.dsl.canvas(bg_color, (len(grid), len(grid[0])))
        return self.dsl.paint(result, largest)

    def _extract_smallest(self, grid: Grid) -> Grid:
        """Extract only the smallest object"""
        objects = self.dsl.objects(grid, diagonal=True, without_bg=True)
        if not objects:
            return grid

        smallest = min(objects, key=len)
        bg_color = self.dsl.mostcolor(grid)
        result = self.dsl.canvas(bg_color, (len(grid), len(grid[0])))
        return self.dsl.paint(result, smallest)

    def _remove_smallest(self, grid: Grid) -> Grid:
        """Remove the smallest object"""
        objects = self.dsl.objects(grid, diagonal=True, without_bg=True)
        if not objects:
            return grid

        smallest = min(objects, key=len)
        return self.dsl.cover(grid, smallest)

    def _isolate_colors(self, grid: Grid) -> Grid:
        """Isolate each color to separate regions"""
        h, w = len(grid), len(grid[0])
        colors = list(self.dsl.palette(grid) - {0})

        if len(colors) <= 1:
            return grid

        # Divide grid into regions
        region_width = w // len(colors)
        result = self.dsl.canvas(0, (h, w))

        for i, color in enumerate(colors):
            # Get all cells of this color
            indices = self.dsl.ofcolor(grid, color)

            # Place in its region
            for pi, pj in indices:
                new_j = i * region_width + (pj % region_width)
                if new_j < w:
                    result = self.dsl.fill(result, color, frozenset([(pi, new_j)]))

        return result

    def search_with_learning(self, task: Dict) -> List[List[List[int]]]:
        """Enhanced search with learning"""
        train_examples = task['train']

        print(f"\nSmart search for {len(train_examples)} training examples...")

        # Try to learn patterns first
        color_mapping = self.dsl.learn_color_mapping(train_examples)
        size_factor = self.dsl.detect_size_transform(train_examples)

        candidates = []

        # If we learned a color mapping, try it
        if color_mapping:
            print(f"  Learned color mapping: {color_mapping}")
            candidates.append(
                (['learned_color_map'],
                 lambda g: self.dsl.apply_color_mapping(g, color_mapping))
            )

        # If we detected size transform, try it
        if size_factor:
            print(f"  Detected size transform: {size_factor}x")
            candidates.append(
                ([f'upscale_{size_factor}'],
                 lambda g: self.dsl.upscale(g, size_factor))
            )

        # Try object-based operations
        for name, func in self.primitives:
            if 'object' in name or 'extract' in name or 'pattern' in name:
                score = self._evaluate_transform(func, train_examples)
                if score > 0.8:
                    print(f"  Found promising: {name} ({score:.1%})")
                    candidates.append(([name], func))

        # Try combinations if needed
        if not candidates or all(self._evaluate_transform(c[1], train_examples) < 1.0
                                for c in candidates):
            print("  Trying 2-step combinations...")
            candidates.extend(self._find_two_step(train_examples))

        # Generate predictions
        return self._generate_predictions(task['test'], candidates)

    def _evaluate_transform(self, transform: Callable, examples: List[Dict]) -> float:
        """Evaluate how well a transform works on training examples"""
        total_score = 0
        count = 0

        for example in examples:
            in_grid = grid_from_list(example['input'])
            expected = grid_from_list(example['output'])

            try:
                result = transform(in_grid)

                # Calculate similarity
                if len(result) == len(expected) and len(result[0]) == len(expected[0]):
                    matches = sum(1 for i in range(len(result))
                                for j in range(len(result[0]))
                                if result[i][j] == expected[i][j])
                    total = len(result) * len(result[0])
                    total_score += matches / total
                    count += 1
            except:
                pass

        return total_score / count if count > 0 else 0

    def _find_two_step(self, examples: List[Dict]) -> List:
        """Find promising 2-step combinations"""
        candidates = []

        # Focus on likely successful combinations
        promising_pairs = [
            ('extract_largest', 'upscale_2'),
            ('color_by_size', 'complete_pattern'),
            ('propagate_pattern', 'mirror_objects_h'),
            ('connect_same_color', 'fill_between_objects'),
        ]

        for name1, name2 in promising_pairs:
            func1 = next((f for n, f in self.primitives if n == name1), None)
            func2 = next((f for n, f in self.primitives if n == name2), None)

            if func1 and func2:
                combined = lambda g: func2(func1(g))
                score = self._evaluate_transform(combined, examples)

                if score > 0.7:
                    candidates.append(([name1, name2], combined))

        return candidates

    def _generate_predictions(self, test_examples: List,
                            candidates: List) -> List[List[List[int]]]:
        """Generate predictions using best candidates"""
        predictions = []

        for test_ex in test_examples:
            test_input = grid_from_list(test_ex['input'])
            attempts = []

            # Use top 2 candidates
            for names, transform in candidates[:2]:
                try:
                    result = transform(test_input)
                    attempts.append(grid_to_list(result))
                except:
                    attempts.append(grid_to_list(test_input))

            while len(attempts) < 2:
                attempts.append(grid_to_list(test_input))

            predictions.append(attempts)

        return predictions


# Test the improved version
if __name__ == "__main__":
    print("=" * 70)
    print("DSL V2 Improved - Test")
    print("=" * 70)

    # Load a sample task
    with open('data/arc-agi_training_challenges.json', 'r') as f:
        train_data = json.load(f)

    searcher = SmartProgramSearcher()

    # Test on first task
    task_id = list(train_data.keys())[0]
    task = train_data[task_id]

    predictions = searcher.search_with_learning(task)
    print(f"\nGenerated {len(predictions)} predictions")
    print("DSL V2 Ready!")