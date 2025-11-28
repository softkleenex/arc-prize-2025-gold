"""
DSL V3: Pattern-Based Operations
Based on analysis of 1000 training tasks
Key findings:
- 30.3% same-size transformations
- 57.9% involve color changes
- 56.8% involve object count increases
"""

import numpy as np
from typing import Tuple, List, Set, Dict, Callable, Optional, Any
from collections import defaultdict, Counter
from arc_dsl_v1 import ARCDSL, Grid, Objects, Object

class PatternDSL(ARCDSL):
    """Enhanced DSL with pattern-based operations"""

    def __init__(self):
        super().__init__()
        self.learned_patterns = {}

    # ============= ADVANCED COLOR LEARNING =============

    def learn_color_transform(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn color transformation from examples"""
        color_maps = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            # Build color mapping
            color_map = {}
            in_colors = set(v for row in in_grid for v in row)
            out_colors = set(v for row in out_grid for v in row)

            # Try direct mapping first
            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                for i in range(len(in_grid)):
                    for j in range(len(in_grid[0])):
                        if in_grid[i][j] != 0:
                            color_map[in_grid[i][j]] = out_grid[i][j]

            color_maps.append(color_map)

        # Find consistent mapping
        final_map = {}
        all_keys = set()
        for cm in color_maps:
            all_keys.update(cm.keys())

        for key in all_keys:
            values = [cm.get(key) for cm in color_maps if key in cm]
            if values:
                # Most common value
                final_map[key] = Counter(values).most_common(1)[0][0]

        return final_map

    def apply_learned_color(self, grid: Grid, color_map: Dict[int, int]) -> Grid:
        """Apply learned color transformation"""
        result = []
        for row in grid:
            new_row = []
            for val in row:
                new_row.append(color_map.get(val, val))
            result.append(tuple(new_row))
        return tuple(result)

    # ============= OBJECT REPLICATION =============

    def replicate_objects(self, grid: Grid, pattern: str = 'grid') -> Grid:
        """Replicate objects based on pattern"""
        objects = self.objects(grid)
        if not objects:
            return grid

        h, w = len(grid), len(grid[0])
        result = [[0] * w for _ in range(h)]

        if pattern == 'grid':
            # Arrange objects in grid pattern
            n_obj = len(objects)
            cols = int(np.sqrt(n_obj)) + 1
            rows = (n_obj + cols - 1) // cols

            for idx, obj in enumerate(objects):
                row_idx = idx // cols
                col_idx = idx % cols

                # Place object in grid position
                for (i, j), color in obj:
                    new_i = row_idx * 3 + i
                    new_j = col_idx * 3 + j
                    if 0 <= new_i < h and 0 <= new_j < w:
                        result[new_i][new_j] = color

        elif pattern == 'diagonal':
            # Place objects along diagonal
            for idx, obj in enumerate(objects):
                offset = idx * 3
                for (i, j), color in obj:
                    new_i = i + offset
                    new_j = j + offset
                    if 0 <= new_i < h and 0 <= new_j < w:
                        result[new_i][new_j] = color

        elif pattern == 'mirror':
            # Mirror objects
            for obj in objects:
                # Original position
                for (i, j), color in obj:
                    result[i][j] = color
                # Mirrored position
                for (i, j), color in obj:
                    mirror_j = w - 1 - j
                    if 0 <= mirror_j < w:
                        result[i][mirror_j] = color

        return tuple(tuple(row) for row in result)

    def multiply_objects(self, grid: Grid, factor: int = 2) -> Grid:
        """Multiply number of objects"""
        objects = self.objects(grid)
        if not objects:
            return grid

        h, w = len(grid), len(grid[0])

        # Calculate new grid size if needed
        new_h = h * factor
        new_w = w * factor

        result = [[0] * new_w for _ in range(new_h)]

        # Place original and copies
        for rep in range(factor * factor):
            offset_i = (rep // factor) * h
            offset_j = (rep % factor) * w

            for obj in objects:
                for (i, j), color in obj:
                    new_i = offset_i + i
                    new_j = offset_j + j
                    if new_i < new_h and new_j < new_w:
                        result[new_i][new_j] = color

        return tuple(tuple(row) for row in result)

    # ============= PATTERN DETECTION =============

    def detect_pattern_type(self, train_examples: List[Dict]) -> str:
        """Detect the type of pattern in examples"""
        patterns = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            h_in, w_in = len(in_grid), len(in_grid[0])
            h_out, w_out = len(out_grid), len(out_grid[0])

            # Check size relationship
            if h_in == h_out and w_in == w_out:
                patterns.append('same_size')
            elif h_out == 2 * h_in and w_out == 2 * w_in:
                patterns.append('scale_2x')
            elif h_out == 3 * h_in and w_out == 3 * w_in:
                patterns.append('scale_3x')
            elif h_out > h_in or w_out > w_in:
                patterns.append('expansion')
            else:
                patterns.append('reduction')

        # Return most common pattern
        if patterns:
            return Counter(patterns).most_common(1)[0][0]
        return 'unknown'

    # ============= OBJECT TRANSFORMATION =============

    def transform_by_property(self, grid: Grid, property_fn: Callable) -> Grid:
        """Transform objects based on their properties"""
        objects = self.objects(grid)
        if not objects:
            return grid

        h, w = len(grid), len(grid[0])
        result = [[0] * w for _ in range(h)]

        for obj in objects:
            # Calculate property
            prop = property_fn(obj)

            # Transform based on property
            for (i, j), color in obj:
                # Example: change color based on size
                new_color = (prop % 9) + 1  # Map to colors 1-9
                result[i][j] = new_color

        return tuple(tuple(row) for row in result)

    def color_by_position(self, grid: Grid) -> Grid:
        """Color objects based on their position"""
        objects = self.objects(grid)
        if not objects:
            return grid

        h, w = len(grid), len(grid[0])
        result = [[0] * w for _ in range(h)]

        # Sort objects by position
        sorted_objs = sorted(objects, key=lambda o: (min(p[0] for p, _ in o),
                                                    min(p[1] for p, _ in o)))

        for idx, obj in enumerate(sorted_objs):
            color = (idx % 9) + 1  # Cycle through colors 1-9
            for (i, j), _ in obj:
                result[i][j] = color

        return tuple(tuple(row) for row in result)

    # ============= PATTERN COMPLETION =============

    def complete_pattern(self, grid: Grid, direction: str = 'all') -> Grid:
        """Complete partial patterns"""
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        # Find non-zero pattern
        pattern_cells = [(i, j, grid[i][j]) for i in range(h)
                        for j in range(w) if grid[i][j] != 0]

        if not pattern_cells:
            return grid

        # Detect pattern period
        rows_with_data = set(i for i, _, _ in pattern_cells)
        cols_with_data = set(j for _, j, _ in pattern_cells)

        if len(rows_with_data) > 1:
            row_period = min(r2 - r1 for r1, r2 in zip(sorted(rows_with_data),
                                                       sorted(rows_with_data)[1:]))
        else:
            row_period = 1

        if len(cols_with_data) > 1:
            col_period = min(c2 - c1 for c1, c2 in zip(sorted(cols_with_data),
                                                       sorted(cols_with_data)[1:]))
        else:
            col_period = 1

        # Complete pattern
        for i, j, color in pattern_cells:
            if direction in ['all', 'horizontal']:
                # Extend horizontally
                for jj in range(j % col_period, w, col_period):
                    result[i][jj] = color

            if direction in ['all', 'vertical']:
                # Extend vertically
                for ii in range(i % row_period, h, row_period):
                    result[ii][j] = color

        return tuple(tuple(row) for row in result)

    # ============= ADVANCED COMPOSITION =============

    def compose_smart(self, train_examples: List[Dict]) -> Grid:
        """Smart composition based on pattern detection"""
        if not train_examples:
            return tuple()

        # Detect pattern type
        pattern_type = self.detect_pattern_type(train_examples)

        # Learn color mapping
        color_map = self.learn_color_transform(train_examples)

        # Get test input
        test_input = train_examples[0]['input']  # Placeholder

        # Apply transformation based on pattern
        if pattern_type == 'same_size':
            # Focus on color transformation and object manipulation
            result = self.apply_learned_color(test_input, color_map)
            objects = self.objects(result)

            if len(objects) > 1:
                # Try object-based transform
                result = self.color_by_position(result)

        elif pattern_type == 'scale_2x':
            result = self.upscale(test_input, 2)
            result = self.apply_learned_color(result, color_map)

        elif pattern_type == 'scale_3x':
            result = self.upscale(test_input, 3)
            result = self.apply_learned_color(result, color_map)

        elif pattern_type == 'expansion':
            # Try object replication
            result = self.replicate_objects(test_input, 'grid')

        else:
            # Default: try color mapping
            result = self.apply_learned_color(test_input, color_map)

        return result

    # ============= RULE LEARNING =============

    def learn_transformation_rule(self, train_examples: List[Dict]) -> Callable:
        """Learn the transformation rule from examples"""
        # Analyze transformations
        rules = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            # Check various rules
            if self.is_rotation(in_grid, out_grid):
                rules.append('rotate')
            elif self.is_mirror(in_grid, out_grid):
                rules.append('mirror')
            elif self.is_scaled(in_grid, out_grid):
                rules.append('scale')
            elif self.has_color_change(in_grid, out_grid):
                rules.append('color')
            else:
                rules.append('complex')

        # Return most common rule
        most_common = Counter(rules).most_common(1)[0][0]

        if most_common == 'rotate':
            return lambda g: self.rotate(g)
        elif most_common == 'mirror':
            return lambda g: self.hmirror(g)
        elif most_common == 'scale':
            factor = len(out_grid) // len(in_grid)
            return lambda g: self.upscale(g, factor)
        elif most_common == 'color':
            color_map = self.learn_color_transform(train_examples)
            return lambda g: self.apply_learned_color(g, color_map)
        else:
            # Complex transformation
            return lambda g: self.compose_smart(train_examples)

    def is_rotation(self, g1: Grid, g2: Grid) -> bool:
        """Check if g2 is rotation of g1"""
        rotations = [g1, self.rotate(g1)]
        for _ in range(2):
            rotations.append(self.rotate(rotations[-1]))
        return g2 in rotations

    def is_mirror(self, g1: Grid, g2: Grid) -> bool:
        """Check if g2 is mirror of g1"""
        return g2 in [self.hmirror(g1), self.vmirror(g1)]

    def is_scaled(self, g1: Grid, g2: Grid) -> bool:
        """Check if g2 is scaled version of g1"""
        h1, w1 = len(g1), len(g1[0])
        h2, w2 = len(g2), len(g2[0])

        if h2 % h1 == 0 and w2 % w1 == 0:
            factor = h2 // h1
            if factor == w2 // w1:
                return self.upscale(g1, factor) == g2
        return False

    def has_color_change(self, g1: Grid, g2: Grid) -> bool:
        """Check if there's a color change pattern"""
        colors1 = set(v for row in g1 for v in row)
        colors2 = set(v for row in g2 for v in row)
        return colors1 != colors2


class AdvancedProgramSearcher:
    """Advanced program synthesis with pattern-based DSL"""

    def __init__(self):
        self.dsl = PatternDSL()
        self.cache = {}

    def search(self, task: Dict, max_programs: int = 3) -> List[List]:
        """Search for programs that solve the task"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        if not train_examples:
            return [[[0]] for _ in test_examples]

        predictions = []

        for test_example in test_examples:
            test_input = test_example['input']
            attempts = []

            # Method 1: Learn transformation rule
            try:
                rule = self.dsl.learn_transformation_rule(train_examples)
                result = rule(test_input)
                if result:
                    attempts.append(result)
            except:
                pass

            # Method 2: Smart composition
            try:
                # Prepare examples with test input
                examples_with_test = train_examples.copy()
                examples_with_test[0]['input'] = test_input
                result = self.dsl.compose_smart(examples_with_test)
                if result and result not in attempts:
                    attempts.append(result)
            except:
                pass

            # Method 3: Pattern-specific transformations
            try:
                pattern_type = self.dsl.detect_pattern_type(train_examples)

                if pattern_type == 'same_size':
                    # Try various same-size operations
                    color_map = self.dsl.learn_color_transform(train_examples)
                    result = self.dsl.apply_learned_color(test_input, color_map)
                    if result and result not in attempts:
                        attempts.append(result)

                    # Try object coloring
                    result = self.dsl.color_by_position(test_input)
                    if result and result not in attempts:
                        attempts.append(result)

                elif 'scale' in pattern_type:
                    factor = 2 if '2x' in pattern_type else 3
                    result = self.dsl.upscale(test_input, factor)
                    color_map = self.dsl.learn_color_transform(train_examples)
                    result = self.dsl.apply_learned_color(result, color_map)
                    if result and result not in attempts:
                        attempts.append(result)

                elif pattern_type == 'expansion':
                    result = self.dsl.replicate_objects(test_input, 'grid')
                    if result and result not in attempts:
                        attempts.append(result)
            except:
                pass

            # Ensure we have required number of attempts
            while len(attempts) < max_programs:
                attempts.append([[0]])

            predictions.append(attempts[:max_programs])

        return predictions