"""
Hybrid Solver: Example-Based Learning + Pattern Matching
Goal: Achieve perfect pixel accuracy
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import Counter, defaultdict
from arc_dsl_v1 import ARCDSL
from dsl_v3_pattern import PatternDSL


class HybridSolver:
    """Hybrid approach combining multiple strategies"""

    def __init__(self):
        self.basic_dsl = ARCDSL()
        self.pattern_dsl = PatternDSL()

    def solve(self, task: Dict) -> List[List]:
        """Main solving interface"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        if not train_examples:
            return [[[0]] for _ in test_examples]

        predictions = []
        for test_example in test_examples:
            test_input = test_example['input']
            attempts = self.generate_attempts(train_examples, test_input)
            predictions.append(attempts[:2])  # Return top 2 attempts

        return predictions

    def generate_attempts(self, train_examples: List[Dict], test_input: List[List]) -> List:
        """Generate multiple solution attempts"""
        attempts = []

        # Strategy 1: Direct pattern matching
        try:
            result = self.direct_pattern_match(train_examples, test_input)
            if result and self.validate_against_examples(train_examples, result):
                attempts.append(result)
        except:
            pass

        # Strategy 2: Transform-based
        try:
            result = self.transform_based(train_examples, test_input)
            if result and result not in attempts:
                attempts.append(result)
        except:
            pass

        # Strategy 3: Example interpolation
        try:
            result = self.example_interpolation(train_examples, test_input)
            if result and result not in attempts:
                attempts.append(result)
        except:
            pass

        # Strategy 4: Rule extraction
        try:
            result = self.rule_based(train_examples, test_input)
            if result and result not in attempts:
                attempts.append(result)
        except:
            pass

        # Ensure we have at least 2 attempts
        while len(attempts) < 2:
            attempts.append([[0]])

        return attempts

    def direct_pattern_match(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Direct pattern matching from examples"""
        # Find most similar training input
        best_match_idx = -1
        best_similarity = 0

        for idx, example in enumerate(train_examples):
            train_in = example['input']
            similarity = self.calculate_similarity(train_in, test_input)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        if best_match_idx >= 0 and best_similarity > 0.8:
            # Apply same transformation
            reference = train_examples[best_match_idx]
            return self.apply_same_transform(reference, test_input)

        return None

    def transform_based(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Identify and apply transformation pattern"""
        # Detect transformation type
        transform_type = self.detect_transform_type(train_examples)

        if transform_type == 'color_map':
            color_map = self.learn_exact_color_map(train_examples)
            return self.apply_color_map(test_input, color_map)

        elif transform_type == 'scale':
            factor = self.detect_scale_factor(train_examples)
            result = self.pattern_dsl.upscale(test_input, factor)
            # Apply color transformation if needed
            color_map = self.learn_exact_color_map(train_examples)
            if color_map:
                result = self.apply_color_map(result, color_map)
            return result

        elif transform_type == 'rotate':
            angle = self.detect_rotation_angle(train_examples)
            for _ in range(angle // 90):
                test_input = self.pattern_dsl.rotate(test_input)
            return test_input

        elif transform_type == 'mirror':
            direction = self.detect_mirror_direction(train_examples)
            if direction == 'horizontal':
                return self.pattern_dsl.hmirror(test_input)
            else:
                return self.pattern_dsl.vmirror(test_input)

        elif transform_type == 'object_based':
            return self.object_based_transform(train_examples, test_input)

        return None

    def example_interpolation(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Interpolate solution from similar examples"""
        # Find patterns in input-output relationships
        patterns = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            # Extract pattern features
            pattern = {
                'in_shape': (len(in_grid), len(in_grid[0])),
                'out_shape': (len(out_grid), len(out_grid[0])),
                'in_colors': set(v for row in in_grid for v in row),
                'out_colors': set(v for row in out_grid for v in row),
                'transform': self.identify_transform(in_grid, out_grid)
            }
            patterns.append(pattern)

        # Find consensus pattern
        test_shape = (len(test_input), len(test_input[0]))
        test_colors = set(v for row in test_input for v in row)

        # Match pattern
        for pattern in patterns:
            if pattern['in_shape'] == test_shape:
                # Apply same size relationship
                if pattern['transform']:
                    return self.apply_transform(test_input, pattern['transform'])

        return None

    def rule_based(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Extract and apply rules from examples"""
        # Extract rules
        rules = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            # Check for specific rules
            if self.is_fill_rule(in_grid, out_grid):
                rules.append(('fill', self.extract_fill_params(in_grid, out_grid)))
            elif self.is_copy_rule(in_grid, out_grid):
                rules.append(('copy', self.extract_copy_params(in_grid, out_grid)))
            elif self.is_pattern_rule(in_grid, out_grid):
                rules.append(('pattern', self.extract_pattern_params(in_grid, out_grid)))

        # Apply most common rule
        if rules:
            rule_type = Counter(r[0] for r in rules).most_common(1)[0][0]

            if rule_type == 'fill':
                return self.apply_fill_rule(test_input, rules[0][1])
            elif rule_type == 'copy':
                return self.apply_copy_rule(test_input, rules[0][1])
            elif rule_type == 'pattern':
                return self.apply_pattern_rule(test_input, rules[0][1])

        return None

    # ============= Helper Methods =============

    def calculate_similarity(self, grid1: List[List], grid2: List[List]) -> float:
        """Calculate similarity between two grids"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return 0.0

        matches = 0
        total = len(grid1) * len(grid1[0])

        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] == grid2[i][j]:
                    matches += 1

        return matches / total

    def validate_against_examples(self, train_examples: List[Dict], result: List[List]) -> bool:
        """Validate if result follows pattern of training examples"""
        for example in train_examples:
            out_grid = example['output']

            # Check if result has similar properties
            if len(result) == len(out_grid) and len(result[0]) == len(out_grid[0]):
                # Check color distribution
                result_colors = Counter(v for row in result for v in row)
                example_colors = Counter(v for row in out_grid for v in row)

                # Similar color distribution?
                color_similarity = sum(min(result_colors[c], example_colors[c])
                                      for c in set(result_colors) & set(example_colors))
                total_colors = sum(result_colors.values())

                if color_similarity / total_colors > 0.7:
                    return True

        return False

    def apply_same_transform(self, reference: Dict, test_input: List[List]) -> List[List]:
        """Apply the same transformation as in reference example"""
        ref_in = reference['input']
        ref_out = reference['output']

        # Detect transformation
        if len(ref_in) == len(ref_out) and len(ref_in[0]) == len(ref_out[0]):
            # Same size - likely color mapping
            color_map = {}
            for i in range(len(ref_in)):
                for j in range(len(ref_in[0])):
                    if ref_in[i][j] != 0:
                        color_map[ref_in[i][j]] = ref_out[i][j]

            # Apply to test
            result = []
            for row in test_input:
                new_row = []
                for val in row:
                    new_row.append(color_map.get(val, val))
                result.append(new_row)
            return result

        return test_input

    def detect_transform_type(self, train_examples: List[Dict]) -> str:
        """Detect the type of transformation"""
        types = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            h_in, w_in = len(in_grid), len(in_grid[0])
            h_out, w_out = len(out_grid), len(out_grid[0])

            if h_in == h_out and w_in == w_out:
                # Check for color mapping
                in_colors = set(v for row in in_grid for v in row)
                out_colors = set(v for row in out_grid for v in row)

                if in_colors != out_colors:
                    types.append('color_map')
                elif self.pattern_dsl.is_rotation(in_grid, out_grid):
                    types.append('rotate')
                elif self.pattern_dsl.is_mirror(in_grid, out_grid):
                    types.append('mirror')
                else:
                    types.append('object_based')
            elif h_out == 2 * h_in and w_out == 2 * w_in:
                types.append('scale')
            else:
                types.append('complex')

        # Return most common
        if types:
            return Counter(types).most_common(1)[0][0]
        return 'unknown'

    def learn_exact_color_map(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn exact color mapping from examples"""
        all_mappings = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                mapping = {}
                for i in range(len(in_grid)):
                    for j in range(len(in_grid[0])):
                        if in_grid[i][j] != 0:
                            mapping[in_grid[i][j]] = out_grid[i][j]
                all_mappings.append(mapping)

        # Find consistent mapping
        final_map = {}
        if all_mappings:
            all_keys = set()
            for m in all_mappings:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [m.get(key) for m in all_mappings if key in m]
                if values:
                    # Most common value
                    final_map[key] = Counter(values).most_common(1)[0][0]

        return final_map

    def apply_color_map(self, grid: List[List], color_map: Dict[int, int]) -> List[List]:
        """Apply color mapping to grid"""
        result = []
        for row in grid:
            new_row = []
            for val in row:
                new_row.append(color_map.get(val, val))
            result.append(new_row)
        return result

    def detect_scale_factor(self, train_examples: List[Dict]) -> int:
        """Detect scaling factor"""
        factors = []

        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            h_in, w_in = len(in_grid), len(in_grid[0])
            h_out, w_out = len(out_grid), len(out_grid[0])

            if h_out % h_in == 0 and w_out % w_in == 0:
                factor_h = h_out // h_in
                factor_w = w_out // w_in
                if factor_h == factor_w:
                    factors.append(factor_h)

        if factors:
            return Counter(factors).most_common(1)[0][0]
        return 1

    def detect_rotation_angle(self, train_examples: List[Dict]) -> int:
        """Detect rotation angle"""
        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            # Check 90, 180, 270 degree rotations
            temp = in_grid
            for angle in [90, 180, 270]:
                temp = self.pattern_dsl.rotate(temp)
                if temp == out_grid:
                    return angle

        return 0

    def detect_mirror_direction(self, train_examples: List[Dict]) -> str:
        """Detect mirror direction"""
        for example in train_examples:
            in_grid = example['input']
            out_grid = example['output']

            if self.pattern_dsl.hmirror(in_grid) == out_grid:
                return 'horizontal'
            if self.pattern_dsl.vmirror(in_grid) == out_grid:
                return 'vertical'

        return 'none'

    def object_based_transform(self, train_examples: List[Dict], test_input: List[List]) -> List[List]:
        """Apply object-based transformation"""
        # Extract objects and apply transformation
        objects = self.pattern_dsl.objects(test_input)

        if not objects:
            return test_input

        # Learn object transformation from examples
        for example in train_examples:
            in_objects = self.pattern_dsl.objects(example['input'])
            out_objects = self.pattern_dsl.objects(example['output'])

            if len(in_objects) == len(out_objects):
                # Apply similar transformation
                return self.pattern_dsl.color_by_position(test_input)

        return test_input

    def identify_transform(self, in_grid: List[List], out_grid: List[List]) -> Optional[str]:
        """Identify transformation between grids"""
        if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
            if self.pattern_dsl.is_rotation(in_grid, out_grid):
                return 'rotate'
            elif self.pattern_dsl.is_mirror(in_grid, out_grid):
                return 'mirror'
            else:
                return 'color'
        elif len(out_grid) > len(in_grid):
            return 'scale'
        else:
            return 'complex'

    def apply_transform(self, grid: List[List], transform: str) -> List[List]:
        """Apply identified transform"""
        if transform == 'rotate':
            return self.pattern_dsl.rotate(grid)
        elif transform == 'mirror':
            return self.pattern_dsl.hmirror(grid)
        elif transform == 'scale':
            return self.pattern_dsl.upscale(grid, 2)
        else:
            return grid

    def is_fill_rule(self, in_grid: List[List], out_grid: List[List]) -> bool:
        """Check if it's a fill rule"""
        # Simple heuristic: output has more of a certain color
        in_colors = Counter(v for row in in_grid for v in row)
        out_colors = Counter(v for row in out_grid for v in row)

        for color, count in out_colors.items():
            if color != 0 and count > in_colors.get(color, 0) * 2:
                return True
        return False

    def is_copy_rule(self, in_grid: List[List], out_grid: List[List]) -> bool:
        """Check if it's a copy rule"""
        # Check if output contains repeated input
        h_in, w_in = len(in_grid), len(in_grid[0])
        h_out, w_out = len(out_grid), len(out_grid[0])

        if h_out >= h_in * 2 or w_out >= w_in * 2:
            return True
        return False

    def is_pattern_rule(self, in_grid: List[List], out_grid: List[List]) -> bool:
        """Check if it's a pattern completion rule"""
        # Check for regular patterns in output
        return self.has_regular_pattern(out_grid)

    def has_regular_pattern(self, grid: List[List]) -> bool:
        """Check if grid has regular pattern"""
        # Simple check for repeating rows or columns
        h, w = len(grid), len(grid[0])

        # Check rows
        for period in range(1, h // 2 + 1):
            matches = True
            for i in range(period, h, period):
                if i + period <= h:
                    if grid[i:i+period] != grid[0:period]:
                        matches = False
                        break
            if matches:
                return True

        return False

    def extract_fill_params(self, in_grid: List[List], out_grid: List[List]) -> Dict:
        """Extract fill parameters"""
        return {'type': 'fill'}

    def extract_copy_params(self, in_grid: List[List], out_grid: List[List]) -> Dict:
        """Extract copy parameters"""
        h_in, w_in = len(in_grid), len(in_grid[0])
        h_out, w_out = len(out_grid), len(out_grid[0])

        return {
            'type': 'copy',
            'factor_h': h_out // h_in if h_in > 0 else 1,
            'factor_w': w_out // w_in if w_in > 0 else 1
        }

    def extract_pattern_params(self, in_grid: List[List], out_grid: List[List]) -> Dict:
        """Extract pattern parameters"""
        return {'type': 'pattern'}

    def apply_fill_rule(self, grid: List[List], params: Dict) -> List[List]:
        """Apply fill rule"""
        # Simple flood fill implementation
        result = [list(row) for row in grid]

        # Find largest connected component and fill
        objects = self.pattern_dsl.objects(grid)
        if objects:
            largest = max(objects, key=len)
            color = max(set(c for _, c in largest), key=lambda c: sum(1 for _, cc in largest if cc == c))

            # Fill background with this color
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] == 0:
                        result[i][j] = color

        return result

    def apply_copy_rule(self, grid: List[List], params: Dict) -> List[List]:
        """Apply copy rule"""
        factor_h = params.get('factor_h', 2)
        factor_w = params.get('factor_w', 2)

        h, w = len(grid), len(grid[0])
        result = [[0] * (w * factor_w) for _ in range(h * factor_h)]

        for rep_h in range(factor_h):
            for rep_w in range(factor_w):
                for i in range(h):
                    for j in range(w):
                        result[rep_h * h + i][rep_w * w + j] = grid[i][j]

        return result

    def apply_pattern_rule(self, grid: List[List], params: Dict) -> List[List]:
        """Apply pattern completion rule"""
        return self.pattern_dsl.complete_pattern(grid, 'all')