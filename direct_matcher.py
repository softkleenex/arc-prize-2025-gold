"""
Direct Example Matcher: Focus on 100% pixel-perfect accuracy
Key strategy: Learn exact transformations from examples
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import itertools


class DirectMatcher:
    """Direct matching approach for pixel-perfect accuracy"""

    def __init__(self):
        self.debug = False

    def solve(self, task: Dict) -> List[List]:
        """Main solving interface"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        if not train_examples:
            return [[[0]] for _ in test_examples]

        predictions = []
        for test_example in test_examples:
            test_input = test_example['input']
            attempts = self.generate_perfect_attempts(train_examples, test_input)
            predictions.append(attempts[:2])

        return predictions

    def generate_perfect_attempts(self, train_examples: List[Dict], test_input: List[List]) -> List:
        """Generate attempts aiming for perfect accuracy"""
        attempts = []

        # Strategy 1: Exact pattern replication
        result = self.exact_pattern_replication(train_examples, test_input)
        if result is not None:
            attempts.append(result)

        # Strategy 2: Deterministic transformation
        result = self.deterministic_transformation(train_examples, test_input)
        if result is not None and result not in attempts:
            attempts.append(result)

        # Strategy 3: Consensus-based
        result = self.consensus_based_solution(train_examples, test_input)
        if result is not None and result not in attempts:
            attempts.append(result)

        # Strategy 4: Template matching
        result = self.template_matching(train_examples, test_input)
        if result is not None and result not in attempts:
            attempts.append(result)

        # Ensure we have at least 2 attempts
        while len(attempts) < 2:
            # Try variations
            if attempts:
                attempts.append(self.create_variation(attempts[0]))
            else:
                attempts.append(test_input)  # Fallback to input

        return attempts

    def exact_pattern_replication(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Replicate exact pattern from training examples"""

        # Check if all training outputs are identical
        outputs = [tuple(tuple(row) for row in ex['output']) for ex in train_examples]
        if len(set(outputs)) == 1:
            # All outputs are the same - return it
            return list(list(row) for row in outputs[0])

        # Check if there's a direct input->output mapping
        input_output_map = {}
        for ex in train_examples:
            in_key = tuple(tuple(row) for row in ex['input'])
            out_val = tuple(tuple(row) for row in ex['output'])
            input_output_map[in_key] = out_val

        test_key = tuple(tuple(row) for row in test_input)
        if test_key in input_output_map:
            return list(list(row) for row in input_output_map[test_key])

        # Check for size-based patterns
        test_h, test_w = len(test_input), len(test_input[0])

        for ex in train_examples:
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])

            if in_h == test_h and in_w == test_w:
                # Same input size - check if transformation is consistent
                if out_h == in_h and out_w == in_w:
                    # Same size transformation
                    return self.apply_pixel_mapping(ex, test_input)
                else:
                    # Size change - apply same ratio
                    return self.apply_size_transform(ex, test_input)

        return None

    def deterministic_transformation(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Apply deterministic transformation learned from examples"""

        # Learn transformation rules
        rules = []

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            # Identify transformation type
            rule = self.identify_exact_rule(in_grid, out_grid)
            if rule:
                rules.append(rule)

        if not rules:
            return None

        # Find most common rule
        rule_types = [r['type'] for r in rules]
        most_common = Counter(rule_types).most_common(1)[0][0]

        # Apply the rule
        for rule in rules:
            if rule['type'] == most_common:
                return self.apply_exact_rule(test_input, rule)

        return None

    def consensus_based_solution(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Generate solution based on consensus from all examples"""

        # Collect all transformations
        transformations = []

        for ex in train_examples:
            transform = self.extract_transformation(ex['input'], ex['output'])
            if transform:
                transformations.append(transform)

        if not transformations:
            return None

        # Find consensus
        consensus = self.find_consensus(transformations)

        if consensus:
            return self.apply_consensus(test_input, consensus)

        return None

    def template_matching(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Match against templates from training examples"""

        # Extract templates from outputs
        templates = []
        for ex in train_examples:
            template = self.extract_template(ex['output'])
            templates.append(template)

        # Find matching template for test input
        test_features = self.extract_features(test_input)

        best_match = None
        best_score = 0

        for ex in train_examples:
            in_features = self.extract_features(ex['input'])
            similarity = self.feature_similarity(in_features, test_features)

            if similarity > best_score:
                best_score = similarity
                best_match = ex

        if best_match and best_score > 0.9:
            # Apply same transformation
            return self.apply_exact_transform(best_match, test_input)

        return None

    def apply_pixel_mapping(self, reference: Dict, test_input: List[List]) -> List[List]:
        """Apply pixel-level mapping from reference"""
        ref_in = reference['input']
        ref_out = reference['output']

        # Build pixel mapping
        pixel_map = {}
        position_map = {}

        h, w = len(ref_in), len(ref_in[0])

        for i in range(h):
            for j in range(w):
                in_val = ref_in[i][j]
                out_val = ref_out[i][j]

                # Color mapping
                if in_val not in pixel_map:
                    pixel_map[in_val] = out_val

                # Position-based mapping
                position_map[(i, j)] = out_val

        # Apply to test input
        result = []
        for i, row in enumerate(test_input):
            new_row = []
            for j, val in enumerate(row):
                # Try position-based first
                if (i, j) in position_map and i < h and j < w:
                    new_row.append(position_map[(i, j)])
                # Then color-based
                elif val in pixel_map:
                    new_row.append(pixel_map[val])
                else:
                    new_row.append(val)
            result.append(new_row)

        return result

    def apply_size_transform(self, reference: Dict, test_input: List[List]) -> List[List]:
        """Apply size transformation from reference"""
        ref_in = reference['input']
        ref_out = reference['output']

        in_h, in_w = len(ref_in), len(ref_in[0])
        out_h, out_w = len(ref_out), len(ref_out[0])

        # Calculate ratios
        h_ratio = out_h / in_h if in_h > 0 else 1
        w_ratio = out_w / in_w if in_w > 0 else 1

        test_h, test_w = len(test_input), len(test_input[0])
        new_h = int(test_h * h_ratio)
        new_w = int(test_w * w_ratio)

        # Create output grid
        result = [[0] * new_w for _ in range(new_h)]

        # Apply transformation
        if h_ratio >= 1 and w_ratio >= 1:
            # Upscaling
            h_scale = int(h_ratio)
            w_scale = int(w_ratio)

            for i in range(test_h):
                for j in range(test_w):
                    val = test_input[i][j]
                    for di in range(h_scale):
                        for dj in range(w_scale):
                            if i * h_scale + di < new_h and j * w_scale + dj < new_w:
                                result[i * h_scale + di][j * w_scale + dj] = val
        else:
            # Downscaling or complex
            for i in range(new_h):
                for j in range(new_w):
                    src_i = min(int(i / h_ratio), test_h - 1)
                    src_j = min(int(j / w_ratio), test_w - 1)
                    result[i][j] = test_input[src_i][src_j]

        # Apply color mapping if exists
        color_map = self.extract_color_map(ref_in, ref_out)
        if color_map:
            for i in range(new_h):
                for j in range(new_w):
                    if result[i][j] in color_map:
                        result[i][j] = color_map[result[i][j]]

        return result

    def identify_exact_rule(self, in_grid: List[List], out_grid: List[List]) -> Optional[Dict]:
        """Identify exact transformation rule"""
        h_in, w_in = len(in_grid), len(in_grid[0])
        h_out, w_out = len(out_grid), len(out_grid[0])

        # Check for rotation
        rotations = [in_grid]
        temp = in_grid
        for angle in [90, 180, 270]:
            temp = self.rotate_90(temp)
            rotations.append(temp)
            if self.grids_equal(temp, out_grid):
                return {'type': 'rotate', 'angle': angle}

        # Check for mirror
        if self.grids_equal(self.mirror_horizontal(in_grid), out_grid):
            return {'type': 'mirror', 'direction': 'horizontal'}
        if self.grids_equal(self.mirror_vertical(in_grid), out_grid):
            return {'type': 'mirror', 'direction': 'vertical'}

        # Check for scaling
        if h_out == 2 * h_in and w_out == 2 * w_in:
            return {'type': 'scale', 'factor': 2}
        elif h_out == 3 * h_in and w_out == 3 * w_in:
            return {'type': 'scale', 'factor': 3}

        # Check for color change
        if h_in == h_out and w_in == w_out:
            color_map = self.extract_color_map(in_grid, out_grid)
            if color_map:
                return {'type': 'color', 'map': color_map}

        return None

    def apply_exact_rule(self, grid: List[List], rule: Dict) -> List[List]:
        """Apply exact rule to grid"""
        rule_type = rule['type']

        if rule_type == 'rotate':
            result = grid
            for _ in range(rule['angle'] // 90):
                result = self.rotate_90(result)
            return result

        elif rule_type == 'mirror':
            if rule['direction'] == 'horizontal':
                return self.mirror_horizontal(grid)
            else:
                return self.mirror_vertical(grid)

        elif rule_type == 'scale':
            return self.scale_grid(grid, rule['factor'])

        elif rule_type == 'color':
            return self.apply_color_mapping(grid, rule['map'])

        return grid

    def extract_transformation(self, in_grid: List[List], out_grid: List[List]) -> Dict:
        """Extract transformation details"""
        return {
            'input_shape': (len(in_grid), len(in_grid[0])),
            'output_shape': (len(out_grid), len(out_grid[0])),
            'color_map': self.extract_color_map(in_grid, out_grid),
            'pattern': self.extract_pattern(in_grid, out_grid)
        }

    def find_consensus(self, transformations: List[Dict]) -> Dict:
        """Find consensus among transformations"""
        if not transformations:
            return {}

        # Check if all have same output shape
        shapes = [t['output_shape'] for t in transformations]
        if len(set(shapes)) == 1:
            # Consensus on shape
            consensus = {'output_shape': shapes[0]}

            # Check color maps
            color_maps = [t['color_map'] for t in transformations if t['color_map']]
            if color_maps:
                # Find common mappings
                common_map = {}
                all_keys = set()
                for cm in color_maps:
                    all_keys.update(cm.keys())

                for key in all_keys:
                    values = [cm.get(key) for cm in color_maps if key in cm]
                    if values and len(set(values)) == 1:
                        common_map[key] = values[0]

                if common_map:
                    consensus['color_map'] = common_map

            return consensus

        return {}

    def apply_consensus(self, grid: List[List], consensus: Dict) -> List[List]:
        """Apply consensus transformation"""
        if 'output_shape' in consensus:
            out_h, out_w = consensus['output_shape']
            in_h, in_w = len(grid), len(grid[0])

            # Create output grid
            if out_h == in_h and out_w == in_w:
                result = [list(row) for row in grid]
            else:
                # Scale to match
                result = self.scale_to_size(grid, out_h, out_w)

            # Apply color map if exists
            if 'color_map' in consensus:
                color_map = consensus['color_map']
                for i in range(len(result)):
                    for j in range(len(result[0])):
                        if result[i][j] in color_map:
                            result[i][j] = color_map[result[i][j]]

            return result

        return grid

    def extract_template(self, grid: List[List]) -> Dict:
        """Extract template from grid"""
        return {
            'shape': (len(grid), len(grid[0])),
            'colors': set(v for row in grid for v in row),
            'pattern': self.detect_pattern(grid)
        }

    def extract_features(self, grid: List[List]) -> Dict:
        """Extract features from grid"""
        h, w = len(grid), len(grid[0])
        colors = [v for row in grid for v in row]

        return {
            'shape': (h, w),
            'colors': set(colors),
            'color_counts': Counter(colors),
            'density': sum(1 for v in colors if v != 0) / (h * w),
            'objects': self.count_objects(grid)
        }

    def feature_similarity(self, f1: Dict, f2: Dict) -> float:
        """Calculate feature similarity"""
        score = 0.0

        # Shape similarity
        if f1['shape'] == f2['shape']:
            score += 0.3

        # Color similarity
        common_colors = len(f1['colors'] & f2['colors'])
        total_colors = len(f1['colors'] | f2['colors'])
        if total_colors > 0:
            score += 0.3 * (common_colors / total_colors)

        # Density similarity
        density_diff = abs(f1['density'] - f2['density'])
        score += 0.2 * (1 - density_diff)

        # Object count similarity
        if f1['objects'] == f2['objects']:
            score += 0.2

        return score

    def apply_exact_transform(self, reference: Dict, test_input: List[List]) -> List[List]:
        """Apply exact transformation from reference"""
        ref_in = reference['input']
        ref_out = reference['output']

        # Determine exact transformation
        transform = self.determine_exact_transform(ref_in, ref_out)

        # Apply to test input
        result = test_input
        for step in transform:
            if step['type'] == 'resize':
                result = self.scale_to_size(result, step['h'], step['w'])
            elif step['type'] == 'color':
                result = self.apply_color_mapping(result, step['map'])
            elif step['type'] == 'rotate':
                result = self.rotate_90(result)
            elif step['type'] == 'mirror_h':
                result = self.mirror_horizontal(result)
            elif step['type'] == 'mirror_v':
                result = self.mirror_vertical(result)

        return result

    def create_variation(self, grid: List[List]) -> List[List]:
        """Create a slight variation of grid"""
        # Just return a copy for now
        return [list(row) for row in grid]

    # ============= Utility Methods =============

    def rotate_90(self, grid: List[List]) -> List[List]:
        """Rotate grid 90 degrees clockwise"""
        return [list(row) for row in zip(*grid[::-1])]

    def mirror_horizontal(self, grid: List[List]) -> List[List]:
        """Mirror grid horizontally"""
        return grid[::-1]

    def mirror_vertical(self, grid: List[List]) -> List[List]:
        """Mirror grid vertically"""
        return [row[::-1] for row in grid]

    def scale_grid(self, grid: List[List], factor: int) -> List[List]:
        """Scale grid by integer factor"""
        h, w = len(grid), len(grid[0])
        result = [[0] * (w * factor) for _ in range(h * factor)]

        for i in range(h):
            for j in range(w):
                val = grid[i][j]
                for di in range(factor):
                    for dj in range(factor):
                        result[i * factor + di][j * factor + dj] = val

        return result

    def scale_to_size(self, grid: List[List], new_h: int, new_w: int) -> List[List]:
        """Scale grid to specific size"""
        h, w = len(grid), len(grid[0])

        if h == new_h and w == new_w:
            return [list(row) for row in grid]

        result = [[0] * new_w for _ in range(new_h)]

        # Simple nearest neighbor scaling
        for i in range(new_h):
            for j in range(new_w):
                src_i = min(i * h // new_h, h - 1)
                src_j = min(j * w // new_w, w - 1)
                result[i][j] = grid[src_i][src_j]

        return result

    def grids_equal(self, g1: List[List], g2: List[List]) -> bool:
        """Check if two grids are equal"""
        if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
            return False

        for i in range(len(g1)):
            for j in range(len(g1[0])):
                if g1[i][j] != g2[i][j]:
                    return False
        return True

    def extract_color_map(self, in_grid: List[List], out_grid: List[List]) -> Dict[int, int]:
        """Extract color mapping"""
        if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]):
            return {}

        color_map = {}
        for i in range(len(in_grid)):
            for j in range(len(in_grid[0])):
                in_val = in_grid[i][j]
                out_val = out_grid[i][j]
                if in_val != 0:
                    if in_val not in color_map:
                        color_map[in_val] = out_val
                    elif color_map[in_val] != out_val:
                        # Inconsistent mapping
                        return {}

        return color_map

    def apply_color_mapping(self, grid: List[List], color_map: Dict[int, int]) -> List[List]:
        """Apply color mapping"""
        result = []
        for row in grid:
            new_row = []
            for val in row:
                new_row.append(color_map.get(val, val))
            result.append(new_row)
        return result

    def extract_pattern(self, in_grid: List[List], out_grid: List[List]) -> str:
        """Extract pattern type"""
        # Simple pattern detection
        if self.grids_equal(in_grid, out_grid):
            return 'identity'
        elif len(out_grid) > len(in_grid):
            return 'expansion'
        elif len(out_grid) < len(in_grid):
            return 'reduction'
        else:
            return 'transformation'

    def detect_pattern(self, grid: List[List]) -> str:
        """Detect pattern in grid"""
        # Simple pattern detection
        h, w = len(grid), len(grid[0])

        # Check for uniform
        first = grid[0][0]
        if all(grid[i][j] == first for i in range(h) for j in range(w)):
            return 'uniform'

        # Check for checkerboard
        is_checker = True
        for i in range(h):
            for j in range(w):
                expected = grid[0][0] if (i + j) % 2 == 0 else grid[0][1]
                if grid[i][j] != expected:
                    is_checker = False
                    break
            if not is_checker:
                break

        if is_checker:
            return 'checkerboard'

        return 'complex'

    def count_objects(self, grid: List[List]) -> int:
        """Count connected components"""
        h, w = len(grid), len(grid[0])
        visited = set()
        count = 0

        def dfs(i, j, color):
            if (i, j) in visited or i < 0 or i >= h or j < 0 or j >= w:
                return
            if grid[i][j] != color or grid[i][j] == 0:
                return
            visited.add((i, j))
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj, color)

        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i][j] != 0:
                    count += 1
                    dfs(i, j, grid[i][j])

        return count

    def determine_exact_transform(self, in_grid: List[List], out_grid: List[List]) -> List[Dict]:
        """Determine exact sequence of transformations"""
        transforms = []

        h_in, w_in = len(in_grid), len(in_grid[0])
        h_out, w_out = len(out_grid), len(out_grid[0])

        # Check size change
        if h_in != h_out or w_in != w_out:
            transforms.append({'type': 'resize', 'h': h_out, 'w': w_out})
            # Resize for comparison
            in_grid = self.scale_to_size(in_grid, h_out, w_out)

        # Check color mapping
        color_map = self.extract_color_map(in_grid, out_grid)
        if color_map:
            transforms.append({'type': 'color', 'map': color_map})

        return transforms