"""
Perfect Solver: Context-aware color learning for 100% accuracy
Focus on exact transformations that our analysis shows are nearly working
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict


class PerfectSolver:
    """Solver aiming for perfect pixel accuracy"""

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
            attempts = self.solve_perfect(train_examples, test_input)
            predictions.append(attempts[:2])

        return predictions

    def solve_perfect(self, train_examples: List[Dict], test_input: List[List]) -> List:
        """Generate perfect solution attempts"""
        attempts = []

        # Method 1: Context-aware color mapping
        result = self.context_aware_color_mapping(train_examples, test_input)
        if result:
            attempts.append(result)

        # Method 2: Pattern-based exact replication
        result = self.pattern_exact_replication(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Method 3: Neighbor-based transformation
        result = self.neighbor_based_transform(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Method 4: Object-aware transformation
        result = self.object_aware_transform(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Ensure at least 2 attempts
        while len(attempts) < 2:
            if attempts:
                # Create variation
                attempts.append(self.create_smart_variation(attempts[0], train_examples))
            else:
                attempts.append(test_input)

        return attempts

    def context_aware_color_mapping(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Map colors considering context and neighbors"""

        # First check if it's a same-size color transformation
        all_same_size = all(
            len(ex['input']) == len(ex['output']) and
            len(ex['input'][0]) == len(ex['output'][0])
            for ex in train_examples
        )

        if not all_same_size:
            return None

        # Learn context-aware mapping rules
        mapping_rules = self.learn_context_rules(train_examples)

        if not mapping_rules:
            return None

        # Apply context-aware mapping
        h, w = len(test_input), len(test_input[0])
        result = [[0] * w for _ in range(h)]

        for i in range(h):
            for j in range(w):
                val = test_input[i][j]
                context = self.get_context(test_input, i, j)

                # Find best matching rule
                new_val = self.apply_context_rule(val, context, mapping_rules)
                result[i][j] = new_val

        return result

    def learn_context_rules(self, train_examples: List[Dict]) -> Dict:
        """Learn color mapping rules with context"""
        rules = defaultdict(list)

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]):
                continue

            h, w = len(in_grid), len(in_grid[0])

            for i in range(h):
                for j in range(w):
                    in_val = in_grid[i][j]
                    out_val = out_grid[i][j]

                    if in_val != out_val:
                        # Color changed - learn the context
                        context = self.get_context(in_grid, i, j)
                        rules[(in_val, context)].append(out_val)

        # Convert to deterministic rules
        final_rules = {}
        for key, values in rules.items():
            if values:
                # Most common output
                final_rules[key] = Counter(values).most_common(1)[0][0]

        # Also learn simple color mappings
        simple_map = self.learn_simple_color_map(train_examples)
        final_rules['simple'] = simple_map

        return final_rules

    def get_context(self, grid: List[List], i: int, j: int) -> Tuple:
        """Get context around a position"""
        h, w = len(grid), len(grid[0])
        context = []

        # Get 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    context.append(grid[ni][nj])
                else:
                    context.append(-1)  # Out of bounds

        return tuple(context)

    def apply_context_rule(self, val: int, context: Tuple, rules: Dict) -> int:
        """Apply context-aware rule"""
        # Try context-specific rule first
        if (val, context) in rules:
            return rules[(val, context)]

        # Try simple mapping
        if 'simple' in rules and val in rules['simple']:
            return rules['simple'][val]

        # No change
        return val

    def learn_simple_color_map(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn simple color mapping"""
        all_mappings = []

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]):
                continue

            mapping = {}
            for i in range(len(in_grid)):
                for j in range(len(in_grid[0])):
                    in_val = in_grid[i][j]
                    out_val = out_grid[i][j]
                    if in_val != 0:
                        if in_val not in mapping:
                            mapping[in_val] = out_val
                        elif mapping[in_val] != out_val:
                            # Inconsistent - this color maps to different values
                            mapping[in_val] = None

            # Only keep consistent mappings
            clean_mapping = {k: v for k, v in mapping.items() if v is not None}
            all_mappings.append(clean_mapping)

        # Find consensus
        final_map = {}
        if all_mappings:
            all_keys = set()
            for m in all_mappings:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [m.get(key) for m in all_mappings if key in m]
                if values and all(v == values[0] for v in values):
                    # All agree
                    final_map[key] = values[0]

        return final_map

    def pattern_exact_replication(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Replicate exact patterns from training"""

        # Check if output is always the same
        outputs = []
        for ex in train_examples:
            out_key = tuple(tuple(row) for row in ex['output'])
            outputs.append(out_key)

        if len(set(outputs)) == 1:
            # All outputs identical
            return [list(row) for row in outputs[0]]

        # Check if there's a pattern based on input features
        test_features = self.extract_detailed_features(test_input)

        best_match = None
        best_score = 0

        for ex in train_examples:
            in_features = self.extract_detailed_features(ex['input'])
            score = self.detailed_feature_match(in_features, test_features)

            if score > best_score:
                best_score = score
                best_match = ex

        if best_match and best_score > 0.95:
            # Very similar input - apply same transformation
            return self.apply_same_transformation(best_match, test_input)

        return None

    def neighbor_based_transform(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Transform based on neighbor patterns"""

        # Learn neighbor-based rules
        neighbor_rules = self.learn_neighbor_rules(train_examples)

        if not neighbor_rules:
            return None

        h, w = len(test_input), len(test_input[0])
        result = [[0] * w for _ in range(h)]

        for i in range(h):
            for j in range(w):
                val = test_input[i][j]
                neighbors = self.get_neighbor_pattern(test_input, i, j)

                # Apply neighbor rule
                new_val = neighbor_rules.get((val, neighbors), val)
                result[i][j] = new_val

        return result

    def object_aware_transform(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Transform considering objects/connected components"""

        # Extract objects from test input
        test_objects = self.extract_objects(test_input)

        # Learn object transformation rules
        object_rules = self.learn_object_rules(train_examples)

        if not object_rules:
            return None

        h, w = len(test_input), len(test_input[0])
        result = [[0] * w for _ in range(h)]

        # Apply object-based transformation
        for obj_id, positions in test_objects.items():
            obj_features = self.get_object_features(test_input, positions)

            # Find matching rule
            new_color = self.apply_object_rule(obj_features, object_rules)

            for i, j in positions:
                result[i][j] = new_color

        return result

    def learn_neighbor_rules(self, train_examples: List[Dict]) -> Dict:
        """Learn rules based on neighbor patterns"""
        rules = defaultdict(list)

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]):
                continue

            for i in range(len(in_grid)):
                for j in range(len(in_grid[0])):
                    in_val = in_grid[i][j]
                    out_val = out_grid[i][j]

                    if in_val != out_val:
                        neighbors = self.get_neighbor_pattern(in_grid, i, j)
                        rules[(in_val, neighbors)].append(out_val)

        # Convert to deterministic
        final_rules = {}
        for key, values in rules.items():
            if values:
                final_rules[key] = Counter(values).most_common(1)[0][0]

        return final_rules

    def get_neighbor_pattern(self, grid: List[List], i: int, j: int) -> Tuple:
        """Get pattern of neighbors"""
        h, w = len(grid), len(grid[0])
        pattern = []

        # 4-connected neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                pattern.append(grid[ni][nj] != 0)  # Is non-zero
            else:
                pattern.append(False)

        return tuple(pattern)

    def extract_objects(self, grid: List[List]) -> Dict[int, List[Tuple[int, int]]]:
        """Extract connected components"""
        h, w = len(grid), len(grid[0])
        visited = set()
        objects = {}
        obj_id = 0

        def dfs(i, j, color, positions):
            if (i, j) in visited or i < 0 or i >= h or j < 0 or j >= w:
                return
            if grid[i][j] != color or grid[i][j] == 0:
                return
            visited.add((i, j))
            positions.append((i, j))
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj, color, positions)

        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i][j] != 0:
                    positions = []
                    dfs(i, j, grid[i][j], positions)
                    objects[obj_id] = positions
                    obj_id += 1

        return objects

    def get_object_features(self, grid: List[List], positions: List[Tuple[int, int]]) -> Dict:
        """Get features of an object"""
        if not positions:
            return {}

        colors = [grid[i][j] for i, j in positions]
        min_i = min(i for i, j in positions)
        max_i = max(i for i, j in positions)
        min_j = min(j for i, j in positions)
        max_j = max(j for i, j in positions)

        return {
            'size': len(positions),
            'color': Counter(colors).most_common(1)[0][0],
            'width': max_j - min_j + 1,
            'height': max_i - min_i + 1,
            'density': len(positions) / ((max_i - min_i + 1) * (max_j - min_j + 1))
        }

    def learn_object_rules(self, train_examples: List[Dict]) -> Dict:
        """Learn object transformation rules"""
        rules = defaultdict(list)

        for ex in train_examples:
            in_objects = self.extract_objects(ex['input'])
            out_objects = self.extract_objects(ex['output'])

            # Match objects by position overlap
            for in_id, in_positions in in_objects.items():
                in_features = self.get_object_features(ex['input'], in_positions)

                # Find corresponding output
                best_overlap = 0
                best_out_color = None

                for out_id, out_positions in out_objects.items():
                    overlap = len(set(in_positions) & set(out_positions))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        out_features = self.get_object_features(ex['output'], out_positions)
                        best_out_color = out_features.get('color')

                if best_out_color is not None:
                    key = (in_features['color'], in_features['size'])
                    rules[key].append(best_out_color)

        # Convert to deterministic
        final_rules = {}
        for key, values in rules.items():
            if values:
                final_rules[key] = Counter(values).most_common(1)[0][0]

        return final_rules

    def apply_object_rule(self, obj_features: Dict, rules: Dict) -> int:
        """Apply object transformation rule"""
        if not obj_features:
            return 0

        key = (obj_features.get('color'), obj_features.get('size'))
        if key in rules:
            return rules[key]

        # Try just color
        for (color, _), new_color in rules.items():
            if color == obj_features.get('color'):
                return new_color

        return obj_features.get('color', 0)

    def extract_detailed_features(self, grid: List[List]) -> Dict:
        """Extract detailed features for matching"""
        h, w = len(grid), len(grid[0])
        flat = [v for row in grid for v in row]

        return {
            'shape': (h, w),
            'colors': tuple(sorted(set(flat))),
            'color_counts': tuple(sorted(Counter(flat).items())),
            'density': sum(1 for v in flat if v != 0) / len(flat),
            'checksum': sum(flat),
            'pattern_hash': hash(tuple(tuple(row) for row in grid))
        }

    def detailed_feature_match(self, f1: Dict, f2: Dict) -> float:
        """Calculate detailed feature match score"""
        score = 0.0

        if f1['shape'] == f2['shape']:
            score += 0.4
        if f1['colors'] == f2['colors']:
            score += 0.2
        if f1['color_counts'] == f2['color_counts']:
            score += 0.2
        if abs(f1['density'] - f2['density']) < 0.01:
            score += 0.1
        if f1['checksum'] == f2['checksum']:
            score += 0.1

        return score

    def apply_same_transformation(self, reference: Dict, test_input: List[List]) -> List[List]:
        """Apply the same transformation as reference"""
        ref_in = reference['input']
        ref_out = reference['output']

        # If same size, try pixel mapping
        if (len(ref_in) == len(ref_out) and len(ref_in[0]) == len(ref_out[0]) and
            len(test_input) == len(ref_in) and len(test_input[0]) == len(ref_in[0])):

            # Build position-specific mapping
            h, w = len(test_input), len(test_input[0])
            result = [[0] * w for _ in range(h)]

            for i in range(h):
                for j in range(w):
                    # Check if same pattern at this position
                    if test_input[i][j] == ref_in[i][j]:
                        result[i][j] = ref_out[i][j]
                    else:
                        # Apply color mapping
                        color_map = self.extract_color_mapping(ref_in, ref_out)
                        result[i][j] = color_map.get(test_input[i][j], test_input[i][j])

            return result

        return test_input

    def extract_color_mapping(self, in_grid: List[List], out_grid: List[List]) -> Dict[int, int]:
        """Extract consistent color mapping"""
        mapping = {}

        if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
            for i in range(len(in_grid)):
                for j in range(len(in_grid[0])):
                    in_val = in_grid[i][j]
                    out_val = out_grid[i][j]

                    if in_val != 0:
                        if in_val not in mapping:
                            mapping[in_val] = out_val

        return mapping

    def create_smart_variation(self, grid: List[List], train_examples: List[Dict]) -> List[List]:
        """Create intelligent variation based on training patterns"""
        # Try applying a different color map
        alt_map = self.learn_alternative_color_map(train_examples)

        if alt_map:
            result = []
            for row in grid:
                new_row = []
                for val in row:
                    new_row.append(alt_map.get(val, val))
                result.append(new_row)
            return result

        return [list(row) for row in grid]

    def learn_alternative_color_map(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn alternative color mapping"""
        # Try second most common mappings
        all_mappings = defaultdict(list)

        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                for i in range(len(in_grid)):
                    for j in range(len(in_grid[0])):
                        in_val = in_grid[i][j]
                        out_val = out_grid[i][j]
                        if in_val != 0:
                            all_mappings[in_val].append(out_val)

        # Get second most common
        alt_map = {}
        for key, values in all_mappings.items():
            if values:
                counts = Counter(values)
                if len(counts) > 1:
                    # Get second most common
                    alt_map[key] = counts.most_common(2)[1][0]
                else:
                    alt_map[key] = counts.most_common(1)[0][0]

        return alt_map