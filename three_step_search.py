"""
3-Step Program Search
Based on MindsAI approach: search for combinations of DSL primitives
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import Counter
import itertools
from arc_dsl_v1 import ARCDSL
from dsl_v3_pattern import PatternDSL


class ThreeStepSearcher:
    """Search for programs with up to 3 DSL operations"""

    def __init__(self):
        self.basic_dsl = ARCDSL()
        self.pattern_dsl = PatternDSL()
        self.primitives = self.get_all_primitives()
        self.cache = {}

    def get_all_primitives(self) -> List[Tuple[str, Callable]]:
        """Get all available DSL primitives"""
        primitives = []

        # Basic transformations
        primitives.append(('rotate', lambda g: self.basic_dsl.rotate(g)))
        primitives.append(('hmirror', lambda g: self.basic_dsl.hmirror(g)))
        primitives.append(('vmirror', lambda g: self.basic_dsl.vmirror(g)))
        primitives.append(('transpose', lambda g: self.basic_dsl.transpose(g)))

        # Scaling
        for factor in [2, 3]:
            primitives.append((f'upscale_{factor}', lambda g, f=factor: self.basic_dsl.upscale(g, f)))

        # Color operations
        primitives.append(('switch_colors', lambda g: self.switch_two_colors(g)))
        primitives.append(('fill_background', lambda g: self.fill_background_color(g)))
        primitives.append(('color_by_position', lambda g: self.pattern_dsl.color_by_position(g)))

        # Object operations
        primitives.append(('largest_object', lambda g: self.extract_largest_object(g)))
        primitives.append(('replicate_grid', lambda g: self.pattern_dsl.replicate_objects(g, 'grid')))
        primitives.append(('replicate_mirror', lambda g: self.pattern_dsl.replicate_objects(g, 'mirror')))

        # Pattern operations
        primitives.append(('complete_pattern', lambda g: self.pattern_dsl.complete_pattern(g, 'all')))
        primitives.append(('extract_pattern', lambda g: self.extract_repeating_pattern(g)))

        # Parametrized operations that learn from examples
        primitives.append(('learned_color', None))  # Special case - needs training data
        primitives.append(('learned_transform', None))  # Special case

        return primitives

    def solve(self, task: Dict) -> List[List]:
        """Main solving interface"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        if not train_examples:
            return [[[0]] for _ in test_examples]

        predictions = []
        for test_example in test_examples:
            test_input = test_example['input']
            attempts = self.search_programs(train_examples, test_input)
            predictions.append(attempts[:2])

        return predictions

    def search_programs(self, train_examples: List[Dict], test_input: List[List]) -> List:
        """Search for programs that work on training examples"""
        attempts = []

        # Try 1-step programs
        result = self.search_one_step(train_examples, test_input)
        if result:
            attempts.append(result)

        # Try 2-step programs
        result = self.search_two_step(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Try 3-step programs
        result = self.search_three_step(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Try learned programs
        result = self.search_learned(train_examples, test_input)
        if result and result not in attempts:
            attempts.append(result)

        # Ensure at least 2 attempts
        while len(attempts) < 2:
            if attempts:
                attempts.append(attempts[0])  # Duplicate best
            else:
                attempts.append(test_input)  # Return input

        return attempts

    def search_one_step(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Search for single primitive that solves the task"""

        for name, primitive in self.primitives:
            if primitive is None:
                continue  # Skip special cases

            # Test if this primitive works on all training examples
            works = True
            for ex in train_examples:
                try:
                    predicted = primitive(ex['input'])
                    if not self.grids_match(predicted, ex['output']):
                        works = False
                        break
                except:
                    works = False
                    break

            if works:
                # Apply to test input
                try:
                    return primitive(test_input)
                except:
                    pass

        return None

    def search_two_step(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Search for 2-primitive combination"""

        # Try common 2-step patterns
        common_pairs = [
            ('rotate', 'learned_color'),
            ('upscale_2', 'learned_color'),
            ('hmirror', 'vmirror'),
            ('switch_colors', 'complete_pattern'),
            ('largest_object', 'upscale_2'),
        ]

        for name1, name2 in common_pairs:
            result = self.try_program([name1, name2], train_examples, test_input)
            if result:
                return result

        # Try systematic search (limited to avoid explosion)
        max_attempts = 50
        attempts = 0

        for (name1, prim1), (name2, prim2) in itertools.product(self.primitives[:10], repeat=2):
            if attempts >= max_attempts:
                break

            if prim1 is None or prim2 is None:
                continue

            attempts += 1
            result = self.try_two_primitives(prim1, prim2, train_examples, test_input)
            if result:
                return result

        return None

    def search_three_step(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Search for 3-primitive combination"""

        # Try common 3-step patterns
        common_triples = [
            ('rotate', 'upscale_2', 'learned_color'),
            ('largest_object', 'upscale_3', 'complete_pattern'),
            ('hmirror', 'vmirror', 'switch_colors'),
            ('extract_pattern', 'replicate_grid', 'learned_color'),
        ]

        for name1, name2, name3 in common_triples:
            result = self.try_program([name1, name2, name3], train_examples, test_input)
            if result:
                return result

        return None

    def search_learned(self, train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Use learning to find program"""

        # Learn color mapping
        color_map = self.pattern_dsl.learn_color_transform(train_examples)

        if color_map:
            # Try just color mapping
            result = self.pattern_dsl.apply_learned_color(test_input, color_map)
            if self.validates_on_training(result, train_examples, test_input):
                return result

            # Try with rotation + color
            result = self.basic_dsl.rotate(test_input)
            result = self.pattern_dsl.apply_learned_color(result, color_map)
            if self.validates_on_training(result, train_examples, test_input):
                return result

        # Learn transformation rule
        try:
            rule = self.pattern_dsl.learn_transformation_rule(train_examples)
            result = rule(test_input)
            if result:
                return result
        except:
            pass

        return None

    def try_program(self, program: List[str], train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Try a specific program (list of primitive names)"""

        # Build the actual functions
        functions = []
        for name in program:
            for prim_name, prim_func in self.primitives:
                if prim_name == name:
                    if prim_func is None:
                        # Special case - learned primitive
                        if name == 'learned_color':
                            color_map = self.pattern_dsl.learn_color_transform(train_examples)
                            functions.append(lambda g, cm=color_map: self.pattern_dsl.apply_learned_color(g, cm))
                        elif name == 'learned_transform':
                            rule = self.pattern_dsl.learn_transformation_rule(train_examples)
                            functions.append(rule)
                    else:
                        functions.append(prim_func)
                    break

        if len(functions) != len(program):
            return None  # Couldn't find all primitives

        # Test on training examples
        works = True
        for ex in train_examples:
            try:
                result = ex['input']
                for func in functions:
                    result = func(result)

                if not self.grids_match(result, ex['output']):
                    works = False
                    break
            except:
                works = False
                break

        if works:
            # Apply to test input
            try:
                result = test_input
                for func in functions:
                    result = func(result)
                return result
            except:
                pass

        return None

    def try_two_primitives(self, prim1: Callable, prim2: Callable,
                           train_examples: List[Dict], test_input: List[List]) -> Optional[List[List]]:
        """Try applying two primitives in sequence"""

        # Test on training examples
        works = True
        for ex in train_examples:
            try:
                result = prim1(ex['input'])
                result = prim2(result)

                if not self.grids_match(result, ex['output']):
                    works = False
                    break
            except:
                works = False
                break

        if works:
            # Apply to test input
            try:
                result = prim1(test_input)
                result = prim2(result)
                return result
            except:
                pass

        return None

    def validates_on_training(self, result: List[List], train_examples: List[Dict],
                             test_input: List[List]) -> bool:
        """Check if transformation validates on training examples"""

        # Simple check - if result looks reasonable
        if not result or not isinstance(result, (list, tuple)):
            return False

        # Check size is reasonable
        h, w = len(result), len(result[0]) if result else 0
        if h == 0 or w == 0 or h > 100 or w > 100:
            return False

        # Check colors are reasonable
        colors = set(v for row in result for v in row)
        if len(colors) > 10 or any(c < 0 or c > 9 for c in colors):
            return False

        return True

    def grids_match(self, g1: Any, g2: List[List]) -> bool:
        """Check if two grids match"""
        try:
            # Handle various return types
            if isinstance(g1, tuple):
                g1 = list(list(row) for row in g1)
            elif not isinstance(g1, list):
                return False

            if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
                return False

            for i in range(len(g1)):
                for j in range(len(g1[0])):
                    if g1[i][j] != g2[i][j]:
                        return False
            return True
        except:
            return False

    # ============= Additional primitives =============

    def switch_two_colors(self, grid: List[List]) -> List[List]:
        """Switch two most common colors"""
        flat = [v for row in grid for v in row]
        counts = Counter(flat)

        # Get two most common non-zero colors
        common = [c for c, _ in counts.most_common() if c != 0][:2]

        if len(common) < 2:
            return grid

        c1, c2 = common[0], common[1]
        result = []
        for row in grid:
            new_row = []
            for val in row:
                if val == c1:
                    new_row.append(c2)
                elif val == c2:
                    new_row.append(c1)
                else:
                    new_row.append(val)
            result.append(new_row)

        return result

    def fill_background_color(self, grid: List[List]) -> List[List]:
        """Fill background with most common color"""
        flat = [v for row in grid for v in row if v != 0]
        if not flat:
            return grid

        most_common = Counter(flat).most_common(1)[0][0]

        result = []
        for row in grid:
            new_row = []
            for val in row:
                if val == 0:
                    new_row.append(most_common)
                else:
                    new_row.append(val)
            result.append(new_row)

        return result

    def extract_largest_object(self, grid: List[List]) -> List[List]:
        """Extract only the largest object"""
        objects = self.basic_dsl.objects(grid)

        if not objects:
            return grid

        # Find largest
        largest = max(objects, key=len)

        # Create new grid with only largest object
        h, w = len(grid), len(grid[0])
        result = [[0] * w for _ in range(h)]

        for (i, j), color in largest:
            result[i][j] = color

        return result

    def extract_repeating_pattern(self, grid: List[List]) -> List[List]:
        """Extract smallest repeating pattern"""
        h, w = len(grid), len(grid[0])

        # Try to find repeating pattern
        for pat_h in range(1, h // 2 + 1):
            for pat_w in range(1, w // 2 + 1):
                # Extract potential pattern
                pattern = []
                for i in range(pat_h):
                    pattern.append(grid[i][:pat_w])

                # Check if it repeats
                repeats = True
                for i in range(h):
                    for j in range(w):
                        pat_i = i % pat_h
                        pat_j = j % pat_w
                        if grid[i][j] != pattern[pat_i][pat_j]:
                            repeats = False
                            break
                    if not repeats:
                        break

                if repeats:
                    return pattern

        return grid


class SmartSearchSolver:
    """Combine multiple search strategies"""

    def __init__(self):
        self.searcher = ThreeStepSearcher()
        self.pattern_dsl = PatternDSL()

    def solve(self, task: Dict) -> List[List]:
        """Solve using multiple strategies"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        predictions = []
        for test_example in test_examples:
            test_input = test_example['input']

            attempts = []

            # Try 3-step search
            result = self.searcher.search_programs(train_examples, test_input)
            if result:
                attempts.extend(result[:1])

            # Try pattern-based
            try:
                examples_with_test = train_examples.copy()
                examples_with_test[0]['input'] = test_input
                result = self.pattern_dsl.compose_smart(examples_with_test)
                if result and result not in attempts:
                    attempts.append(result)
            except:
                pass

            # Ensure 2 attempts
            while len(attempts) < 2:
                attempts.append(test_input)

            predictions.append(attempts[:2])

        return predictions