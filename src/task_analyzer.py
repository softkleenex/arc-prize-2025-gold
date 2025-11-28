"""
ARC Prize 2025 - Task Analyzer Module

This module analyzes ARC tasks to identify patterns and transformation types.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter


class TaskAnalyzer:
    """Analyzer for identifying patterns in ARC tasks."""

    def __init__(self):
        self.analysis_cache = {}

    def analyze_task(self, task: Dict) -> Dict:
        """
        Analyze a single task and extract features.

        Args:
            task: Task dictionary with 'train' and 'test' keys

        Returns:
            Dictionary of task features and characteristics
        """
        analysis = {
            'n_train_examples': len(task['train']),
            'n_test_examples': len(task['test']),
            'transformations': [],
            'patterns': []
        }

        # Analyze training examples
        train_analysis = []
        for example in task['train']:
            train_analysis.append(self.analyze_example(example))

        analysis['train_examples'] = train_analysis

        # Identify common patterns
        analysis['transformation_type'] = self.identify_transformation_type(train_analysis)
        analysis['size_change'] = self.analyze_size_change(train_analysis)
        analysis['color_mapping'] = self.analyze_color_mapping(train_analysis)
        analysis['symmetry'] = self.analyze_symmetry(train_analysis)

        return analysis

    def analyze_example(self, example: Dict) -> Dict:
        """
        Analyze a single input-output example.

        Args:
            example: Dictionary with 'input' and 'output' keys

        Returns:
            Dictionary of example features
        """
        input_grid = np.array(example['input'])
        output_grid = np.array(example.get('output', []))

        analysis = {
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape if output_grid.size > 0 else None,
            'input_colors': self.get_color_info(input_grid),
            'output_colors': self.get_color_info(output_grid) if output_grid.size > 0 else None,
            'input_grid': input_grid,
            'output_grid': output_grid if output_grid.size > 0 else None
        }

        if output_grid.size > 0:
            analysis['size_ratio'] = (
                output_grid.shape[0] / input_grid.shape[0],
                output_grid.shape[1] / input_grid.shape[1]
            )
            analysis['size_change_type'] = self.classify_size_change(
                input_grid.shape, output_grid.shape
            )

        return analysis

    def get_color_info(self, grid: np.ndarray) -> Dict:
        """Get color statistics from a grid."""
        unique_colors = np.unique(grid)
        color_counts = Counter(grid.flatten())

        return {
            'unique_colors': set(unique_colors.tolist()),
            'n_colors': len(unique_colors),
            'color_counts': dict(color_counts),
            'background_color': self.identify_background_color(grid),
            'most_common': color_counts.most_common(3)
        }

    def identify_background_color(self, grid: np.ndarray) -> int:
        """Identify the background color (most frequent)."""
        color_counts = Counter(grid.flatten())
        return color_counts.most_common(1)[0][0]

    def classify_size_change(self, input_shape: Tuple, output_shape: Tuple) -> str:
        """Classify the type of size change."""
        if input_shape == output_shape:
            return "same_size"
        elif output_shape[0] > input_shape[0] and output_shape[1] > input_shape[1]:
            return "expansion"
        elif output_shape[0] < input_shape[0] and output_shape[1] < input_shape[1]:
            return "contraction"
        elif output_shape[0] == input_shape[0] and output_shape[1] != input_shape[1]:
            return "width_change"
        elif output_shape[0] != input_shape[0] and output_shape[1] == input_shape[1]:
            return "height_change"
        else:
            return "complex_change"

    def identify_transformation_type(self, train_analysis: List[Dict]) -> List[str]:
        """Identify possible transformation types based on training examples."""
        types = []

        # Check size changes
        size_changes = [ex.get('size_change_type') for ex in train_analysis if 'size_change_type' in ex]
        if all(sc == 'same_size' for sc in size_changes):
            types.append('in_place_transformation')
        elif all(sc == 'expansion' for sc in size_changes):
            types.append('scaling_up')
        elif all(sc == 'contraction' for sc in size_changes):
            types.append('scaling_down')

        # Check color changes
        for ex in train_analysis:
            if ex.get('input_colors') and ex.get('output_colors'):
                input_colors = ex['input_colors']['unique_colors']
                output_colors = ex['output_colors']['unique_colors']

                if input_colors == output_colors:
                    if 'color_rearrangement' not in types:
                        types.append('color_rearrangement')
                elif output_colors.issubset(input_colors):
                    if 'color_filtering' not in types:
                        types.append('color_filtering')
                elif input_colors.issubset(output_colors):
                    if 'color_addition' not in types:
                        types.append('color_addition')
                else:
                    if 'color_replacement' not in types:
                        types.append('color_replacement')

        return types if types else ['unknown']

    def analyze_size_change(self, train_analysis: List[Dict]) -> Dict:
        """Analyze size change patterns."""
        size_ratios = []
        size_changes = []

        for ex in train_analysis:
            if 'size_ratio' in ex:
                size_ratios.append(ex['size_ratio'])
            if 'size_change_type' in ex:
                size_changes.append(ex['size_change_type'])

        return {
            'consistent_ratio': len(set(size_ratios)) == 1 if size_ratios else False,
            'ratios': size_ratios,
            'change_types': size_changes,
            'consistent_type': len(set(size_changes)) == 1 if size_changes else False
        }

    def analyze_color_mapping(self, train_analysis: List[Dict]) -> Dict:
        """Analyze color mapping patterns."""
        color_mappings = []

        for ex in train_analysis:
            if ex.get('input_colors') and ex.get('output_colors'):
                input_colors = ex['input_colors']['unique_colors']
                output_colors = ex['output_colors']['unique_colors']

                color_mappings.append({
                    'input': input_colors,
                    'output': output_colors,
                    'added': output_colors - input_colors,
                    'removed': input_colors - output_colors,
                    'preserved': input_colors & output_colors
                })

        return {
            'mappings': color_mappings,
            'consistent_mapping': self.check_consistent_mapping(color_mappings)
        }

    def check_consistent_mapping(self, mappings: List[Dict]) -> bool:
        """Check if color mappings are consistent across examples."""
        if len(mappings) < 2:
            return True

        # Check if added/removed colors are consistent
        added_colors = [m['added'] for m in mappings]
        removed_colors = [m['removed'] for m in mappings]

        # Consider consistent if patterns are similar
        return True  # Simplified for now

    def analyze_symmetry(self, train_analysis: List[Dict]) -> Dict:
        """Analyze symmetry patterns in grids."""
        symmetries = []

        for ex in train_analysis:
            input_grid = ex.get('input_grid')
            output_grid = ex.get('output_grid')

            if input_grid is not None:
                input_sym = self.check_symmetries(input_grid)
                symmetries.append({
                    'input': input_sym,
                    'output': self.check_symmetries(output_grid) if output_grid is not None else None
                })

        return symmetries

    def check_symmetries(self, grid: np.ndarray) -> Dict:
        """Check various symmetries in a grid."""
        if grid.size == 0:
            return {}

        return {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'diagonal': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False,
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2))
        }

    def find_repeated_patterns(self, grid: np.ndarray, pattern_size: int = 2) -> List[Tuple]:
        """Find repeated patterns in a grid."""
        patterns = []
        h, w = grid.shape

        for i in range(h - pattern_size + 1):
            for j in range(w - pattern_size + 1):
                pattern = grid[i:i+pattern_size, j:j+pattern_size]
                # Search for this pattern elsewhere
                for ii in range(h - pattern_size + 1):
                    for jj in range(w - pattern_size + 1):
                        if (ii, jj) != (i, j):
                            candidate = grid[ii:ii+pattern_size, jj:jj+pattern_size]
                            if np.array_equal(pattern, candidate):
                                patterns.append(((i, j), (ii, jj), pattern))

        return patterns


def analyze_dataset(split: str = "training", max_tasks: int = None) -> Dict:
    """
    Analyze entire dataset and categorize tasks.

    Args:
        split: 'training' or 'evaluation'
        max_tasks: Maximum number of tasks to analyze (None for all)

    Returns:
        Dictionary with dataset-wide statistics
    """
    from data_loader import ARCDataLoader

    loader = ARCDataLoader()
    analyzer = TaskAnalyzer()

    challenges = loader.load_challenges(split)
    solutions = loader.load_solutions(split) if split != 'test' else {}

    task_ids = list(challenges.keys())[:max_tasks] if max_tasks else list(challenges.keys())

    analyses = {}
    transformation_types = Counter()
    size_change_types = Counter()

    print(f"Analyzing {len(task_ids)} tasks from {split} set...")

    for i, task_id in enumerate(task_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(task_ids)} tasks...")

        task = challenges[task_id]
        analysis = analyzer.analyze_task(task)
        analyses[task_id] = analysis

        # Collect statistics
        for trans_type in analysis['transformation_type']:
            transformation_types[trans_type] += 1

        if analysis['size_change']['consistent_type']:
            for sc_type in analysis['size_change']['change_types']:
                size_change_types[sc_type] += 1

    return {
        'n_tasks': len(task_ids),
        'task_analyses': analyses,
        'transformation_types': dict(transformation_types),
        'size_change_types': dict(size_change_types)
    }


if __name__ == "__main__":
    print("="*70)
    print("ARC Prize 2025 - Task Pattern Analysis")
    print("="*70)

    # Analyze a small sample first
    print("\nAnalyzing first 20 training tasks...")
    results = analyze_dataset("training", max_tasks=20)

    print(f"\nTransformation type distribution:")
    for trans_type, count in sorted(results['transformation_types'].items(), key=lambda x: -x[1]):
        print(f"  {trans_type}: {count}")

    print(f"\nSize change type distribution:")
    for size_type, count in sorted(results['size_change_types'].items(), key=lambda x: -x[1]):
        print(f"  {size_type}: {count}")

    # Show detailed analysis of first task
    first_task_id = list(results['task_analyses'].keys())[0]
    first_analysis = results['task_analyses'][first_task_id]

    print(f"\n" + "="*70)
    print(f"Detailed Analysis of Task: {first_task_id}")
    print("="*70)
    print(f"Training examples: {first_analysis['n_train_examples']}")
    print(f"Transformation types: {', '.join(first_analysis['transformation_type'])}")
    print(f"Size change consistent: {first_analysis['size_change']['consistent_type']}")

    for i, ex in enumerate(first_analysis['train_examples']):
        print(f"\n  Example {i+1}:")
        print(f"    Input:  {ex['input_shape']} with {ex['input_colors']['n_colors']} colors")
        if ex.get('output_shape'):
            print(f"    Output: {ex['output_shape']} with {ex['output_colors']['n_colors']} colors")
            print(f"    Change type: {ex.get('size_change_type', 'N/A')}")
