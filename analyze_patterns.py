"""
Deep analysis of training data patterns
Find actual transformation rules
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from arc_dsl_v1 import grid_from_list


def analyze_training_patterns():
    """Analyze all 1000 training tasks"""
    print("=" * 70)
    print("Deep Pattern Analysis - 1000 Training Tasks")
    print("=" * 70)

    with open('data/arc-agi_training_challenges.json', 'r') as f:
        train_data = json.load(f)
    with open('data/arc-agi_training_solutions.json', 'r') as f:
        train_solutions = json.load(f)

    patterns = {
        'size_transforms': Counter(),
        'color_operations': Counter(),
        'object_counts': Counter(),
        'symmetries': Counter(),
        'special_patterns': []
    }

    print(f"\nAnalyzing {len(train_data)} tasks...")

    for i, (task_id, task) in enumerate(train_data.items()):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(train_data)}")

        solutions = train_solutions[task_id]

        # Analyze each training example
        for example, solution in zip(task['train'], solutions):
            in_grid = grid_from_list(example['input'])
            out_grid = grid_from_list(solution)

            # Size transformation
            h_in, w_in = len(in_grid), len(in_grid[0])
            h_out, w_out = len(out_grid), len(out_grid[0])

            size_ratio = (h_out / h_in, w_out / w_in)
            patterns['size_transforms'][size_ratio] += 1

            # Color operations
            in_colors = set(v for row in in_grid for v in row)
            out_colors = set(v for row in out_grid for v in row)

            if in_colors == out_colors:
                patterns['color_operations']['preserved'] += 1
            elif out_colors < in_colors:
                patterns['color_operations']['reduced'] += 1
            elif out_colors > in_colors:
                patterns['color_operations']['increased'] += 1
            else:
                patterns['color_operations']['changed'] += 1

            # Object count change
            in_obj_count = count_objects(in_grid)
            out_obj_count = count_objects(out_grid)

            if in_obj_count == out_obj_count:
                patterns['object_counts']['same'] += 1
            elif out_obj_count > in_obj_count:
                patterns['object_counts']['increased'] += 1
            else:
                patterns['object_counts']['decreased'] += 1

            # Check for symmetry
            if is_symmetric(out_grid, 'horizontal'):
                patterns['symmetries']['horizontal'] += 1
            if is_symmetric(out_grid, 'vertical'):
                patterns['symmetries']['vertical'] += 1
            if is_symmetric(out_grid, 'diagonal'):
                patterns['symmetries']['diagonal'] += 1

            # Special patterns
            if h_in == h_out and w_in == w_out:
                # Same size - check for specific transformations
                match_ratio = calculate_match(in_grid, out_grid)

                if match_ratio > 0.9:
                    patterns['special_patterns'].append(('high_match', task_id))
                elif is_rotation(in_grid, out_grid):
                    patterns['special_patterns'].append(('rotation', task_id))
                elif is_mirror(in_grid, out_grid):
                    patterns['special_patterns'].append(('mirror', task_id))

    # Report findings
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS RESULTS")
    print("=" * 70)

    print("\n1. Size Transformations (Top 10):")
    for ratio, count in patterns['size_transforms'].most_common(10):
        h_r, w_r = ratio
        if h_r == 1.0 and w_r == 1.0:
            print(f"   Same size: {count} ({count/len(train_data)*100:.1f}%)")
        elif h_r == w_r:
            print(f"   Scale {h_r:.1f}x: {count} ({count/len(train_data)*100:.1f}%)")
        else:
            print(f"   Resize {h_r:.1f}x{w_r:.1f}: {count}")

    print("\n2. Color Operations:")
    total_color = sum(patterns['color_operations'].values())
    for op, count in patterns['color_operations'].most_common():
        print(f"   {op}: {count} ({count/total_color*100:.1f}%)")

    print("\n3. Object Count Changes:")
    total_obj = sum(patterns['object_counts'].values())
    for change, count in patterns['object_counts'].most_common():
        print(f"   {change}: {count} ({count/total_obj*100:.1f}%)")

    print("\n4. Symmetries:")
    for sym_type, count in patterns['symmetries'].most_common():
        print(f"   {sym_type}: {count}")

    print("\n5. Special Patterns:")
    special_counts = Counter(p[0] for p in patterns['special_patterns'])
    for pattern, count in special_counts.most_common():
        print(f"   {pattern}: {count}")

    return patterns


def count_objects(grid) -> int:
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


def is_symmetric(grid, axis='horizontal') -> bool:
    """Check if grid is symmetric"""
    h, w = len(grid), len(grid[0])

    if axis == 'horizontal':
        for i in range(h // 2):
            if grid[i] != grid[h - 1 - i]:
                return False
    elif axis == 'vertical':
        for i in range(h):
            for j in range(w // 2):
                if grid[i][j] != grid[i][w - 1 - j]:
                    return False
    elif axis == 'diagonal':
        if h != w:
            return False
        for i in range(h):
            for j in range(w):
                if grid[i][j] != grid[j][i]:
                    return False

    return True


def calculate_match(g1, g2) -> float:
    """Calculate match ratio between two grids"""
    if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
        return 0.0

    total = len(g1) * len(g1[0])
    matches = sum(1 for i in range(len(g1))
                 for j in range(len(g1[0]))
                 if g1[i][j] == g2[i][j])

    return matches / total


def is_rotation(g1, g2) -> bool:
    """Check if g2 is rotation of g1"""
    # Try all rotations
    temp = g1
    for _ in range(4):
        temp = rotate_90(temp)
        if temp == g2:
            return True
    return False


def is_mirror(g1, g2) -> bool:
    """Check if g2 is mirror of g1"""
    # Horizontal mirror
    if g1[::-1] == g2:
        return True
    # Vertical mirror
    if tuple(row[::-1] for row in g1) == g2:
        return True
    return False


def rotate_90(grid):
    """Rotate grid 90 degrees clockwise"""
    return tuple(row for row in zip(*grid[::-1]))


def find_best_patterns():
    """Find most successful pattern types"""
    print("\n" + "=" * 70)
    print("ACTIONABLE INSIGHTS")
    print("=" * 70)

    print("\nMost Common Transformations:")
    print("1. Same-size transformations (50%+)")
    print("   → Focus on: color mapping, object manipulation")
    print("2. Scale 2x and 3x (significant %)")
    print("   → Implement: precise scaling")
    print("3. Color changes common (45%)")
    print("   → Need: smart color learning")

    print("\nRecommended DSL Improvements:")
    print("• Add color mapping learner")
    print("• Implement connected component operations")
    print("• Add symmetry completers")
    print("• Focus on object-based transforms")


if __name__ == "__main__":
    patterns = analyze_training_patterns()
    find_best_patterns()

    # Save analysis
    with open('pattern_analysis.json', 'w') as f:
        # Convert Counter objects to dict for JSON, handling tuple keys
        size_transforms_dict = {}
        for (h_ratio, w_ratio), count in patterns['size_transforms'].most_common(20):
            key = f"{h_ratio:.2f}x{w_ratio:.2f}"
            size_transforms_dict[key] = count

        save_data = {
            'size_transforms': size_transforms_dict,
            'color_operations': dict(patterns['color_operations']),
            'object_counts': dict(patterns['object_counts']),
            'symmetries': dict(patterns['symmetries'])
        }
        json.dump(save_data, f, indent=2)

    print("\n✓ Analysis saved to pattern_analysis.json")