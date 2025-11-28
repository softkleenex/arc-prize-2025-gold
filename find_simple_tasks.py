"""
Find and analyze simple tasks that might be solvable
Focus on tasks with clear, deterministic transformations
"""

import json
import numpy as np


def find_simple_tasks():
    print("=" * 70)
    print("FINDING SIMPLE TASKS")
    print("Looking for deterministic transformations")
    print("=" * 70)

    # Load data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)

    simple_tasks = {
        'identity': [],
        'rotation': [],
        'mirror': [],
        'scale_2x': [],
        'scale_3x': [],
        'single_color': [],
        'small_grid': []
    }

    print(f"\nAnalyzing {len(eval_data)} tasks...")

    for task_id, task in eval_data.items():
        train_examples = task.get('train', [])

        if not train_examples:
            continue

        # Check for simple patterns
        is_simple = False
        pattern_type = None

        # Check if all outputs are identical
        outputs = [tuple(tuple(row) for row in ex['output']) for ex in train_examples]
        if len(set(outputs)) == 1:
            simple_tasks['identity'].append(task_id)
            is_simple = True
            pattern_type = 'identity'

        # Check for consistent size transformations
        size_ratios = []
        for ex in train_examples:
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])

            if in_h > 0 and in_w > 0:
                h_ratio = out_h / in_h
                w_ratio = out_w / in_w
                size_ratios.append((h_ratio, w_ratio))

        # Check for consistent scaling
        if size_ratios and all(r == size_ratios[0] for r in size_ratios):
            h_r, w_r = size_ratios[0]
            if h_r == 2.0 and w_r == 2.0:
                simple_tasks['scale_2x'].append(task_id)
                is_simple = True
                pattern_type = 'scale_2x'
            elif h_r == 3.0 and w_r == 3.0:
                simple_tasks['scale_3x'].append(task_id)
                is_simple = True
                pattern_type = 'scale_3x'

        # Check for small grids (easier to solve)
        all_small = True
        for ex in train_examples:
            if len(ex['input']) > 5 or len(ex['input'][0]) > 5:
                all_small = False
                break
            if len(ex['output']) > 5 or len(ex['output'][0]) > 5:
                all_small = False
                break

        if all_small:
            simple_tasks['small_grid'].append(task_id)
            if not is_simple:
                is_simple = True
                pattern_type = 'small_grid'

        # Check for rotation/mirror patterns
        for ex in train_examples:
            in_grid = ex['input']
            out_grid = ex['output']

            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                # Check rotation
                for rot in range(1, 4):
                    rotated = rotate_grid(in_grid, rot)
                    if grids_equal(rotated, out_grid):
                        simple_tasks['rotation'].append(task_id)
                        is_simple = True
                        pattern_type = f'rotate_{rot*90}'
                        break

                # Check mirror
                if grids_equal(mirror_h(in_grid), out_grid):
                    simple_tasks['mirror'].append(task_id)
                    is_simple = True
                    pattern_type = 'mirror_h'
                elif grids_equal(mirror_v(in_grid), out_grid):
                    simple_tasks['mirror'].append(task_id)
                    is_simple = True
                    pattern_type = 'mirror_v'

        if is_simple and pattern_type:
            print(f"  {task_id[:8]}... : {pattern_type}")

    print("\n" + "=" * 70)
    print("SIMPLE TASKS FOUND:")
    print("=" * 70)

    total_simple = 0
    for category, tasks in simple_tasks.items():
        if tasks:
            unique_tasks = list(set(tasks))
            print(f"\n{category.upper()}: {len(unique_tasks)} tasks")
            for task in unique_tasks[:3]:
                print(f"  • {task}")
            total_simple += len(unique_tasks)

    print(f"\nTotal simple tasks: {total_simple}")

    return simple_tasks


def rotate_grid(grid, times):
    """Rotate grid 90 degrees clockwise times"""
    result = grid
    for _ in range(times % 4):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def mirror_h(grid):
    """Mirror horizontally"""
    return grid[::-1]


def mirror_v(grid):
    """Mirror vertically"""
    return [row[::-1] for row in grid]


def grids_equal(g1, g2):
    """Check if two grids are equal"""
    if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
        return False

    for i in range(len(g1)):
        for j in range(len(g1[0])):
            if g1[i][j] != g2[i][j]:
                return False
    return True


def solve_simple_tasks(simple_tasks):
    """Try to solve the simple tasks"""
    print("\n" + "=" * 70)
    print("ATTEMPTING TO SOLVE SIMPLE TASKS")
    print("=" * 70)

    # Load evaluation data
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_data = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)

    correct_count = 0
    attempts = 0

    # Try identity tasks (output same for all inputs)
    if simple_tasks['identity']:
        print("\nTrying IDENTITY tasks...")
        for task_id in simple_tasks['identity'][:5]:
            task = eval_data[task_id]
            solution = eval_solutions[task_id]

            # Get the constant output
            constant_output = task['train'][0]['output']

            # Check if it matches
            attempts += 1
            if solution[0] == constant_output:
                correct_count += 1
                print(f"  ✓ {task_id}: CORRECT!")
            else:
                print(f"  ✗ {task_id}: Failed")

    # Try scale 2x tasks
    if simple_tasks['scale_2x']:
        print("\nTrying SCALE_2X tasks...")
        for task_id in simple_tasks['scale_2x'][:5]:
            task = eval_data[task_id]
            solution = eval_solutions[task_id]

            test_input = task['test'][0]['input']
            # Scale 2x
            scaled = scale_2x(test_input)

            attempts += 1
            if grids_equal(scaled, solution[0]):
                correct_count += 1
                print(f"  ✓ {task_id}: CORRECT!")
            else:
                print(f"  ✗ {task_id}: Failed")

    print(f"\nResults: {correct_count}/{attempts} correct")
    print(f"Accuracy: {correct_count/attempts*100:.1f}%" if attempts > 0 else "N/A")

    return correct_count, attempts


def scale_2x(grid):
    """Scale grid by 2x"""
    h, w = len(grid), len(grid[0])
    result = [[0] * (w * 2) for _ in range(h * 2)]

    for i in range(h):
        for j in range(w):
            val = grid[i][j]
            result[i*2][j*2] = val
            result[i*2][j*2+1] = val
            result[i*2+1][j*2] = val
            result[i*2+1][j*2+1] = val

    return result


if __name__ == "__main__":
    simple_tasks = find_simple_tasks()

    if any(simple_tasks.values()):
        solve_simple_tasks(simple_tasks)
    else:
        print("\n❌ No simple tasks found")

    print("\n✓ Analysis complete")