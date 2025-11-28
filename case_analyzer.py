"""
심층 케이스 분석기
높은 partial match 케이스들의 실패 원인 정확히 파악
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class CaseAnalyzer:
    """개별 케이스 심층 분석"""

    def __init__(self):
        self.eval_challenges = self.load_json('data/arc-agi_evaluation_challenges.json')
        self.eval_solutions = self.load_json('data/arc-agi_evaluation_solutions.json')

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def analyze_case(self, task_id: str):
        """특정 케이스 상세 분석"""
        print(f"\n{'='*70}")
        print(f"Analyzing Task: {task_id}")
        print(f"{'='*70}\n")

        task = self.eval_challenges[task_id]
        solutions = self.eval_solutions[task_id]

        # Training examples 분석
        print("TRAINING EXAMPLES:")
        print("-" * 70)

        size_transforms = []
        color_mappings = []

        for i, example in enumerate(task['train']):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            print(f"\nExample {i+1}:")
            print(f"  Input shape:  {input_grid.shape}")
            print(f"  Output shape: {output_grid.shape}")

            # 크기 변환
            if input_grid.shape != output_grid.shape:
                ratio = (output_grid.shape[0] / input_grid.shape[0],
                        output_grid.shape[1] / input_grid.shape[1])
                size_transforms.append(ratio)
                print(f"  Size ratio: {ratio}")
            else:
                size_transforms.append((1.0, 1.0))
                print(f"  Size ratio: (1.0, 1.0) - same size")

            # 색상 분석
            in_colors = set(input_grid.flatten())
            out_colors = set(output_grid.flatten())
            print(f"  Input colors:  {sorted(in_colors)}")
            print(f"  Output colors: {sorted(out_colors)}")

            # 새로운 색상
            new_colors = out_colors - in_colors
            if new_colors:
                print(f"  New colors: {sorted(new_colors)}")

            # 제거된 색상
            removed_colors = in_colors - out_colors
            if removed_colors:
                print(f"  Removed colors: {sorted(removed_colors)}")

        # 크기 변환 패턴
        print(f"\n{'='*70}")
        print("SIZE TRANSFORMATION PATTERN:")
        print("-" * 70)
        if all(st == size_transforms[0] for st in size_transforms):
            print(f"✓ Consistent ratio: {size_transforms[0]}")
        else:
            print(f"✗ Inconsistent ratios: {size_transforms}")

        # Test 예제
        print(f"\n{'='*70}")
        print("TEST EXAMPLES:")
        print("-" * 70)

        for i, (test_input, expected_output) in enumerate(zip(task['test'], solutions)):
            test_grid = np.array(test_input['input'])
            expected_grid = np.array(expected_output)

            print(f"\nTest {i+1}:")
            print(f"  Input shape:  {test_grid.shape}")
            print(f"  Expected output shape: {expected_grid.shape}")
            print(f"  Input colors:  {sorted(set(test_grid.flatten()))}")
            print(f"  Output colors: {sorted(set(expected_grid.flatten()))}")

        return {
            'task_id': task_id,
            'size_transforms': size_transforms,
            'consistent_size': all(st == size_transforms[0] for st in size_transforms),
        }

    def compare_with_prediction(self, task_id: str, solver_file: str):
        """예측과 정답 비교"""
        print(f"\n{'='*70}")
        print(f"Comparing Prediction with Ground Truth: {task_id}")
        print(f"{'='*70}\n")

        # Solver 실행
        import importlib.util
        spec = importlib.util.spec_from_file_location("solver", solver_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        solver_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and 'Solver' in name:
                solver_class = obj
                break

        solver = solver_class()

        task = self.eval_challenges[task_id]
        solutions = self.eval_solutions[task_id]

        # 예측 생성
        predictions = solver.solve(task, max_attempts=2)

        for i, (pred_attempts, expected) in enumerate(zip(predictions, solutions)):
            pred = np.array(pred_attempts[0])
            expected_grid = np.array(expected)

            print(f"\nTest {i+1}:")
            print(f"  Predicted shape: {pred.shape}")
            print(f"  Expected shape:  {expected_grid.shape}")

            if pred.shape == expected_grid.shape:
                match_count = np.sum(pred == expected_grid)
                total = expected_grid.size
                match_ratio = match_count / total

                print(f"  ✓ Shape matches!")
                print(f"  Match ratio: {match_ratio:.2%} ({match_count}/{total})")

                if match_ratio < 1.0:
                    # 어디가 틀렸는지
                    diff_positions = np.argwhere(pred != expected_grid)
                    print(f"  {len(diff_positions)} cells differ")

                    if len(diff_positions) <= 10:
                        print(f"\n  Different positions:")
                        for pos in diff_positions[:10]:
                            i, j = pos
                            print(f"    [{i},{j}]: predicted={pred[i,j]}, expected={expected_grid[i,j]}")
            else:
                print(f"  ✗ Shape mismatch!")

    def find_transformation_rule(self, task_id: str):
        """변환 규칙 추론"""
        print(f"\n{'='*70}")
        print(f"Finding Transformation Rule: {task_id}")
        print(f"{'='*70}\n")

        task = self.eval_challenges[task_id]

        rules = []

        for i, example in enumerate(task['train']):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            print(f"Example {i+1} Analysis:")

            # 규칙 1: 모든 셀이 변환되었나?
            if input_grid.shape == output_grid.shape:
                unchanged = np.sum(input_grid == output_grid)
                total = input_grid.size
                print(f"  Unchanged cells: {unchanged}/{total} ({unchanged/total:.1%})")

                if unchanged > total * 0.9:
                    print(f"  → Rule: Most cells unchanged (identity-like)")
                elif unchanged < total * 0.1:
                    print(f"  → Rule: Most cells changed (full transformation)")
                else:
                    print(f"  → Rule: Partial transformation")

            # 규칙 2: 크기 변환
            if input_grid.shape != output_grid.shape:
                ratio = (output_grid.shape[0] / input_grid.shape[0],
                        output_grid.shape[1] / input_grid.shape[1])
                print(f"  → Rule: Size transform {ratio}")

            # 규칙 3: 색상 매핑
            in_colors = set(input_grid.flatten())
            out_colors = set(output_grid.flatten())

            if in_colors == out_colors:
                print(f"  → Rule: Same color set (spatial rearrangement)")
            else:
                print(f"  → Rule: Color transformation")
                print(f"     Input colors: {sorted(in_colors)}")
                print(f"     Output colors: {sorted(out_colors)}")

            print()


def main():
    analyzer = CaseAnalyzer()

    # 높은 partial match 케이스들 분석
    high_match_cases = [
        '135a2760',  # 98.93%
        'e376de54',  # 98.83%
        '8e5c0c38',  # 97.73%
    ]

    print("="*70)
    print("HIGH PARTIAL MATCH CASES ANALYSIS")
    print("="*70)

    for task_id in high_match_cases:
        # 케이스 분석
        analysis = analyzer.analyze_case(task_id)

        # 변환 규칙 추론
        analyzer.find_transformation_rule(task_id)

        # V2 예측과 비교
        analyzer.compare_with_prediction(task_id, 'kaggle_notebook_v2.py')

        print("\n" + "="*70 + "\n")
        input("Press Enter to continue to next case...")


if __name__ == "__main__":
    main()
