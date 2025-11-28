"""
ARC Prize 2025 - Version 6: Learning + Augmentation

Based on research insights:
- Test-Time Training principles
- Training example augmentation
- Rule extraction from examples
- Shortest program preference
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Tuple
from collections import Counter


class LearningSolver:
    """Learning-based solver with augmentation and rule extraction."""

    def __init__(self):
        # 기본 atomic operations
        self.atomic_ops = {
            'identity': lambda g: g.copy(),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
            'rot90': lambda g: np.rot90(g, k=-1),
            'rot180': lambda g: np.rot90(g, k=2),
            'rot270': lambda g: np.rot90(g, k=-3),
            'transpose': lambda g: g.T,
        }

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """Solve with learning and augmentation."""
        predictions = []

        # 1. Training examples 증강
        augmented_train = self.augment_training(task['train'])

        # 2. 변환 규칙 학습
        learned_rules = self.learn_rules(augmented_train)

        # 3. Test input 처리
        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            # 학습된 규칙 적용
            attempts = []

            for rule in learned_rules[:max_attempts]:
                try:
                    output = self.apply_rule(test_input, rule)
                    attempts.append(output.tolist())
                except:
                    attempts.append(test_input.tolist())

            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def augment_training(self, train_examples: List[dict]) -> List[dict]:
        """Training examples 증강 - 회전/반전 적용."""
        augmented = list(train_examples)  # Original

        # 각 example에 대해 augmentation 적용
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            # 회전 변형
            for k in [1, 2, 3]:  # 90, 180, 270도
                aug_input = np.rot90(input_grid, k=k)
                aug_output = np.rot90(output_grid, k=k)
                augmented.append({
                    'input': aug_input.tolist(),
                    'output': aug_output.tolist(),
                    'augmentation': f'rot{k*90}'
                })

            # 반전 변형
            for flip_func, name in [(np.fliplr, 'flip_h'), (np.flipud, 'flip_v')]:
                aug_input = flip_func(input_grid)
                aug_output = flip_func(output_grid)
                augmented.append({
                    'input': aug_input.tolist(),
                    'output': aug_output.tolist(),
                    'augmentation': name
                })

        return augmented

    def learn_rules(self, train_examples: List[dict]) -> List[Dict]:
        """Training examples에서 변환 규칙 학습."""
        rules = []

        # 규칙 1: 크기 변환 학습
        size_rule = self.learn_size_transform(train_examples[:len(train_examples)//2])
        if size_rule:
            rules.append(size_rule)

        # 규칙 2: 색상 매핑 학습
        color_rule = self.learn_color_mapping(train_examples[:len(train_examples)//2])
        if color_rule:
            rules.append(color_rule)

        # 규칙 3: 원자 연산 조합
        atomic_rules = self.find_atomic_combinations(train_examples[:len(train_examples)//2])
        rules.extend(atomic_rules[:5])

        # 규칙 4: 패턴 기반 (fallback)
        pattern_rules = self.extract_pattern_rules(train_examples[:len(train_examples)//2])
        rules.extend(pattern_rules[:3])

        return rules

    def learn_size_transform(self, train_examples: List[dict]) -> Dict:
        """크기 변환 규칙 학습."""
        size_ratios = []

        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            ratio = (
                output_grid.shape[0] / input_grid.shape[0],
                output_grid.shape[1] / input_grid.shape[1]
            )
            size_ratios.append(ratio)

        # 일관된 비율인지 확인
        if len(set(size_ratios)) == 1:
            ratio = size_ratios[0]
            return {
                'type': 'size_transform',
                'ratio': ratio,
                'apply': lambda g: self.resize_grid(g, ratio)
            }

        return None

    def resize_grid(self, grid: np.ndarray, ratio: Tuple[float, float]) -> np.ndarray:
        """Grid 크기 조정."""
        h_ratio, w_ratio = ratio

        if h_ratio == w_ratio == 0.5:
            # 반으로 축소
            return grid[::2, ::2]
        elif h_ratio == w_ratio == 2.0:
            # 2배 확대
            return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
        elif h_ratio == w_ratio == 3.0:
            # 3배 확대
            return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
        else:
            # 기타 비율
            new_h = int(grid.shape[0] * h_ratio)
            new_w = int(grid.shape[1] * w_ratio)
            return np.zeros((new_h, new_w), dtype=grid.dtype)

    def learn_color_mapping(self, train_examples: List[dict]) -> Dict:
        """색상 매핑 학습."""
        color_maps = []

        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            if input_grid.shape == output_grid.shape:
                # 위치별 색상 변화 추적
                mapping = {}
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        in_color = input_grid[i, j]
                        out_color = output_grid[i, j]
                        if in_color != out_color:
                            if in_color not in mapping:
                                mapping[in_color] = []
                            mapping[in_color].append(out_color)

                if mapping:
                    color_maps.append(mapping)

        # 일관된 매핑 찾기
        if color_maps:
            consistent_map = {}
            for color in set().union(*[set(m.keys()) for m in color_maps]):
                all_targets = []
                for m in color_maps:
                    if color in m:
                        all_targets.extend(m[color])

                if all_targets:
                    # 가장 흔한 target
                    most_common = Counter(all_targets).most_common(1)[0][0]
                    consistent_map[color] = most_common

            if consistent_map:
                return {
                    'type': 'color_mapping',
                    'mapping': consistent_map,
                    'apply': lambda g: self.apply_color_mapping(g, consistent_map)
                }

        return None

    def apply_color_mapping(self, grid: np.ndarray, mapping: Dict) -> np.ndarray:
        """색상 매핑 적용."""
        result = grid.copy()
        for in_color, out_color in mapping.items():
            result[grid == in_color] = out_color
        return result

    def find_atomic_combinations(self, train_examples: List[dict]) -> List[Dict]:
        """원자 연산 조합으로 해결 시도."""
        successful_ops = []

        # 단일 연산 테스트
        for op_name, op_func in self.atomic_ops.items():
            score = 0
            for ex in train_examples:
                input_grid = np.array(ex['input'])
                expected = np.array(ex['output'])

                try:
                    result = op_func(input_grid)
                    if result.shape == expected.shape:
                        if np.array_equal(result, expected):
                            score += 100
                except:
                    pass

            if score > 0:
                successful_ops.append({
                    'type': 'atomic',
                    'operation': op_name,
                    'score': score,
                    'apply': op_func
                })

        # 점수 순 정렬
        successful_ops.sort(key=lambda x: -x['score'])
        return successful_ops

    def extract_pattern_rules(self, train_examples: List[dict]) -> List[Dict]:
        """패턴 기반 규칙 추출."""
        rules = []

        # Same-size transformation인지 확인
        all_same_size = all(
            np.array(ex['input']).shape == np.array(ex['output']).shape
            for ex in train_examples
        )

        if all_same_size:
            # 변화한 셀 비율 계산
            change_ratios = []
            for ex in train_examples:
                input_grid = np.array(ex['input'])
                output_grid = np.array(ex['output'])
                changed = np.sum(input_grid != output_grid)
                total = input_grid.size
                change_ratios.append(changed / total)

            avg_change = np.mean(change_ratios)

            if avg_change < 0.05:
                # 거의 변화 없음 - identity 비슷
                rules.append({
                    'type': 'minimal_change',
                    'apply': lambda g: g.copy()
                })
            elif avg_change > 0.95:
                # 거의 모든 셀 변경
                rules.append({
                    'type': 'full_transform',
                    'apply': lambda g: 9 - g  # Invert
                })

        return rules

    def apply_rule(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """규칙 적용."""
        return rule['apply'](grid)


def main():
    print("ARC Prize 2025 - Learning + Augmentation Solver V6")
    print("=" * 70)

    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    solver = LearningSolver()
    submission = {}

    print("\nGenerating predictions with learning + augmentation...")
    for i, (task_id, task) in enumerate(test_challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_challenges)} tasks...")

        predictions = solver.solve(task, max_attempts=2)

        task_submission = []
        for test_predictions in predictions:
            test_dict = {
                "attempt_1": test_predictions[0],
                "attempt_2": test_predictions[1]
            }
            task_submission.append(test_dict)

        submission[task_id] = task_submission

    print(f"\n✓ Generated predictions for {len(submission)} tasks")

    output_path = Path('/kaggle/working/submission.json')
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"✓ Saved submission to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")

    print("\n" + "=" * 70)
    print("V6 Learning-based submission complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
