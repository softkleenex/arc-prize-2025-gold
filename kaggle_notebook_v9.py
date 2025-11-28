"""
ARC Prize 2025 - Version 9: Data-Driven Approach

Training data 분석 기반:
- Color transformation 44.5% → 색상 변환 집중!
- Upscale 18% → 정교한 크기 변환
- Pattern learning from actual data
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


class DataDrivenSolver:
    """실제 Training 데이터 패턴 기반 solver"""

    def __init__(self):
        # 실제 데이터에서 발견한 흔한 변환들
        self.common_transforms = self._init_common_transforms()

    def _init_common_transforms(self):
        """데이터 분석 기반 변환들"""
        return {
            # Color transforms (44.5%가 color changing!)
            'color_shift_1': lambda g: (g + 1) % 10,
            'color_shift_2': lambda g: (g + 2) % 10,
            'color_map_mod3': lambda g: g % 3,
            'color_map_mod5': lambda g: g % 5,
            'color_binary': lambda g: (g > 0).astype(g.dtype),

            # Upscale (18%, 가장 흔한 것들)
            'scale_3x': lambda g: np.repeat(np.repeat(g, 3, axis=0), 3, axis=1),
            'scale_2x': lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),

            # Geometric
            'rot90': lambda g: np.rot90(g, k=-1),
            'rot180': lambda g: np.rot90(g, k=2),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),

            # Identity
            'identity': lambda g: g.copy(),
        }

    def solve(self, task: dict, max_attempts: int = 2) -> List[List[List[int]]]:
        """데이터 기반 솔루션"""
        predictions = []

        # Training examples 깊이 분석
        analysis = self.analyze_training(task['train'])

        # 분석 기반 최적 변환 찾기
        best_transforms = self.find_transforms_by_analysis(task['train'], analysis)

        for test_example in task['test']:
            test_input = np.array(test_example['input'])

            attempts = []
            for transform_func in best_transforms[:max_attempts]:
                try:
                    output = transform_func(test_input)
                    attempts.append(output.tolist())
                except:
                    attempts.append(test_input.tolist())

            while len(attempts) < max_attempts:
                attempts.append(test_input.tolist())

            predictions.append(attempts)

        return predictions

    def analyze_training(self, train_examples: List[dict]) -> Dict:
        """Training examples 심층 분석"""
        analysis = {
            'size_pattern': None,
            'color_pattern': None,
            'dominant_transform': None,
        }

        # Size 패턴
        size_transforms = []
        for ex in train_examples:
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])

            if in_grid.shape != out_grid.shape:
                h_ratio = out_grid.shape[0] / in_grid.shape[0]
                w_ratio = out_grid.shape[1] / in_grid.shape[1]
                size_transforms.append((round(h_ratio, 1), round(w_ratio, 1)))

        if size_transforms and len(set(size_transforms)) == 1:
            analysis['size_pattern'] = size_transforms[0]

        # Color 패턴
        color_preserving = 0
        color_changing = 0

        for ex in train_examples:
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])

            in_colors = set(in_grid.flatten())
            out_colors = set(out_grid.flatten())

            if in_colors == out_colors:
                color_preserving += 1
            else:
                color_changing += 1

        if color_changing > color_preserving:
            analysis['color_pattern'] = 'changing'

            # 어떤 색상 변환인지 추론
            color_diffs = []
            for ex in train_examples:
                in_grid = np.array(ex['input'])
                out_grid = np.array(ex['output'])

                if in_grid.shape == out_grid.shape:
                    # 위치별 색상 차이
                    diff = out_grid - in_grid
                    non_zero_diff = diff[diff != 0]
                    if len(non_zero_diff) > 0:
                        avg_diff = int(np.mean(non_zero_diff))
                        color_diffs.append(avg_diff)

            if color_diffs:
                analysis['color_shift'] = Counter(color_diffs).most_common(1)[0][0]
        else:
            analysis['color_pattern'] = 'preserving'

        return analysis

    def find_transforms_by_analysis(self, train_examples: List[dict], analysis: Dict) -> List:
        """분석 결과 기반 변환 찾기"""
        candidates = []

        # Size pattern 기반
        if analysis['size_pattern']:
            h_ratio, w_ratio = analysis['size_pattern']

            if h_ratio == 3.0 and w_ratio == 3.0:
                candidates.append(self.common_transforms['scale_3x'])
            elif h_ratio == 2.0 and w_ratio == 2.0:
                candidates.append(self.common_transforms['scale_2x'])

        # Color pattern 기반
        if analysis['color_pattern'] == 'changing':
            if 'color_shift' in analysis:
                shift = analysis['color_shift']
                if shift == 1:
                    candidates.append(self.common_transforms['color_shift_1'])
                elif shift == 2:
                    candidates.append(self.common_transforms['color_shift_2'])

            # 일반적인 color transforms 시도
            candidates.extend([
                self.common_transforms['color_map_mod3'],
                self.common_transforms['color_binary'],
            ])

        # 기하학적 변환
        for name in ['rot90', 'flip_h', 'flip_v']:
            candidates.append(self.common_transforms[name])

        # 검증된 것만 반환
        verified = []
        for candidate in candidates:
            if self.verify_transform(candidate, train_examples):
                verified.append(candidate)

        # 검증된 게 없으면 전체 시도
        if not verified:
            verified = list(self.common_transforms.values())

        return verified[:10]

    def verify_transform(self, transform_func, train_examples: List[dict]) -> bool:
        """변환이 training examples에 맞는지 검증"""
        for ex in train_examples:
            try:
                in_grid = np.array(ex['input'])
                expected = np.array(ex['output'])
                result = transform_func(in_grid)

                if result.shape != expected.shape:
                    return False

                if not np.array_equal(result, expected):
                    return False
            except:
                return False

        return True


class ColorTransformLearner:
    """색상 변환 학습 전문"""

    @staticmethod
    def learn_color_mapping(train_examples: List[dict]) -> Dict:
        """Color mapping 학습"""
        mappings = {}

        for ex in train_examples:
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])

            if in_grid.shape == out_grid.shape:
                for i in range(in_grid.shape[0]):
                    for j in range(in_grid.shape[1]):
                        in_color = in_grid[i, j]
                        out_color = out_grid[i, j]

                        if in_color not in mappings:
                            mappings[in_color] = []
                        mappings[in_color].append(out_color)

        # 일관된 매핑만 추출
        consistent_mapping = {}
        for in_color, out_colors in mappings.items():
            counter = Counter(out_colors)
            most_common = counter.most_common(1)[0]

            # 80% 이상 일관성
            if most_common[1] / len(out_colors) > 0.8:
                consistent_mapping[in_color] = most_common[0]

        return consistent_mapping


def main():
    print("ARC Prize 2025 - Data-Driven Approach V9")
    print("=" * 70)

    input_path = Path('/kaggle/input/arc-prize-2025')

    print("\nLoading test data...")
    with open(input_path / 'arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test tasks")

    solver = DataDrivenSolver()
    submission = {}

    print("\nGenerating predictions with data-driven approach...")
    print("(Based on training data pattern analysis)")

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
    print("V9 Data-Driven submission complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
