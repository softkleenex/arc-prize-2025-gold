"""
Local Evaluator for ARC Prize 2025
평가 데이터셋으로 로컬에서 성능 측정
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util
import sys


class LocalEvaluator:
    """로컬 평가 시스템"""

    def __init__(self, eval_data_path: str, solutions_path: str):
        self.eval_data_path = Path(eval_data_path)
        self.solutions_path = Path(solutions_path)
        self.challenges = self.load_challenges()
        self.solutions = self.load_solutions()

    def load_challenges(self) -> Dict:
        """평가 챌린지 로드"""
        with open(self.eval_data_path, 'r') as f:
            return json.load(f)

    def load_solutions(self) -> Dict:
        """평가 정답 로드"""
        with open(self.solutions_path, 'r') as f:
            return json.load(f)

    def load_solver(self, solver_file: str):
        """동적으로 solver 모듈 로드"""
        spec = importlib.util.spec_from_file_location("solver", solver_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["solver"] = module
        spec.loader.exec_module(module)
        return module

    def evaluate_solver(self, solver_file: str, max_attempts: int = 2) -> Dict:
        """특정 solver 평가"""
        print(f"\n{'='*70}")
        print(f"Evaluating: {solver_file}")
        print(f"{'='*70}\n")

        # Solver 로드
        try:
            module = self.load_solver(solver_file)

            # Solver 클래스 찾기
            solver_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and 'Solver' in name:
                    solver_class = obj
                    break

            if solver_class is None:
                print("Error: No Solver class found!")
                return {}

            solver = solver_class()
            print(f"✓ Loaded {solver_class.__name__}")

        except Exception as e:
            print(f"Error loading solver: {e}")
            return {}

        # 평가 실행
        results = {
            'total': len(self.challenges),
            'correct': 0,
            'attempt_1_correct': 0,
            'attempt_2_correct': 0,
            'failed_tasks': [],
            'success_tasks': [],
            'partial_match_tasks': []
        }

        for i, (task_id, task) in enumerate(self.challenges.items()):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(self.challenges)} tasks...")

            try:
                # 예측 생성
                predictions = solver.solve(task, max_attempts=max_attempts)

                # 정답과 비교
                expected = self.solutions[task_id]

                task_correct = False
                for test_idx, (pred_attempts, expected_outputs) in enumerate(zip(predictions, expected)):
                    attempt_1 = np.array(pred_attempts[0])
                    attempt_2 = np.array(pred_attempts[1])
                    expected_output = np.array(expected_outputs)

                    # Attempt 1 체크
                    if attempt_1.shape == expected_output.shape:
                        if np.array_equal(attempt_1, expected_output):
                            results['attempt_1_correct'] += 1
                            task_correct = True

                    # Attempt 2 체크 (Attempt 1 실패시)
                    if not task_correct:
                        if attempt_2.shape == expected_output.shape:
                            if np.array_equal(attempt_2, expected_output):
                                results['attempt_2_correct'] += 1
                                task_correct = True

                    # 부분 매치 체크
                    if not task_correct:
                        if attempt_1.shape == expected_output.shape:
                            match_ratio = np.sum(attempt_1 == expected_output) / expected_output.size
                            if match_ratio > 0.5:
                                results['partial_match_tasks'].append({
                                    'task_id': task_id,
                                    'match_ratio': match_ratio
                                })

                if task_correct:
                    results['correct'] += 1
                    results['success_tasks'].append(task_id)
                else:
                    results['failed_tasks'].append(task_id)

            except Exception as e:
                results['failed_tasks'].append(task_id)
                continue

        # 통계 계산
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['attempt_1_accuracy'] = (results['attempt_1_correct'] / results['total']) * 100
        results['attempt_2_accuracy'] = (results['attempt_2_correct'] / results['total']) * 100

        return results

    def print_results(self, results: Dict, solver_name: str):
        """결과 출력"""
        print(f"\n{'='*70}")
        print(f"Results for {solver_name}")
        print(f"{'='*70}\n")

        print(f"Total tasks: {results['total']}")
        print(f"Correct: {results['correct']} ({results['accuracy']:.2f}%)")
        print(f"  - Attempt 1: {results['attempt_1_correct']} ({results['attempt_1_accuracy']:.2f}%)")
        print(f"  - Attempt 2: {results['attempt_2_correct']} ({results['attempt_2_accuracy']:.2f}%)")
        print(f"Failed: {len(results['failed_tasks'])} ({100 - results['accuracy']:.2f}%)")
        print(f"Partial matches: {len(results['partial_match_tasks'])}")

        if results['partial_match_tasks']:
            print(f"\nTop 5 partial matches:")
            sorted_partial = sorted(results['partial_match_tasks'],
                                   key=lambda x: -x['match_ratio'])[:5]
            for item in sorted_partial:
                print(f"  - {item['task_id']}: {item['match_ratio']:.2%}")

        print(f"\n{'='*70}\n")

    def compare_solvers(self, solver_files: List[str]) -> Dict:
        """여러 solver 비교"""
        all_results = {}

        for solver_file in solver_files:
            solver_name = Path(solver_file).stem
            results = self.evaluate_solver(solver_file)
            all_results[solver_name] = results
            self.print_results(results, solver_name)

        # 비교 테이블
        print(f"\n{'='*70}")
        print("Comparison Summary")
        print(f"{'='*70}\n")

        print(f"{'Version':<25} {'Accuracy':<15} {'Attempt 1':<15} {'Attempt 2':<15}")
        print(f"{'-'*70}")

        for name, results in all_results.items():
            print(f"{name:<25} {results['accuracy']:>6.2f}% "
                  f"{results['attempt_1_accuracy']:>10.2f}% "
                  f"{results['attempt_2_accuracy']:>10.2f}%")

        print(f"\n{'='*70}\n")

        return all_results

    def analyze_failures(self, solver_file: str, num_tasks: int = 10):
        """실패 케이스 상세 분석"""
        print(f"\n{'='*70}")
        print(f"Failure Analysis: {solver_file}")
        print(f"{'='*70}\n")

        # Solver 로드
        module = self.load_solver(solver_file)
        solver_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and 'Solver' in name:
                solver_class = obj
                break

        solver = solver_class()

        failure_patterns = {
            'size_mismatch': [],
            'color_wrong': [],
            'pattern_wrong': [],
            'completely_wrong': []
        }

        analyzed_count = 0
        for task_id, task in self.challenges.items():
            if analyzed_count >= num_tasks:
                break

            try:
                predictions = solver.solve(task, max_attempts=2)
                expected = self.solutions[task_id]

                for test_idx, (pred_attempts, expected_outputs) in enumerate(zip(predictions, expected)):
                    attempt_1 = np.array(pred_attempts[0])
                    expected_output = np.array(expected_outputs)

                    # 정답이 아닌 경우만 분석
                    if not np.array_equal(attempt_1, expected_output):
                        failure_info = {
                            'task_id': task_id,
                            'input_shape': task['test'][test_idx]['input'],
                            'output_shape': expected_output.shape,
                            'predicted_shape': attempt_1.shape,
                        }

                        # 크기 불일치
                        if attempt_1.shape != expected_output.shape:
                            failure_patterns['size_mismatch'].append(failure_info)
                        else:
                            # 색상 분포 비교
                            expected_colors = set(expected_output.flatten())
                            predicted_colors = set(attempt_1.flatten())

                            if expected_colors != predicted_colors:
                                failure_patterns['color_wrong'].append(failure_info)
                            else:
                                # 패턴이 틀림
                                match_ratio = np.sum(attempt_1 == expected_output) / expected_output.size
                                if match_ratio > 0.3:
                                    failure_patterns['pattern_wrong'].append(failure_info)
                                else:
                                    failure_patterns['completely_wrong'].append(failure_info)

                        analyzed_count += 1
                        if analyzed_count >= num_tasks:
                            break
            except:
                continue

        # 결과 출력
        print(f"Analyzed {analyzed_count} failure cases:\n")

        for pattern_type, cases in failure_patterns.items():
            if cases:
                print(f"\n{pattern_type.upper()}: {len(cases)} cases")
                for case in cases[:3]:
                    print(f"  - Task: {case['task_id']}")
                    print(f"    Output shape: {case['output_shape']}, "
                          f"Predicted: {case['predicted_shape']}")

        return failure_patterns


def main():
    """메인 실행"""
    print("ARC Prize 2025 - Local Evaluator")
    print("=" * 70)

    # 경로 설정
    eval_data = "data/arc-agi_evaluation_challenges.json"
    solutions = "data/arc-agi_evaluation_solutions.json"

    evaluator = LocalEvaluator(eval_data, solutions)

    # V2-V5 비교
    solver_files = [
        "kaggle_notebook_v2.py",
        "kaggle_notebook_v3.py",
        "kaggle_notebook_v4.py",
        "kaggle_notebook_v5.py",
    ]

    results = evaluator.compare_solvers(solver_files)

    # 최고 성능 버전의 실패 케이스 분석
    best_solver = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest solver: {best_solver[0]} ({best_solver[1]['accuracy']:.2f}%)")
    print("\nAnalyzing failures of best solver...")

    failure_patterns = evaluator.analyze_failures(
        f"{best_solver[0]}.py",
        num_tasks=20
    )

    # 결과 저장
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'comparison': {k: {
                'accuracy': v['accuracy'],
                'attempt_1_accuracy': v['attempt_1_accuracy'],
                'attempt_2_accuracy': v['attempt_2_accuracy'],
                'correct': v['correct'],
                'total': v['total']
            } for k, v in results.items()},
            'best_solver': best_solver[0],
            'best_accuracy': best_solver[1]['accuracy']
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"\nNext steps:")
    print("1. 실패 케이스 패턴 기반으로 V6 개발")
    print("2. 로컬 평가로 V6 검증")
    print("3. 내일 V6 제출")


if __name__ == "__main__":
    main()
