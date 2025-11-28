"""
제출 전 자동 검증 스크립트
사용: python validate_submission.py

이 스크립트는 submission.json 파일을 검증하여
Kaggle 제출 형식 오류를 사전에 방지합니다.
"""

import json
import os
from typing import Dict, List

def validate_submission_format(submission: Dict) -> bool:
    """제출 형식 전체 검증"""

    print("=" * 70)
    print("제출 형식 검증 시작")
    print("=" * 70)

    # 1. Task 수 확인
    print("\n[1/5] Task 수 확인...")
    if len(submission) != 240:
        print(f"  ❌ Task 수 오류: {len(submission)} (필요: 240)")
        return False
    print(f"  ✓ Task 수 정확: {len(submission)}")

    # 2. 형식 검증
    print("\n[2/5] 형식 구조 검증...")
    errors = []

    for task_id, predictions in list(submission.items())[:10]:  # 처음 10개만
        if not isinstance(predictions, list):
            errors.append(f"{task_id}: predictions는 list여야 함")
            continue

        for i, pred_dict in enumerate(predictions):
            if not isinstance(pred_dict, dict):
                errors.append(f"{task_id} test {i}: dict여야 함, {type(pred_dict)} 발견")
                continue

            if 'attempt_1' not in pred_dict or 'attempt_2' not in pred_dict:
                errors.append(f"{task_id} test {i}: 'attempt_1', 'attempt_2' 키 필요")
                errors.append(f"  실제 키: {list(pred_dict.keys())}")
                continue

            for attempt_key in ['attempt_1', 'attempt_2']:
                attempt = pred_dict[attempt_key]
                if not isinstance(attempt, list):
                    errors.append(f"{task_id} test {i} {attempt_key}: list여야 함")
                    continue
                if len(attempt) > 0 and not isinstance(attempt[0], list):
                    errors.append(f"{task_id} test {i} {attempt_key}: 2D list여야 함")

    if errors:
        print("  ❌ 형식 오류 발견:")
        for error in errors[:5]:  # 처음 5개만 출력
            print(f"     - {error}")
        if len(errors) > 5:
            print(f"     ... 외 {len(errors) - 5}개 오류")
        return False

    print("  ✓ 형식 구조 정확")

    # 3. 파일 크기 확인
    print("\n[3/5] 파일 크기 확인...")
    size_mb = os.path.getsize('submission.json') / (1024 * 1024)
    print(f"  ✓ 파일 크기: {size_mb:.2f} MB")

    if size_mb > 100:
        print(f"  ⚠️ 경고: 파일이 매우 큼 ({size_mb:.2f} MB)")
    elif size_mb < 0.1:
        print(f"  ⚠️ 경고: 파일이 너무 작음 ({size_mb:.2f} MB)")

    # 4. Sample과 비교 (있는 경우)
    print("\n[4/5] Sample과 비교...")
    if os.path.exists('sample_submission.json'):
        with open('sample_submission.json', 'r') as f:
            sample = json.load(f)

        sample_task_id = list(sample.keys())[0]
        our_task_id = list(submission.keys())[0]

        sample_structure = type(sample[sample_task_id])
        our_structure = type(submission[our_task_id])

        if sample_structure == our_structure:
            print("  ✓ 구조 일치")
        else:
            print(f"  ❌ 구조 불일치:")
            print(f"     Sample: {sample_structure}")
            print(f"     Ours: {our_structure}")
            return False

        # 키 비교
        if len(sample[sample_task_id]) > 0 and len(submission[our_task_id]) > 0:
            sample_keys = set(sample[sample_task_id][0].keys())
            our_keys = set(submission[our_task_id][0].keys())

            if sample_keys == our_keys:
                print("  ✓ 키 이름 일치")
            else:
                print(f"  ❌ 키 불일치:")
                print(f"     Sample: {sample_keys}")
                print(f"     Ours: {our_keys}")
                return False
    else:
        print("  ⚠️ sample_submission.json 없음 (건너뜀)")

    # 5. 최종 검증
    print("\n[5/5] 최종 검증...")
    sample_task_id = list(submission.keys())[0]
    sample_pred = submission[sample_task_id]

    print(f"  Sample task: {sample_task_id}")
    print(f"  Type: {type(sample_pred)}")
    print(f"  Length: {len(sample_pred)}")
    if len(sample_pred) > 0:
        print(f"  Keys: {list(sample_pred[0].keys())}")
        attempt_1 = sample_pred[0]['attempt_1']
        if len(attempt_1) > 0 and len(attempt_1[0]) > 0:
            print(f"  attempt_1 shape: {len(attempt_1)}x{len(attempt_1[0])}")

    print("\n" + "=" * 70)
    print("✅ 모든 검증 통과!")
    print("=" * 70)
    print("\n다음 단계:")
    print("  1. kaggle kernels push -p .")
    print("  2. 커널 페이지에서 실행 확인 (5-10분)")
    print("  3. Submit to Competition 클릭")
    print("  4. 점수 확인 대기")
    print("=" * 70)

    return True


def check_dataset_file():
    """올바른 데이터셋 파일 사용 확인"""

    print("\n[보너스] 데이터셋 파일 확인...")

    # 노트북 파일 확인
    notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb') and 'submission' in f.lower()]

    for nb_file in notebook_files:
        with open(nb_file, 'r', encoding='utf-8') as f:
            content = f.read()

            if 'test_challenges.json' in content:
                print(f"  ✓ {nb_file}: test_challenges.json 사용")
            elif 'evaluation_challenges.json' in content:
                print(f"  ❌ {nb_file}: evaluation_challenges.json 사용 (잘못됨!)")
                print(f"     → test_challenges.json으로 변경 필요")
                return False
            else:
                print(f"  ⚠️ {nb_file}: 데이터셋 파일 로드 확인 안됨")

    return True


def main():
    """메인 실행"""

    print("\n" + "=" * 70)
    print("ARC Prize 2025 - 제출 검증 스크립트")
    print("=" * 70)

    if not os.path.exists('submission.json'):
        print("\n❌ submission.json 파일이 없습니다!")
        print("\n다음을 실행하세요:")
        print("  python generate_correct_submission.py")
        return

    # submission.json 검증
    with open('submission.json', 'r') as f:
        submission = json.load(f)

    if not validate_submission_format(submission):
        print("\n❌ 검증 실패 - 수정 필요")
        return

    # 데이터셋 파일 확인
    if not check_dataset_file():
        print("\n❌ 노트북 파일에서 잘못된 데이터셋 사용 감지")
        return

    print("\n✅ 모든 검증 완료! 제출 준비 완료!")


if __name__ == "__main__":
    main()
