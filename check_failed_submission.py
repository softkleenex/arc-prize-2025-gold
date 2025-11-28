#!/usr/bin/env python3
"""실패한 제출의 로그 확인"""

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("="*80)
print("🔍 실패한 커널 분석")
print("="*80)

# Easiest-First 커널 확인
kernel_ref = "softkleenex/arc-compressarc-easiest-first-strategy"

print(f"\n커널: {kernel_ref}")
print("-"*80)

try:
    # 커널 출력 다운로드
    print("커널 출력 다운로드 중...")
    api.kernels_output(kernel_ref, path="failed_kernel_output")
    print("✓ 다운로드 완료: failed_kernel_output/")

    # 로그 확인
    import os
    if os.path.exists("failed_kernel_output"):
        files = os.listdir("failed_kernel_output")
        print(f"\n생성된 파일: {files}")

        # submission.json 확인
        if "submission.json" in files:
            import json
            with open("failed_kernel_output/submission.json", 'r') as f:
                sub = json.load(f)
            print(f"\nsubmission.json 확인:")
            print(f"  - Task 개수: {len(sub)}")
            if len(sub) > 0:
                first_task = list(sub.keys())[0]
                print(f"  - 첫 번째 task: {first_task}")
                print(f"  - 데이터: {sub[first_task][:100]}...")

        # 로그 파일 확인
        log_files = [f for f in files if f.endswith('.log')]
        if log_files:
            print(f"\n로그 파일: {log_files[0]}")
            with open(f"failed_kernel_output/{log_files[0]}", 'r') as f:
                log_content = f.read()
            print(f"로그 내용 (마지막 50줄):")
            print("-"*80)
            print('\n'.join(log_content.split('\n')[-50:]))

except Exception as e:
    print(f"❌ 에러: {e}")

print("\n" + "="*80)
