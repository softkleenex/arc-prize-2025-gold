#!/usr/bin/env python3
"""커널 실행 상태 확인"""

from kaggle.api.kaggle_api_extended import KaggleApi
import time

api = KaggleApi()
api.authenticate()

kernel_ref = "softkleenex/arc-compressarc-easiest-first-strategy"

print("="*80)
print(f"🔍 커널 상태 확인: {kernel_ref}")
print("="*80)

try:
    status = api.kernel_status(kernel_ref)

    print(f"\n현재 상태:")
    print(f"  Status: {status.get('status', 'N/A')}")
    print(f"  Failure Message: {status.get('failureMessage', 'None')}")

    if 'status' in status:
        current_status = status['status']

        if current_status == 'running':
            print(f"\n⏳ 커널 실행 중...")
            print(f"  → Run All이 진행 중입니다")
            print(f"  → 완료될 때까지 기다려주세요")

        elif current_status == 'complete':
            print(f"\n✓ 커널 실행 완료!")
            print(f"  → 이제 'Submit to Competition' 클릭 가능")

        elif current_status == 'error':
            print(f"\n❌ 커널 실행 실패!")
            print(f"  → 에러 메시지 확인 필요")

        else:
            print(f"\n상태: {current_status}")

except Exception as e:
    print(f"❌ 에러: {e}")

print("\n" + "="*80)
print("커널 페이지: https://www.kaggle.com/code/softkleenex/arc-compressarc-easiest-first-strategy")
print("="*80)
