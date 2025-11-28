#!/usr/bin/env python3
"""현재 상황 체크: 제출 이력, 커널 상태, 점수"""

from kaggle.api.kaggle_api_extended import KaggleApi
import json

api = KaggleApi()
api.authenticate()

print("="*80)
print("🔍 ARC Prize 2025 - 현재 상황 체크")
print("="*80)

# 1. 제출 이력 확인
print("\n📊 1. 제출 이력 (최근 5개)")
print("-"*80)
try:
    submissions = api.competitions_submissions_list("arc-prize-2025")
    if submissions:
        for i, sub in enumerate(submissions[:5], 1):
            print(f"{i}. 제출 날짜: {sub.date}")
            print(f"   상태: {sub.status}")
            print(f"   점수: {sub.publicScore if hasattr(sub, 'publicScore') and sub.publicScore else 'N/A'}")
            print(f"   파일: {sub.fileName if hasattr(sub, 'fileName') else 'N/A'}")
            print()
    else:
        print("   제출 이력 없음")
except Exception as e:
    print(f"   ❌ 에러: {e}")

# 2. 내 커널 목록
print("\n📁 2. 내 커널 목록 (ARC 관련)")
print("-"*80)
try:
    kernels = api.kernels_list(user="softkleenex", page_size=20)
    arc_kernels = [k for k in kernels if "arc" in k.title.lower()]

    if arc_kernels:
        for i, kernel in enumerate(arc_kernels, 1):
            print(f"{i}. {kernel.title}")
            print(f"   ID: {kernel.ref}")
            print(f"   URL: https://www.kaggle.com/code/{kernel.ref}")
            print()
    else:
        print("   ARC 관련 커널 없음")
except Exception as e:
    print(f"   ❌ 에러: {e}")

# 3. 로컬 준비된 파일들
print("\n💾 3. 로컬 준비된 커널 폴더")
print("-"*80)
import os

folders_to_check = [
    "easiest_first_kernel",
    "easiest_first_submit",
    "update_existing_kernel",
    "forked_kernel",
    "CompressARC"
]

for folder in folders_to_check:
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"✓ {folder}/")
        for f in files[:5]:  # 처음 5개만
            print(f"   - {f}")
    else:
        print(f"✗ {folder}/ (없음)")

# 4. 현재 최고 점수
print("\n🏆 4. 현재 최고 점수")
print("-"*80)
try:
    submissions = api.competitions_submissions_list("arc-prize-2025")
    if submissions:
        scores = [float(s.publicScore) for s in submissions if hasattr(s, 'publicScore') and s.publicScore]
        if scores:
            best_score = max(scores)
            print(f"   최고 점수: {best_score}")
            print(f"   목표 점수: 4.58 (Gold Medal)")
            print(f"   격차: {4.58 - best_score:.2f} ({((4.58/best_score - 1)*100):.1f}% 향상 필요)")
        else:
            print("   점수 데이터 없음")
    else:
        print("   제출 이력 없음")
except Exception as e:
    print(f"   ❌ 에러: {e}")

print("\n" + "="*80)
print("✅ 상황 체크 완료")
print("="*80)
