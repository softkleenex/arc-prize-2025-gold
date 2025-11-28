#!/usr/bin/env python3
"""Find the actual kernel URL"""

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

print("내 커널 목록 조회 중...\n")

# Get my kernels
kernels = api.kernels_list(user="softkleenex", page_size=20)

print(f"총 {len(kernels)} 개의 커널 발견:\n")

for i, kernel in enumerate(kernels, 1):
    print(f"{i}. {kernel.ref}")
    print(f"   Title: {kernel.title}")
    print(f"   URL: https://www.kaggle.com/code/{kernel.ref}")
    print()

# Find the easiest-first kernel
easiest_kernels = [k for k in kernels if "easiest" in k.title.lower() or "gold" in k.title.lower()]

if easiest_kernels:
    print("\n🎯 Easiest-First 커널 발견!")
    for k in easiest_kernels:
        print(f"   ✓ https://www.kaggle.com/code/{k.ref}")
