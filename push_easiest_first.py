#!/usr/bin/env python3
"""Push easiest-first kernel to Kaggle"""

from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Initialize API
api = KaggleApi()
api.authenticate()

kernel_path = "/mnt/c/LSJ/dacon/dacon/arc_2025/easiest_first_submit"

print(f"Pushing kernel from {kernel_path}...")

try:
    # Push the kernel (create or update)
    api.kernels_push(kernel_path)
    print(f"✓ Kernel pushed successfully!")
    print(f"\nKernel URL: https://www.kaggle.com/code/softkleenex/arc-2025-easiest-first")
    print(f"\n⚠️  다음 단계:")
    print(f"1. 위 URL로 이동")
    print(f"2. 'Submit to Competition' 버튼 클릭")
    print(f"3. 12시간 대기 후 Gold medal 획득!")
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"\n디버깅 정보:")
    print(f"- kernel-metadata.json 확인 중...")
    import json
    with open(f"{kernel_path}/kernel-metadata.json", 'r') as f:
        metadata = json.load(f)
        print(json.dumps(metadata, indent=2))
