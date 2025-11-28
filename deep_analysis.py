"""
Training 데이터 심층 분석
실제 문제 패턴 파악
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# Training 데이터 로드
with open('data/arc-agi_training_challenges.json', 'r') as f:
    train_data = json.load(f)
with open('data/arc-agi_training_solutions.json', 'r') as f:
    train_solutions = json.load(f)

print('=== TRAINING DATA DEEP ANALYSIS ===\n')

# 패턴 분석
patterns = {
    'same_size_transform': 0,
    'upscale': 0,
    'downscale': 0,
    'color_preserving': 0,
    'color_changing': 0,
    'high_match': 0,
}

sample_tasks = list(train_data.items())[:100]  # 첫 100개 분석

for task_id, task in sample_tasks:
    solution = train_solutions[task_id]

    for ex, sol in zip(task['train'], solution):
        in_grid = np.array(ex['input'])
        out_grid = np.array(sol)

        # Size 패턴
        if in_grid.shape == out_grid.shape:
            patterns['same_size_transform'] += 1

            # Match ratio
            match_ratio = np.sum(in_grid == out_grid) / in_grid.size
            if match_ratio > 0.9:
                patterns['high_match'] += 1
        elif out_grid.shape[0] > in_grid.shape[0]:
            patterns['upscale'] += 1
        else:
            patterns['downscale'] += 1

        # Color 패턴
        in_colors = set(in_grid.flatten())
        out_colors = set(out_grid.flatten())
        if in_colors == out_colors:
            patterns['color_preserving'] += 1
        else:
            patterns['color_changing'] += 1

print('패턴 분포 (100개 태스크의 train examples):')
total = sum(patterns.values())
for pattern, count in patterns.items():
    pct = count / total * 100 if total > 0 else 0
    print(f'  {pattern}: {count} ({pct:.1f}%)')

print('\n=== 특정 케이스 심층 분석 ===')

# 첫 5개 태스크 상세 분석
for i, (task_id, task) in enumerate(list(train_data.items())[:5]):
    print(f'\nTask {i+1}: {task_id}')
    print(f'  Train examples: {len(task["train"])}')
    print(f'  Test cases: {len(task["test"])}')

    # 첫 예제 분석
    ex = task['train'][0]
    sol = train_solutions[task_id][0]
    in_grid = np.array(ex['input'])
    out_grid = np.array(sol)

    print(f'  Input: {in_grid.shape}, colors: {sorted(set(in_grid.flatten()))}')
    print(f'  Output: {out_grid.shape}, colors: {sorted(set(out_grid.flatten()))}')

    # 변환 추론
    if in_grid.shape == out_grid.shape:
        match_ratio = np.sum(in_grid == out_grid) / in_grid.size
        print(f'  → Same size, {match_ratio:.1%} unchanged')

        if match_ratio > 0.9:
            # 어떤 부분이 바뀌었나?
            diff = np.argwhere(in_grid != out_grid)
            print(f'  → {len(diff)} cells changed')
            if len(diff) <= 10:
                print(f'    Changed positions: {diff.tolist()[:5]}...')
    else:
        h_ratio = out_grid.shape[0] / in_grid.shape[0]
        w_ratio = out_grid.shape[1] / in_grid.shape[1]
        print(f'  → Size transform: {h_ratio:.1f}x height, {w_ratio:.1f}x width')

print('\n=== 가장 흔한 변환 규칙 추정 ===')

# Size transform 분석
size_transforms = []
for task_id, task in list(train_data.items())[:100]:
    solution = train_solutions[task_id]

    for ex, sol in zip(task['train'], solution):
        in_grid = np.array(ex['input'])
        out_grid = np.array(sol)

        if in_grid.shape != out_grid.shape:
            h_ratio = round(out_grid.shape[0] / in_grid.shape[0], 1)
            w_ratio = round(out_grid.shape[1] / in_grid.shape[1], 1)
            size_transforms.append((h_ratio, w_ratio))

if size_transforms:
    common_transforms = Counter(size_transforms).most_common(10)
    print('\n가장 흔한 크기 변환:')
    for transform, count in common_transforms:
        print(f'  {transform}: {count}회')
else:
    print('\n대부분 same-size transformation')

print('\n' + '='*70)
print('분석 완료!')
