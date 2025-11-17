#!/usr/bin/env python3
"""
Validate the fixed_easiest_first.ipynb notebook
"""

import json
import ast

def validate_notebook(notebook_path):
    """노트북 검증"""
    print(f"Validating {notebook_path}...")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # 1. 기본 구조 검증
    print("\n1. Basic structure check:")
    print(f"   - Number of cells: {len(nb['cells'])}")

    code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
    markdown_cells = [c for c in nb['cells'] if c.get('cell_type') == 'markdown']
    print(f"   - Code cells: {len(code_cells)}")
    print(f"   - Markdown cells: {len(markdown_cells)}")

    # 2. Import 검증
    print("\n2. Import validation:")
    required_imports = [
        'torch', 'numpy', 'json', 'multiprocessing',
        'solve_task', 'visualization', 'solution_selection'
    ]

    all_code = ''
    for cell in code_cells:
        code = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        all_code += code + '\n'

    for imp in required_imports:
        if f'import {imp}' in all_code or f'from {imp}' in all_code:
            print(f"   ✅ {imp} imported")
        else:
            print(f"   ⚠️  {imp} might not be imported directly")

    # 3. 중요 함수 검증
    print("\n3. Key functions check:")
    key_functions = ['parallelize_runs', 'set_all_seeds']
    for func in key_functions:
        if f'def {func}' in all_code:
            print(f"   ✅ {func} defined")
        else:
            print(f"   ❌ {func} NOT found")

    # 4. submission.json 생성 코드 검증
    print("\n4. submission.json generation check:")
    if "with open('submission.json', 'w')" in all_code:
        print("   ✅ submission.json write code found")
    else:
        print("   ❌ submission.json write code NOT found")

    if "json.dump(solutions_dict" in all_code:
        print("   ✅ solutions_dict dump found")
    else:
        print("   ❌ solutions_dict dump NOT found")

    # 5. Visualization 코드 확인 (없어야 함)
    print("\n5. Visualization code check (should be removed):")
    viz_patterns = [
        'visualize_arc_results',
        'plot_arc_task',
        'matplotlib',
        'plt.show()',
        'if fake_mode:\n    #'
    ]

    viz_found = False
    for pattern in viz_patterns:
        if pattern in all_code:
            print(f"   ⚠️  Found: {pattern}")
            viz_found = True

    if not viz_found:
        print("   ✅ No visualization code found (good!)")

    # 6. GPU 설정 확인
    print("\n6. GPU configuration check:")
    if "torch.cuda.device_count()" in all_code:
        print("   ✅ GPU detection code found")
    if "torch.set_default_device('cuda')" in all_code:
        print("   ✅ CUDA default device set")

    # 7. 경로 확인
    print("\n7. Path validation:")
    paths = [
        '/kaggle/input/publiccompressarc',
        '../input/arc-prize-2025/arc-agi_'
    ]
    for path in paths:
        if path in all_code:
            print(f"   ✅ Path found: {path}")

    # 8. 최종 셀 확인
    print("\n8. Final cell check:")
    last_code_cell = code_cells[-1] if code_cells else None
    if last_code_cell:
        last_code = ''.join(last_code_cell['source']) if isinstance(last_code_cell['source'], list) else last_code_cell['source']
        if "submission.json created successfully" in last_code:
            print("   ✅ Confirmation message found in last cell")
        else:
            print("   ⚠️  Confirmation message not in last cell")

    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    validate_notebook('fixed_easiest_first.ipynb')