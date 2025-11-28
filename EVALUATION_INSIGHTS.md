# 🔍 로컬 평가 결과 분석

**평가 시각**: 2025-10-14 16:30
**평가 데이터**: 120 tasks (evaluation set)

---

## 📊 핵심 발견

### ⚠️ 충격적이지만 예상된 결과
```
모든 버전 (V2-V5): 0.00% 정확도
BUT: 97-98% Partial Matches!
```

**이것이 의미하는 것**:
- ✅ 우리의 접근 방향은 **거의 정확함**
- ❌ 하지만 ARC는 **100% 정확**해야만 점수 획득
- 💡 **1-3% 차이**가 0점과 만점의 차이

---

## 🎯 실패 패턴 분석

### 20개 실패 케이스 분류

#### 1. SIZE_MISMATCH (30% - 6 cases)
**문제**: 출력 크기를 잘못 예측
```
예시:
- Task 0934a4d8: 예상 (9, 3) → 예측 (30, 30)
- Task 136b0064: 예상 (19, 7) → 예측 (19, 15)
- Task 20270e3b: 예상 (5, 10) → 예측 (11, 10)
```

**원인**:
- 변환 함수들이 크기를 임의로 변경 (mirror, tile 등)
- Training examples에서 크기 변환 규칙을 학습하지 않음

**해결책**:
```python
def learn_size_transformation(train_examples):
    """Training examples에서 크기 변환 규칙 학습"""
    size_ratios = []
    for ex in train_examples:
        in_h, in_w = len(ex['input']), len(ex['input'][0])
        out_h, out_w = len(ex['output']), len(ex['output'][0])
        size_ratios.append((out_h / in_h, out_w / in_w))

    # 일관된 비율이 있는지 확인
    if all_same(size_ratios):
        return create_size_transform(size_ratios[0])
    return None
```

#### 2. PATTERN_WRONG (50% - 10 cases)
**문제**: 크기는 맞지만 패턴이 틀림
```
예시:
- Task 135a2760: 97.93% 매치 (거의 다 맞음!)
- Task 142ca369: 97%+ 매치
```

**원인**:
- 단순 변환으로는 복잡한 패턴 캡처 불가
- Training examples의 논리적 규칙을 이해하지 못함

**해결책**:
```python
def extract_transformation_logic(train_examples):
    """Training examples에서 변환 논리 추출"""
    # 1. 입력-출력 쌍 분석
    # 2. 공통 패턴 찾기
    # 3. 규칙 생성

    rules = []
    for ex in train_examples:
        input_grid = ex['input']
        output_grid = ex['output']

        # 변화한 부분 감지
        diff = find_differences(input_grid, output_grid)

        # 규칙 추출
        rule = infer_rule(diff)
        rules.append(rule)

    # 모든 예제에 적용되는 규칙 찾기
    common_rule = find_common_rule(rules)
    return common_rule
```

#### 3. COLOR_WRONG (15% - 3 cases)
**문제**: 색상 분포가 틀림
```
예시:
- Task 13e47133: 크기 맞음, 색상 틀림
- Task 16de56c4: 크기 맞음, 색상 틀림
```

**원인**:
- 색상 변환 규칙을 추측만 함
- Training examples에서 색상 매핑을 학습하지 않음

**해결책**:
```python
def learn_color_mapping(train_examples):
    """Training examples에서 색상 매핑 학습"""
    color_map = {}

    for ex in train_examples:
        in_colors = get_unique_colors(ex['input'])
        out_colors = get_unique_colors(ex['output'])

        # 색상 매핑 추출
        mapping = extract_color_mapping(
            ex['input'],
            ex['output']
        )

        for in_c, out_c in mapping.items():
            if in_c not in color_map:
                color_map[in_c] = []
            color_map[in_c].append(out_c)

    # 일관된 매핑 찾기
    consistent_map = find_consistent_mapping(color_map)
    return consistent_map
```

#### 4. COMPLETELY_WRONG (5% - 1 case)
**문제**: 완전히 틀림 (30% 미만 매치)

---

## 💡 근본적인 문제

### 현재 접근의 한계

```python
# 현재 방식 (V1-V5)
class CurrentApproach:
    """미리 정의된 변환 함수들"""

    transformations = [
        flip, rotate, scale, tile, ...
    ]

    def solve(task):
        # Training examples로 최적 변환 찾기
        best_transforms = find_best(task['train'])

        # Test input에 적용
        return apply_transforms(test_input, best_transforms)
```

**문제점**:
1. ❌ 변환 함수가 **고정됨** - 새로운 패턴 처리 불가
2. ❌ Training examples를 **얕게만** 활용
3. ❌ **논리적 추론** 없음 - 그냥 패턴 매칭

### 필요한 접근

```python
# 새로운 방식 (V6+)
class LearningApproach:
    """Training examples에서 학습"""

    def solve(task):
        # 1. Training examples 깊이 분석
        patterns = analyze_train_examples(task['train'])

        # 2. 변환 규칙 추출
        rules = extract_rules(patterns)

        # 3. 프로그램 합성
        program = synthesize_program(rules)

        # 4. Test input에 적용
        return execute_program(test_input, program)
```

**개선점**:
1. ✅ **동적 프로그램 생성** - 각 태스크마다 맞춤 솔루션
2. ✅ Training examples를 **깊게 학습**
3. ✅ **논리적 추론** 포함

---

## 🚀 V6 개발 전략

### 핵심 아이디어: "Learn, Don't Guess"

#### 1. Training Examples 분석기
```python
class TrainingAnalyzer:
    """Training examples에서 패턴 추출"""

    def analyze(self, train_examples):
        analysis = {
            'size_transform': self.analyze_size_changes(train_examples),
            'color_mapping': self.analyze_color_changes(train_examples),
            'spatial_patterns': self.analyze_spatial_patterns(train_examples),
            'object_operations': self.analyze_object_ops(train_examples),
        }
        return analysis

    def analyze_size_changes(self, examples):
        """크기 변환 규칙 학습"""
        ratios = []
        for ex in examples:
            in_shape = np.array(ex['input']).shape
            out_shape = np.array(ex['output']).shape
            ratio = (out_shape[0] / in_shape[0],
                    out_shape[1] / in_shape[1])
            ratios.append(ratio)

        if all(r == ratios[0] for r in ratios):
            return {'type': 'consistent', 'ratio': ratios[0]}
        return {'type': 'variable'}
```

#### 2. 규칙 기반 프로그램 합성
```python
class RuleBasedSynthesizer:
    """학습된 규칙으로 프로그램 생성"""

    def synthesize(self, analysis, train_examples):
        program = []

        # 크기 변환
        if analysis['size_transform']['type'] == 'consistent':
            ratio = analysis['size_transform']['ratio']
            program.append(('resize', ratio))

        # 색상 매핑
        if analysis['color_mapping']:
            program.append(('remap_colors', analysis['color_mapping']))

        # 공간 패턴
        if analysis['spatial_patterns']:
            program.append(('apply_pattern', analysis['spatial_patterns']))

        return program

    def execute(self, grid, program):
        """프로그램 실행"""
        result = grid.copy()
        for operation, params in program:
            result = self.apply_operation(result, operation, params)
        return result
```

#### 3. 검증 시스템
```python
def verify_program(program, train_examples):
    """Training examples로 프로그램 검증"""
    for ex in train_examples:
        predicted = execute_program(ex['input'], program)
        expected = ex['output']

        if not np.array_equal(predicted, expected):
            return False, predicted, expected

    return True, None, None
```

---

## 📈 예상 개선

### V5 → V6

| 측면 | V5 | V6 (목표) |
|------|-------|-----------|
| **접근** | 고정된 변환 | 동적 프로그램 합성 |
| **학습** | 얕은 매칭 | 깊은 규칙 학습 |
| **크기 예측** | 임의 | 학습된 규칙 |
| **색상 처리** | 추측 | 학습된 매핑 |
| **예상 정확도** | 0% | 1-5% |

---

## 🎓 배운 점

### 1. ARC의 진짜 난이도
- **97-98% 맞음 = 0점**
- **100% 정확 = 만점**
- 작은 차이가 큰 차이

### 2. 단순 변환의 한계
- 미리 정의된 함수로는 불충분
- 각 태스크마다 **맞춤 솔루션** 필요

### 3. Training Examples의 중요성
- 단순히 "어떤 변환이 맞나?" 찾기 ❌
- "무슨 **규칙**으로 변환하나?" 학습 ✅

---

## 🔬 심층 분석 필요

### 즉시 분석할 케이스

#### Case 1: 135a2760 (97.93% 매치)
```
거의 다 맞았는데 뭐가 틀렸을까?
→ Training examples 보고 정확히 뭘 놓쳤는지 분석
```

#### Case 2: 크기 불일치 케이스들
```
왜 크기를 잘못 예측했을까?
→ Training examples의 크기 변환 규칙 찾기
```

#### Case 3: 색상 틀린 케이스들
```
색상 규칙이 뭐였을까?
→ Training examples의 색상 매핑 추출
```

---

## 📋 다음 단계

### 즉시 (지금)
- [x] 로컬 평가 완료
- [x] 실패 패턴 분석
- [ ] 3-5개 케이스 심층 분석
- [ ] Public notebooks 연구

### 오늘 (6시간 내)
- [ ] V6 개발 (Learning-based)
- [ ] V6 로컬 검증
- [ ] V7 개발 (DSL-based)

### 내일
- [ ] V6, V7 제출
- [ ] 점수 확인
- [ ] 추가 개선

---

## 💪 긍정적인 면

1. ✅ **방향은 올바름** - 97-98% 매치
2. ✅ **로컬 평가 시스템** 구축 완료
3. ✅ **실패 패턴** 명확히 파악
4. ✅ **개선 방향** 명확함

**다음 목표**: 97% → 100% 만들기!

---

**현재 상태**: 실패 원인 파악 완료 ✓
**다음 액션**: 심층 분석 + V6 개발
