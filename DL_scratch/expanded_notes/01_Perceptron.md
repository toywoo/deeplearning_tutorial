# [Expanded] 퍼셉트론 (Perceptron)

이 문서는 `1_2_perceptron.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"퍼셉트론은 어떻게 단순한 입력들의 조합으로 논리적인 판단을 내리고, 더 복잡한 비선형적 문제를 해결할 수 있는가?"**

퍼셉트론(Perceptron)은 딥러닝과 신경망의 가장 기초가 되는 기본 단위(Building Block)입니다. 다수의 신호를 입력으로 받아 0과 1 (또는 -1과 1)이라는 하나의 신호를 출력하는 알고리즘으로, 기계가 어떻게 간단한 연산을 수행하고 이를 조합해 복잡한 지능을 모사할 수 있는지에 대한 출발점을 제공합니다.

## 2. Key Mechanism (해결 원리)

### 단층 퍼셉트론과 논리 게이트 (Single-layer Perceptron & Logic Gates)
퍼셉트론은 각각의 입력 신호($x$)에 고유한 가중치($w$)를 곱한 후, 이들의 총합이 임계값을 넘을 때 활성화(1 출력)됩니다.
- **가중치 (Weight)**: 입력 신호가 결과에 미치는 중요도를 조절합니다.
- **편향 (Bias)**: 뉴런이 얼마나 쉽게 활성화(1을 출력)되는지를 조정하는 매개변수입니다. (임계값 $\theta$를 $-b$로 치환하여 사용)

가중치와 편향의 값을 적절히 세팅함으로써 신경망 모델이 파라미터를 통해 학습하는 기본 원리를 맛볼 수 있습니다. AND, NAND, OR 게이트는 모두 동일한 퍼셉트론 구조를 가지지만, 오직 **가중치와 편향 값의 차이**만으로 서로 다른 논리 연산을 수행합니다.

### 다층 퍼셉트론과 비선형성 (Multi-layer Perceptron & Non-linearity)
단층 퍼셉트론의 치명적인 한계는 선형 영역(직선) 하나로만 공간을 나눌 수 있다는 점입니다. 따라서 선형적으로 분리할 수 없는 **XOR (배타적 논리합)** 문제는 단층 퍼셉트론으로 해결할 수 없습니다.

이를 극복하기 위해 퍼셉트론을 층층이 쌓아올린 **다층 퍼셉트론 (Multi-layer Perceptron, MLP)**이 등장했습니다. 기존의 NAND, OR, AND 게이트를 조합(Layering)하여 비선형적인 결정 경계(Non-linear Decision Boundary)를 만들어냄으로써 XOR 연산을 성공적으로 구현할 수 있게 됩니다.

## 3. Mathematical Foundation (수학적 기반)

### 단층 퍼셉트론의 수식
가중치를 $w$, 편향을 $b$, 입력을 $x$라고 할 때, 퍼셉트론의 출력 $y$는 다음과 같은 1차 선형 방정식과 계단 함수(Step Function)로 표현됩니다.

$$
y = \begin{cases} 
0 & (\sum_{i} w_ix_i + b \le 0) \\ 
1 & (\sum_{i} w_ix_i + b > 0) 
\end{cases}
$$

- $w \cdot x$: 입력 벡터와 가중치 벡터의 내적 (Dot Product)
- 컴퓨터의 기계학습(Machine Learning)이란, 목표하는 출력(예: AND 테이블)을 만들기 위해 이 식을 만족하는 최적의 **$w$와 $b$ 값을 찾는 과정**입니다.

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return 1 if np.sum(w*x) + b > 0 else 0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return 1 if np.sum(w*x) + b > 0 else 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return 1 if np.sum(w*x) + b > 0 else 0

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

print("--- 퍼셉트론 논리 게이트 테스트 ---")
for xs in [(0,0), (1,0), (0,1), (1,1)]:
    print(f"XOR{xs} = {XOR(*xs)}")
```

## 🚀 추가 학습 및 실습 제안

1. **[코딩 실습] 기하학적 결정 경계(Decision Boundary) 시각화**
   - Matplotlib을 활용하여 $w_1x_1 + w_2x_2 + b = 0$ 이라는 직선의 방정식이 2차원 평면(x1, x2)에서 AND와 OR의 데이터 포인트를 어떻게 나누고 있는지 직접 그려보세요.
2. **[수학적 고찰] 퍼셉트론 학습 규칙 (Perceptron Learning Rule)**
   - 본 코드는 가중치를 사람이 직접 하드코딩(`w1, w2 = 0.5, 0.5`)했습니다. 퍼셉트론이 오차를 기반으로 스스로 가중치를 업데이트하는 규칙($w \leftarrow w + \alpha (y - \hat{y}) x$)을 수학적으로 찾아보고 적용해보세요.
3. **[논문 리뷰 제안] Marvin Minsky & Seymour Papert, "Perceptrons" (1969)**
   - "단일 다층 퍼셉트론으로는 XOR 문제를 풀 수 없다"는 것을 수학적으로 증명하여 첫 번째 AI 겨울(AI Winter)을 촉발시켰던 전설적인 논문의 핵심 요약본을 찾아 읽어보며 역사적 맥락을 이해해 보세요.
