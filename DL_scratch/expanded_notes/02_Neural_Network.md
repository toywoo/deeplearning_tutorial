# [Expanded] 신경망 (Neural Network)

이 문서는 `1_3_neural_network.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"다층 퍼셉트론의 가중치를 사람이 직접 설정하는 한계를 벗어나, 기계가 스스로 매개변수를 자동 학습하려면 어떤 구조적 변화가 필요한가?"**

퍼셉트론 모델은 논리적 연산과 비선형 문제(XOR 등)를 풀 수 있다는 가능성을 보여주었지만, 가중치와 편향을 작업자가 수동으로 설정해야 한다는 치명적인 단점이 있었습니다. **신경망(Neural Network)**은 '활성화 함수(Activation Function)'를 매끄러운 형태로 변경하여 모델이 스스로 가중치를 학습할 수 있는 기반(미분 가능성)을 마련하고, 복잡한 데이터를 효율적으로 추론하는 구조로 진화했습니다.

## 2. Key Mechanism (해결 원리)

### 활성화 함수 (Activation Function)의 진화
- **계단 함수(Step Function)의 한계**: 임계치를 넘으면 1, 아니면 0을 반환하는 함수로, 불연속적이라 미분이 불가능하여 역전파(Backpropagation) 학습이 불가능합니다.
- **시그모이드(Sigmoid)와 렐루(ReLU)**: 신경망은 신호를 연속적인 실수 값으로 변환하는 은닉층 활성화 함수를 도입했습니다. 특히 현대 딥러닝에서는 기울기 소실(Gradient Vanishing) 문제를 방지하고 연산이 빠른 **ReLU (Rectified Linear Unit)**가 가장 기본적으로 사용됩니다.
- *참고:* 활성화 함수는 반드시 **비선형 함수(Non-linear Function)**여야 합니다. 선형 함수를 사용하면 아무리 층을 깊게 쌓아도 은닉층이 없는 단일 계층 네트워크와 수학적으로 동일해지기 때문입니다.

### 다차원 배열 연산 (순전파와 배치 처리)
- 3층 신경망의 복잡한 뉴런 연결은 NumPy의 행렬 내적 연산(`np.dot`)을 통해 $\mathbf{A} = \mathbf{X}\mathbf{W} + \mathbf{B}$ 형태로 매우 우아하고 효율적으로 처리됩니다.
- **배치(Batch) 처리**: 데이터(이미지)를 1장씩 넣지 않고 100장 단위로 묶어서(배치) 입력하면, 큰 행렬의 곱셈 연산으로 변환되어 컴퓨터의 수치 해석 라이브러리와 캐시 메모리를 효율적으로 활용해 연산 속도가 극적으로 향상됩니다.

### 출력층 설계 (Output Layer Design)
문제의 종류에 따라 출력층의 활성화 함수가 달라집니다.
- **회귀(Regression)**: 예측해야 하는 값이 연속적인 수치이므로 **항등 함수(Identity Function)**를 사용하여 피드포워드된 값을 그대로 출력합니다.
- **분류(Classification)**: 특정 카테고리에 속할 확률을 구해야 하므로 **소프트맥스 함수(Softmax Function)**를 사용합니다. 출력값은 0과 1 사이의 실수가 되며, 전체 출력의 합이 1이 되어 '확률 분포(Probability Distribution)'로 해석할 수 있습니다.

## 3. Mathematical Foundation (수학적 기반)

### 주요 활성화 함수 (Activation Functions)
- **시그모이드 함수 (Sigmoid)**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **렐루 함수 (ReLU)**: $h(x) = \begin{cases} x & (x > 0) \\ 0 & (x \le 0) \end{cases} = \max(0, x)$

### 소프트맥스 함수와 오버플로(Overflow) 방지
표준 소프트맥스 수식은 지수 함수($e^x$)를 사용하므로, 입력값이 조금만 커져도 결괏값이 무한대(inf)로 발산하여 `NaN` 오류를 일으킵니다(오버플로 현상).
이를 방지하기 위해 입력 신호 중 **최댓값(C)**을 빼주어 연산적 안정성을 달성합니다.

$$
y_k = \frac{e^{a_k}}{\sum_{i=1}^{n} e^{a_i}} = \frac{e^{a_k - C}}{\sum_{i=1}^{n} e^{a_i - C}}
$$

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a) # 오버플로 방지
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)

def init_network():
    return {
        'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b1': np.array([0.1, 0.2, 0.3]),
        'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        'b2': np.array([0.1, 0.2]),
        'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3': np.array([0.1, 0.2])
    }

def forward(network, x):
    # Layer 1
    a1 = np.dot(x, network['W1']) + network['b1']
    z1 = sigmoid(a1)
    # Layer 2
    a2 = np.dot(z1, network['W2']) + network['b2']
    z2 = sigmoid(a2)
    # Layer 3 (Output)
    a3 = np.dot(z2, network['W3']) + network['b3']
    y = softmax(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5]) # 임의의 입력 데이터
y = forward(network, x)

print(f"입력: {x}")
print(f"소프트맥스 출력(확률분포): {y}")
print(f"총합: {np.sum(y)}")
```

## 🚀 추가 학습 및 실습 제안

1. **[수학적 고찰] 비선형 활성화 함수의 필요성 증명**
   - $h(x) = cx$ 라는 선형 활성화 함수를 사용하는 3층 신경망 $y(x) = h(h(h(x)))$ 가 있다고 가정해 봅니다. 이 수식을 전개하여 결국 $y(x) = c^3x$ 라는 단일 선형 연산($a=c^3$)과 본질적으로 차이가 없음을 직접 손으로 증명해 보세요.
2. **[코딩 실습] Softmax 오버플로 재현 실험**
   - 크기가 큰 배열 `np.array([1010, 1000, 990])`을 만들어 원래의 소프트맥스 함수(최댓값을 빼지 않은 버전)에 입력해 보고 어떤 오류(NaN)가 나오는지 확인합니다. 그런 다음 상수 $C$(입력의 최댓값)를 이용한 개선된 버전에 넣어보고 정상적으로 확률 비율이 도출되는 원리를 코드로 체감해 봅니다.
3. **[논문 리뷰 제안] Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (2012)**
   - 딥러닝의 부흥을 이끈 전설적인 'AlexNet' 논문입니다. 이 눈문에서 어떻게 Sigmoid나 Tanh 대신 **ReLU(Rectified Linear Units)**를 도입함으로써 딥러닝의 수렴 속도를 획기적으로 가속시켰는지 논문의 3.1절 (ReLU Nonlinearity)을 읽어보시길 권합니다.
