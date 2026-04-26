# [Expanded] 오차역전파법 (Backpropagation)

이 문서는 `1_5_backpropagation.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"매개변수가 수백만 개에 달하는 거대한 신경망에서, 느린 '수치 미분' 대신 기울기를 순식간에 계산해 내는 마법 같은 알고리즘은 무엇인가?"**

앞 장에서 배운 '수치 미분(Numerical Differentiation)'은 구현이 직관적이라는 장점이 있으나, 매개변수마다 일일이 $h$를 더하고 빼며 손실을 재계산해야 하므로 변수가 많아질수록 연산 시간이 기하급수적으로 늘어나는 치명적인 단점이 있습니다.
이를 극복하기 위해 등장한 **오차역전파법(Backpropagation)**은 복잡한 전체 수식을 쳐다보는 대신, **계산 그래프(Computational Graph)**상에서 각 노드의 '국소적 미분'만을 곱하며 거꾸로 전파하여, 단 한 번의 에러 피드백(역방향 스윕)만으로 모든 매개변수의 기울기를 효율적으로 도출해 냅니다.

## 2. Key Mechanism (해결 원리)

### 계산 그래프와 연쇄 법칙 (Computational Graph & Chain Rule)
계산 그래프는 복잡한 수식을 단순한 노드(Node, 예: 덧셈, 곱셈)와 에지(Edge)로 쪼개어 시각화합니다. 
하류(출력)에서 발생한 손실(에러)을 역방향으로 전달할 때, 합성함수의 미분 법칙인 **연쇄 법칙(Chain Rule)**을 통해 각 노드별 '국소적 미분'을 곱해주기만 하면 최초의 입력에 대한 최종 미분값을 쉽게 얻을 수 있습니다.
> 💡 **코드 연계**: 이런 국소적 미분 원리를 코드로 구현한 곱셈 노드 `MulLayer`의 `forward()`와 `backward()` 메서드는 [1_5_backpropagation.ipynb:L59-L76](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_5_backpropagation.ipynb#L59-L76)에서 확인할 수 있습니다.

````

### 계층(Layer)의 모듈화
복잡한 신경망을 구현하기 위해 ReLU, Affine(행렬 내적), Softmax-with-Loss 등의 각 처리 단계를 하나의 클래스(모듈)로 구현합니다. 
순전파(`forward`)와 역전파(`backward`) 규격을 통일해 두면, 여러 층의 신경망을 단순히 '레고 블록' 조립하듯 연결하여 순식간에 모델을 확장할 수 있습니다.
> 💡 **코드 연계**: 조립된 레이어들을 역순으로 호출하며 역전파를 수행하는 세련된 `gradient()` 메서드 구현은 [1_5_backpropagation.ipynb:L231-L248](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_5_backpropagation.ipynb#L231-L248)을 참고하세요.

````

### 소프트맥스-위드-로스 (Softmax-with-Loss) 계층의 우아함
소프트맥스 함수와 교차 엔트로피 오차를 합친 이 계층의 역전파 결과는 $y - t$ (즉, '신경망의 예측값 - 실제 정답')라는 매우 깔끔하고 직관적인 값으로 떨어집니다. 이 오차 그 자체가 역전파의 출발점이 되어 앞쪽 계층들에 에러를 잔소리하듯 전달합니다.

## 3. Mathematical Foundation (수학적 기반)

### 기본 노드의 역전파
- **덧셈 노드 ($z = x + y$)**: 상류에서 전해진 미분을 그대로 하류로 흘려보냅니다 ($\frac{\partial z}{\partial x} = 1$).
- **곱셈 노드 ($z = xy$)**: 상류의 값에 순전파 때의 입력 신호들을 '서로 바꾼' 값을 곱해서 보냅니다 ($\frac{\partial z}{\partial x} = y, \frac{\partial z}{\partial y} = x$).

### Affine 계층(행렬 내적)의 역전파
행렬 $\mathbf{X}$와 가중치 $\mathbf{W}$의 내적 $\mathbf{Y} = \mathbf{X} \cdot \mathbf{W} + \mathbf{B}$을 전개하는 Affine 계층의 역전파는 **전치 행렬(Transpose)**을 사용합니다. 행렬의 형상(Shape)을 맞춰주는 수학적 특성에 기인합니다.

$$ \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^T $$
$$ \frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}} $$

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y # 순전파 입력을 바꿈
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout * 1, dout * 1 # 그대로 흘려보냄

# 사과 쇼핑 로직: 가격 100원, 2개 구매, 소비세 1.1 적용
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(f"순전파 최종 계산된 가격: {price}")

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print("역전파 과정 국소적 미분 (Gradient) 결과:")
print(f"  d(가격) / d(사과가격) = {dapple}")
print(f"  d(가격) / d(사과개수) = {dapple_num}")
print(f"  d(가격) / d(소비세) = {dtax}")
```

## 🚀 추가 학습 및 실습 제안

1. **[코딩 실습] 기울기 확인 (Gradient Check)**
   - 오차역전파법 코드를 방금 다 구현하셨나요? [1_5_backpropagation.ipynb:L283-L288](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_5_backpropagation.ipynb#L283-L288)에 있는 '기울기 확인' 셀을 다시 실행시켜, 느린 '수치 미분' 결과와 빠른 '해석적 미분(오차역전파법)' 결과의 오차가 `1e-9` 수준으로 일치하는지 꼭 검증해 보세요. (버그를 잡는 핵심 로직입니다)

````
2. **[수학적 고찰] 시그모이드(Sigmoid) 역전파 유도**
   - $y = \frac{1}{1 + e^{-x}}$라는 복잡해 보이는 식이, 계산 그래프를 통해 국소적 미분으로 나누어 역전파하면 최종적으로 $\frac{\partial L}{\partial y} y(1 - y)$ 라는 매우 우아한 형태로 떨어집니다. 직접 종이에 계산 그래프 노드 $+$, $\exp$, $/$ 를 분리해서 그려보며 $y(1-y)$를 수학적으로 유도해 보세요.
3. **[논문 리뷰 제안] Rumelhart, Hinton, Williams, "Learning representations by back-propagating errors" (1986)**
   - 오늘날 딥러닝 성공의 근간이 된 1986년 네이처 논문입니다. 백프로파게이션(Backpropagation) 알고리즘이 은닉층의 표현(Representation)을 어떻게 기계가 스스로 학습할 수 있게 만들었는지 그 역사적 가치를 조명해 봅니다.
