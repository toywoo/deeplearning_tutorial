# [Expanded] 신경망 학습 (Training Neural Network)

이 문서는 `1_4_train_neu_net.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"기계가 학습하기 위해 현재 상태를 평가해야 한다면, 왜 직관적인 '정확도(Accuracy)' 대신 수식적으로 복잡한 '손실 함수(Loss Function)'를 기준으로 삼는가?"**

사람은 문제를 풀 때 몇 개를 맞췄는지(정확도)로 실력을 평가하지만, 신경망은 '손실 함수(오차)'의 값을 최소화하는 방식을 택합니다. 이는 기계학습의 핵심인 **미분(기울기)**을 구하기 위함입니다. 정확도를 지표로 삼으면 매개변수(가중치)를 아주 미세하게 조정하더라도 정확도는 변하지 않거나 불연속적(예: 32% -> 33%)으로 널뛰기 때문에 미분값이 대부분 0이 되어버립니다. 반면 손실 함수는 매개변수의 미세한 변화에 연속적인 실수 값으로 민감하게 반응하므로, 미분을 통해 어느 방향으로 가중치를 갱신해야 할지 수학적 길잡이를 제공할 수 있습니다.

## 2. Key Mechanism (해결 원리)

### 미니배치(Mini-batch) 기반의 데이터 주도 학습
수만 장의 훈련 데이터를 모두 보며 손실을 계산하는 것은 연산량이 너무 많습니다. 따라서 전체 데이터 중 일부(예: 100장)를 무작위로 추려내어 학습하는 **미니배치(Mini-batch)** 방식을 사용합니다. 
이를 통해 데이터에 편향되지 않는 범용적인 특징을 빠르게 추출하고 오버피팅(Overfitting)을 방지합니다.

### 수치 미분과 기울기 (Gradient)
컴퓨터는 극한의 개념인 해석적 미분($\lim_{h \to 0}$)을 완벽하게 구현할 수 없으므로, 아주 작은 차이 $h$ (예: $10^{-4}$)를 이용해 두 점 사이의 기울기를 구하는 **수치 미분(Numerical Differentiation)**을 수행합니다. 
> 💡 **코드 연계**: 오차를 줄이기 위해 전방 차분이 아닌 중앙 차분 `(f(x+h) - f(x-h)) / (2*h)`을 이용한 구현체는 [1_4_train_neu_net.ipynb:L180-L182](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_4_train_neu_net.ipynb#L180-L182)에서 확인할 수 있습니다.

````

이러한 미분 연산을 모든 변수(가중치 행렬의 모든 원소)에 대해 묶어서 벡터화한 것을 **기울기(Gradient)**라고 합니다. 기울기는 함수의 출력값을 가장 크게 줄이는 방향을 가리킵니다.
> 💡 **코드 연계**: 모든 원소에 대한 부분 편미분 과정인 `numerical_gradient` 함수는 [1_4_train_neu_net.ipynb:L262-L280](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_4_train_neu_net.ipynb#L262-L280)에 구현되어 있습니다.

````

### 경사 하강법 (Gradient Descent)
산 정상에서 가장 가파른 내리막길로 한 걸음씩 내려오듯, 현 위치에서 기울기 방향으로 일정 반경(학습률)만큼 이동하며 매개변수를 반복적으로 갱신하는 최적화 기법입니다.
> 💡 **코드 연계**: 미니배치 추출부터 기울기 산출, SGD에 기반한 매개변수 매 스텝 갱신(`learning_rate * grad[key]`)의 완전한 학습 루프는 [1_4_train_neu_net.ipynb:L528-L545](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_4_train_neu_net.ipynb#L528-L545)에서 확인 가능합니다.

````
## 3. Mathematical Foundation (수학적 기반)

### 교차 엔트로피 오차 (Cross-Entropy Error)
다중 클래스 분류 신경망에서 가장 흔히 쓰이는 손실 함수입니다. 정답일 때의 예측 확률 값이 커질수록 오차는 0에 수렴하며, 예측이 틀릴수록 오차 수치가 급격히 발산하는 로그($\log$) 함수의 성질을 띱니다.

$$E = -\frac{1}{N} \sum_{n} \sum_{k} t_{nk} \log(y_{nk} + 10^{-7})$$
- $t_{nk}$: 정답 레이블 (원-핫 인코딩 구조)
- $y_{nk}$: 신경망의 예측 결과 (소프트맥스 활성화 값)
- $10^{-7}$: `np.log(0)` 발생 시 `-inf` 가 뜨는 것을 막기 위한 매우 작은 스무딩 상수
> 💡 **코드 연계**: 위 수학적 엣지 케이스를 막는 미세 조정이 도입된 함수 구현은 [1_4_train_neu_net.ipynb:L83-L88](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_4_train_neu_net.ipynb#L83-L88)을 참고하세요.

````

### 경사 하강법 파라미터 업데이트 수식
$$ \mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}} $$
$\eta$는 학습률(Learning Rate)이라는 하이퍼파라미터로, 한 번에 얼마나 기울기를 따라 이동할지를 결정합니다.

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

# 임의의 아주 간단한 손실 함수 테스트용 (f(W) = W_1^2 + W_2^2)
W = np.array([3.0, 4.0])
def loss_function(w):
    return np.sum(w**2)

print("시작 전 가중치:", W)
lr = 0.1 # 학습률 (Learning Rate)

for step in range(3):
    grad = numerical_gradient(loss_function, W)
    W -= lr * grad # 경사 하강법 단계
    print(f"Step {step+1}: 기울기 {grad}, 업데이트된 가중치 = {W}")
```

## 🚀 추가 학습 및 실습 제안

1. **[코딩 실습] 학습률(Learning Rate) 튜닝 실험**
   - 현재 노트북 하단 훈련 코드의 `learning_rate` 값을 0.1에서 `10.0` (너무 큼) 혹은 `0.0001` (너무 작음)로 극단적으로 변경해 보세요. 학습 경과 그래프가 어떻게 망가지는지 직접 관찰해 하이퍼파라미터 튜닝의 중요성을 깨달아 보세요.
2. **[수학적 고찰] 국소 최적해(Local Minimum) 탈출의 어려움**
   - 수치 미분과 경사 하강법은 "가장 가파른 아래" 방향만을 지시하므로, 산맥에서 작은 구덩이(Local Minimum)나 평형 지대(Saddle Point)에 빠질 경우 더 이상 학습이 되지 않습니다. 이 수학적 한계를 어떻게 극복할 수 있을지 아이디어를 그려보세요.
3. **[성능 분석] 수치 미분의 한계점 체감**
   - 소스코드 맨 마지막 셀에서 학습 1 에폭(Epoch)도 엄청나게 오래 걸리거나 IndexError로 멈추는 이유를 고민해 보세요. 실제 딥러닝 프레임워크는 이런 느린 '수치 미분' 대신 빠르고 정확한 **오차역전파(Backpropagation)**를 사용합니다. 이 흐름을 파악하며 다음 5장을 준비해 보세요.
