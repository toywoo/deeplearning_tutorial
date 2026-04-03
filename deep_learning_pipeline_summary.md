# 🚀 딥러닝 핵심 개념 총정리: 학습 파이프라인 (Deep Learning Pipeline)

딥러닝 모델이 학습하는 전체 과정을 파이프라인 순서대로 정리했습니다. CS231n 강의 내용과 밑바닥부터 시작하는 딥러닝(DL_scratch)의 구현 원리를 모두 통합하여, 각 단계마다 구체적인 기법이 왜 필요한지 수학적/직관적 핵심 원리와 함께 정리했습니다.

---

## 1. 🛠️ 데이터 준비, 전처리 및 미니배치 (Data Prep & Mini-Batching)
전체 데이터를 한 번에 학습하는 것(Batch)은 메모리와 계산 비용 측면에서 비효율적이므로 미니배치를 사용합니다. 또한, 원본 이미지는 조명, 가려짐, 형태 변화 등에 취약하므로 전처리가 필수적입니다.

*   **미니배치 (Mini-Batch):** 확률적(Stochastic) 노이즈를 통해 지역 최소점(Local Minima)을 탈출하는 효과가 있습니다.
*   **데이터 전처리 (Preprocessing):** 각 채널별로 평균을 빼고 표준편차로 나누어주는 정규화를 주로 사용합니다. (Zero-centered & Normalized data)
*   **데이터 증강 (Data Augmentation):** 수평 뒤집기(Horizontal flip), 무작위 자르기(Crop), 색상 변형(Color jitter) 등을 통해 훈련 데이터에 인위적인 변화를 주어 모델의 일반화 성능을 극대화합니다.
*   **💡 더 학습할 거리:** k-NN에서 L1/L2 거리의 기하학적 의미 차이 (L1은 노이즈에 강하고 개별 특징 보존, L2는 회전에 불변).

---

## 2. 🏗️ 모델 아키텍처 및 선형 분류기 (Model Architecture)
입력 이미지를 템플릿과 매칭하여 클래스별 점수를 계산합니다. 

*   **선형 분류기 (Linear Classifier):** $f(x, W) = Wx + b$. 가중치 $W$는 각 클래스의 시각적 '템플릿' 역할을 합니다.
*   **CNN (Convolutional Neural Networks):** 수동으로 특징(Feature)을 추출하던 기존 방식(HoG 등)을 대체하여, 필터 가중치 학습을 통해 특징을 추출합니다.
    *   **VGGNet:** 큰 필터 대신 3x3 필터를 여러 층 쌓아 매개변수는 줄이고 비선형성은 높이는 구조를 채택했습니다.
    *   **ResNet:** 레이어가 깊어질수록 학습이 어려워지는 문제를 해결하기 위해, 입력을 출력에 바로 더하는 잔여 학습(Residual Learning, $F(x) + x$)을 도입했습니다.

---

## 3. ⚖️ 가중치 초기화 (Weight Initialization)
가중치를 모두 0으로 초기화하면 모든 뉴런이 똑같이 학습되는 문제(Symmetry Breaking 실패)가 발생합니다. 초기 가중치 $W$에 따라 기울기 소실(Vanishing Gradient)이나 폭발이 결정됩니다.

*   **Xavier 초기화:** Sigmoid, Tanh에 적합. 층의 노드 수 $n$에 대해 $\sqrt{\frac{1}{n}}$의 표준편차를 사용합니다.
*   **He 초기화 (Kaiming Init):** ReLU 계열에 적합. ReLU가 음수 영역을 0으로 만들기 때문에 분산이 반토막 나는 것을 보정하기 위해 표준편차 $\sqrt{\frac{2}{n}}$를 사용합니다.
*   **💡 더 학습할 거리:** 최근 트렌드인 Orthogonal Initialization 및 모델 구조(Transformer 등)에 따른 초기화.

---

## 4. ➡️ 순전파 및 활성화 함수 (Forward Propagation & Activation)
단순한 선형 연산의 반복을 방지하고 표현력을 높이기 위해 합성곱/선형 레이어 사이에 비선형 활성화 함수를 추가합니다.

*   **Sigmoid의 한계:** 양 끝단에서 미분값이 0에 가까워져 층이 깊어지면 기울기 소실(Vanishing Gradient)이 발생합니다.
*   **ReLU (Rectified Linear Unit):** 양수 영역에서 기울기가 1이므로 기울기 소실 문제를 해결하며 연산이 매우 빠릅니다. 하지만 음수 영역에서 기울기가 0이 되어 뉴런이 죽는 'Dying ReLU' 문제가 있습니다.
*   **GELU (Gaussian Error Linear Unit):** 0 부근에서 부드러운 곡선(Smooth)을 가져 죽은 뉴런 문제를 완화한 트렌디한 활성화 함수입니다.

---

## 5. 🛡️ 배치 정규화 (Batch Normalization)
각 층의 활성화 값 분포가 학습 도중 계속 변하는 현상(Internal Covariate Shift)을 막아줍니다.

*   **필요성:** 학습을 매우 안정적으로 만들고, 가중치 초기화에 대한 의존도를 낮추며, 높은 학습률(Learning Rate)을 허용하여 학습 속도를 대폭 향상시킵니다.
*   **원리:** 미니배치 단위로 평균 0, 분산 1로 정규화한 뒤, 학습 가능한 파라미터 $\gamma$(Scale)와 $\beta$(Shift)를 적용합니다.

---

## 6. 🎯 손실 함수 (Loss Function)
모델의 예측값과 실제 정답 간의 오차를 수치화하여 '불만족도'를 측정합니다.

*   **Softmax 분류기 (Cross-Entropy Loss):** 원본 점수(Logit)를 $exp()$를 통해 양수로 만들고 정규화하여 확률값으로 변환합니다. 정답 확률의 로그값에 마이너스를 붙여 손실을 계산하며, 오답에 대한 페널티가 기하급수적으로 커집니다.
*   **SVM Loss (Hinge Loss):** 정답 클래스 점수가 오답 클래스 점수보다 일정 마진(Margin) 이상 크도록 유도합니다.
*   **💡 더 학습할 거리:** Focal Loss, Label Smoothing.

---

## 7. ⬅️ 오차 역전파 (Backward Propagation)
손실 함수에서 구한 오차를 출력층에서부터 역방향으로 전달하여 각 가중치의 기울기(Gradient)를 계산합니다.

*   **수치 미분 vs 해석적 미분:** 수치 미분($\frac{f(x+h) - f(x)}{h}$)은 느리고 오차가 있습니다. 따라서 연쇄 법칙(Chain Rule)을 이용한 해석적 미분을 통해 고속으로 정확한 기울기를 구합니다.
*   **원리:** 국소적 미분(Local Gradient)과 상류에서 흘러들어온 미분(Upstream Gradient)을 곱해 하류로 전달(Downstream Gradient)합니다.

---

## 8. 🏃 최적화 (Optimization)
역전파로 구한 기울기를 바탕으로 가중치를 업데이트합니다. 지그재그 현상을 줄이고 최적점에 빠르게 도달하는 것이 목표입니다.

*   **SGD:** 현재 위치에서 기울기 방향으로 무작정 이동합니다.
*   **Momentum:** 과거의 이동 방향(관성)을 기억($m_t$)하여 지그재그 현상을 줄입니다 (산에서 공이 굴러가는 직관).
*   **AdaGrad:** 자주 업데이트되는 가중치는 보폭(학습률)을 줄이고, 적게 업데이트된 가중치는 보폭을 늘립니다.
*   **Adam (Adaptive Moment Estimation):** Momentum(방향)과 AdaGrad(보폭)의 장점을 결합하고, 초기 학습 쏠림을 막기 위한 편향 보정(Bias Correction)을 추가했습니다.
*   **AdamW:** Adam에서 Weight Decay(가중치 감소)를 손실 함수가 아닌 가중치 업데이트 식에 직접(Decoupled) 적용하여 모델의 일반화 성능을 크게 개선했습니다.

---

## 9. 🔒 규제화 및 과적합 방지 (Regularization)
모델이 훈련 데이터에 너무 맞춰져(Overfitting) 테스트 성능이 떨어지는 것을 방지합니다.

*   **Weight Decay (L2 정규화):** 큰 가중치에 페널티를 주어 특정 가중치에 과도하게 의존하는 것을 막습니다. $L_{total} = L_{data} + \frac{\lambda}{2} \|W\|^2$.
*   **Dropout (드롭아웃):** 훈련 시 은닉층의 뉴런을 무작위 확률로 0으로 끕니다. 특정 뉴런에 대한 의존도를 낮추고 앙상블 효과를 냅니다. **주의:** 테스트 시에는 모든 뉴런을 켜야 하므로 출력값의 스케일을 조정(Scaling)해야 합니다.

---

## 10. 🎛️ 하이퍼파라미터 튜닝 (Hyperparameter Tuning)
가중치 외에 모델이 직접 학습하지 못하고 사람이 설정해야 하는 값들(학습률, 미니배치 크기, 규제 강도 등)을 최적화합니다.

*   **검증 세트 (Validation Set):** 훈련 데이터와 테스트 데이터 외에, 하이퍼파라미터 튜닝만을 위한 검증 데이터를 분리하거나 교차 검증(Cross Validation)을 사용합니다.
*   **탐색 기법:** 그리드 탐색(Grid Search)보다 무작위 탐색(Random Search)이 효율적이며, 학습률 등은 10의 거듭제곱 단위(Log Scale)로 범위를 지정하여 탐색하는 것이 좋습니다.
