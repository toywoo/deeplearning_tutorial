# [Expanded] 학습 관련 기술 (Optimization Techniques)

이 문서는 `1_6_optimization.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"딥러닝 모델이 단순히 목적지를 향해 터벅터벅 걷는 것(SGD)을 넘어, 최단 시간 내에 협곡을 빠져나와 최적해에 도달하고 실전(Test)에서도 좋은 점수를 받게 할 방법은 무엇인가?"**

순수한 '확률적 경사 하강법(SGD)'은 비등방성(Anisotropic, 방향마다 기울기 스케일이 확연히 다른) 함수 지형에서 지그재그로 비효율적으로 요동치는 한계가 있습니다. 또한, 초기 가중치를 잘못 설정하면 학슴이 시작되기도 전에 소멸해 버리고, 훈련 데이터에 너무 집중하면 '과적합(Overfitting)'에 빠져버립니다. 본 챕터에서는 학습 속도를 폭발적으로 높이는 **옵티마이저(Optimizer)**, 기울기 흐름을 살리는 **가중치 초기화(Weight Initialization) 및 배치 정규화(Batch Normalization)**, 오버피팅을 억제하는 **정규화(Regularization)** 등 '더 깊은' 신경망 훈련을 위한 최신 노하우를 집대성합니다.

## 2. Key Mechanism (해결 원리)

### 비등방성 지형을 탈출하는 옵티마이저
- **모멘텀 (Momentum)**: 물리법칙의 '관성'을 모방하여, 이전 기울기 방향으로의 속도를 유지시켜 지그재그 움직임을 감쇠합니다.
- **Adam (Adaptive Moment Estimation)**: 관성(Momentum)과 각 매개변수별 적응적 보폭 조절(AdaGrad)을 융합한, 현대 딥러닝에서 가장 사랑받는 강력한 최적화 기법입니다.
> 💡 **코드 연계**: Adam 옵티마이저가 1차/2차 모멘트를 지수 이동 평균으로 추적하며 보폭($v$)과 방향($m$)을 업데이트하는 핵심 구현 코드는 [1_6_optimization.ipynb:L149-L174](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_6_optimization.ipynb#L149-L174)에서 확인하세요!

````

### 활성화값의 분포와 가중치 초기화
가중치를 0으로 초기화하면 모든 뉴런이 대칭적으로 업데이트되므로 학습이 붕괴됩니다. 초기 활성화값들이 0이나 1로 치우치면 미분값이 사라지는 **기울기 소실(Gradient Vanishing)**이 일어납니다.
- **Xavier (글로럿) 초깃값**: 활성화 함수가 선형적(비대칭성)일 때, 앞 층 노드 수의 분산($1/\sqrt{n}$)에 맞추어 분포시킵니다.
- **He (카이밍) 초깃값**: 음수 영역을 0으로 꺼버리는 ReLU의 성질을 고려해, 분산을 두 배($\sqrt{2/n}$) 더 넓혀 분포의 밸런스를 바로잡습니다.

### 과적합 억제: 가중치 감소와 드롭아웃
모델이 지나치게 데이터를 외우지 못하게 하려면 '페널티'가 필요합니다.
- **L2 가중치 감소 (Weight Decay)**: 가중치가 비정상적으로 커지는 것을 막기 위해 오차에 패널티($L2$ 규제)를 부여합니다.
- **드롭아웃 (Dropout)**: 훈련 시 뉴런들을 임의 비율로 무작위 삭제하여, 특정 뉴런에 의존하지 않는 앙상블(Ensemble) 효과를 만들어냅니다.
> 💡 **코드 연계**: 훈련 플래그(`train_flag`)에 따라 임의의 `mask`를 씌워 뉴런을 삭제하는 기발한 드롭아웃 로직은 [1_6_optimization.ipynb:L220-L235](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_6_optimization.ipynb#L220-L235)에서 확인할 수 있습니다.

````

## 3. Mathematical Foundation (수학적 기반)

### Adam 옵티마이저와 AdamW 변형
Adam 옵티마이저의 가중치 업데이트 식은 아래와 같습니다. (최적 편향 보정 단계 포함)
$$ W_{t+1} = W_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

하지만 기존 코어 수식 내에 'L2 가중치 감소(Weight Decay)' 항을 섞어버리면, 어댑티브 러닝레이트 분모($v_t$)에 의해 가중치 감쇠율마저 스케일이 제각각 변형됩니다. 이를 해결하여 가중치 감쇠를 **가장 마지막 업데이트 단계에 별도로 빼버린** 수학적 교정 기법이 바로 **AdamW**입니다.

### L2 가중치 감소 수식
원래의 데이터 손실함수 $L_{data}$에 매개변수 가중치 텐서 $\mathbf{W}$의 L2-Norm(제곱합) 절반을 더해 새로운 총 손실 $L_{total}$을 정의합니다.
$$ L_{total} = L_{data} + \frac{\lambda}{2} \|\mathbf{W}\|^2 $$
미분하면 기존 역전파 기울기에 $\lambda \mathbf{W}$만 직관적으로 더해지게 됩니다.

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None # 관성(속도) 초기화
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            # 물리 법칙에 따른 속도 누적
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# He 가중치 초깃값 (ReLU 활성화 함수에 최적화)
def init_he_weights(node_num_in, node_num_out):
    # 표준편차가 sqrt(2 / 입력노드수) 인 정규분포 유지
    std = np.sqrt(2.0 / node_num_in)
    return np.random.randn(node_num_in, node_num_out) * std

# Dummy data test
params = {'W1': init_he_weights(100, 50)}
grads = {'W1': np.random.randn(100, 50)}
optimizer = Momentum()

print(f"초기 가중치 평균: {np.mean(params['W1']):.4f}, 분산: {np.var(params['W1']):.4f}")
optimizer.update(params, grads)
print(f"적용 완료 후 가중치 평균: {np.mean(params['W1']):.4f}")
```

## 🚀 추가 학습 및 실습 제안

1. **[코딩 실습] 옵티마이저 데스매치 (Optimizer Race)**
   - 똑같이 구성된 5층 신경망에 각각 SGD, Momentum, AdaGrad, Adam 옵티마이저를 물린 후 손실함수 감소 그래프 커브가 어떻게 변하는지 한 화면에 Plot 하여 직관적으로 관찰해 보세요.
2. **[수학적 고찰] 배치 정규화(Batch Normalization)의 마법**
   - 계층으로 들어가기 직전, 미니배치 데이터 집합 $\mathbf{X}$의 평균을 0, 분산을 1로 맞춘 뒤 척도(Scale $\gamma$)와 이동(Shift $\beta$) 파라미터로 데이터를 자유자재 변형하는 배치 정규화 알고리즘의 역전파 수식을 손으로 직접 유도해 봅시다. 놀랍도록 복잡하지만 오버피팅을 강력히 억제합니다.
3. **[논문 리뷰 제안] "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014)**
   - 드롭아웃 개념을 세상에 널리 알리고 대중화한 근본 논문입니다. 단순히 랜덤하게 노드를 끄는 행동이 왜 수많은 다른 형태의 신경망들을 평가하고 평균 내는 '앙상블(Ensemble)' 학습과 수학적으로 동치인지 증명을 읽어보시기를 강력 추천합니다.
