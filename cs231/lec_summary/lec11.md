# Lecture 11: Large Scale Distributed Training (대규모 분산 훈련)

본 문서는 사용자의 강의 메모를 분석하여 오탈자 및 번역이 필요한 부분을 교정하고, 최신 LLM 시스템 훈련의 핵심 개념을 `Core Question -> Key Mechanism -> Mathematical Foundation` 형식으로 구조화하여 대폭 확장(Expanded)한 문서입니다.

과거에는 모든 딥러닝 모델 훈련이 단일 GPU에서 이루어졌지만, Llama 2 (405B)와 같은 초거대 모델의 등장으로 단일 GPU 훈련은 물리적으로 불가능해졌습니다. 이제 시스템 인프라 측면의 분산 처리와 병렬화 최적화는 새로운 '표준' 훈련 방식입니다.

---

## 1. 하드웨어적 배경 (GPU Cluster & Memory Traffic)
거대한 딥러닝 모델의 연산 병목은 대부분 '연산 속도' 자체보다 '데이터 이동(메모리 트래픽)'에서 발생합니다.

- **GPU (H100 등)와 TPU**: 그래픽 처리를 위해 고안된 병렬 아키텍처가 행렬곱 연산에 최적화되어 도입되었습니다. 내부에 전통적인 부동소수점(FP) 코어와 강력한 행렬 연산 전용 **텐서 코어(Tensor Core)**가 혼재되어 있습니다. 
- **혼합 정밀도(Mixed Precision)**: 텐서 코어는 주로 16비트 연산을 고속으로 처리하므로, PyTorch 등에서 올바른 데이터 타입(dtype)을 지정하지 않으면 텐서 코어가 활성화되지 않아 연산 속도가 극도로 느려질 수 있습니다.
- **클러스터 구조**: 단일 서버에 보통 8개의 GPU가 배치되고, 랙(Rack) 및 포드(Pod) 구조로 수천 개의 GPU 클러스터를 형성합니다. 이 클러스터 계층 구조에 맞게 통신량을 줄이는 것이 모델 FLOPs 활용도(MFU) 상승의 핵심입니다.

## 2. 5가지 차원의 병렬화 기술 (Parallelism Dimensions)
수천 개의 GPU를 효율적으로 쓰기 위해 Transformer 모델이 갖는 축(데이터 배치, 모델 깊이, 가중치 크기, 시퀀스 길이)을 어떻게 쪼갤지 결정합니다.

### 2.1 데이터 병렬 처리 (Data Parallelism, DP & DDP)
- **Core Question**: 배치 사이즈가 너무 커서 한 번에 훈련할 수 없거나, 단순히 학습 속도를 M배 올리고 싶다면?
- **Key Mechanism**:
  - 훈련 데이터셋 모음(Global Batch)을 개별 GPU들이 각각 처리할 로컬 미니배치(Local Batches)로 쪼개어 나누어 줍니다.
  - **DDP (Distributed Data Parallel) 동작 과정**:
    1. 모든 GPU는 각자 '동일한 모델 가중치'와 '옵티마이저'의 완벽한 복사본을 메모리에 로드합니다.
    2. 각 GPU가 독립적으로 자신에게 할당받은 미니배치 데이터에 대해 순전파(Forward pass)를 실행해 개별 Loss를 구합니다.
    3. 각 GPU 내부에서 역전파(Backward pass)를 수행해 개별적인 기울기(Gradients)를 계산합니다.
    4. **All-Reduce 연산**: 각 GPU가 본 데이터가 다르므로 계산된 기울기도 각기 다릅니다. 네트워크를 통해 파라미터 간 동기화(통신)를 수행하여 기울기의 '평균'을 구합니다.
    5. 산출된 평균 기울기를 사용해 모든 복제된 파라미터가 정확히 똑같은 수치로 Weight를 갱신합니다.
- **Mathematical Foundation**:
  - $\nabla L_{total} = \frac{1}{M} \sum_{i=1}^{M} \nabla L_i$ ($M$: GPU 개수)

### 2.2 가중치 분산 데이터 병렬 처리 (FSDP & HSDP)
- **Core Question**: 모델 파라미터 수 자체가 너무 매개변수가 많아 단일 GPU에 '복사본' 하나조차 올릴 수 없다면?
- **Key Mechanism**:
  - **FSDP (Fully Sharded Data Parallel)**: 전체 모델의 가중치, 그래디언트, 옵티마이저 파라미터를 조각(Shard)내어 여러 GPU에 흩뿌려 보관합니다. 순전파나 역전파 과정에서 특정 레이어의 연산이 필요한 순간에만 잠시 통신을 통해 해당 조각들을 모으고(Gather), 연산이 끝나면 즉시 삭제하여 메모리를 극도로 절약합니다.
  - **HSDP (Hybrid Sharded Data Parallel)**: 서버 간 통신 부하를 줄이기 위해, 가까운 그룹(노드 한 대) 내에서는 FSDP처럼 쪼개고, 노드와 노드 사이에는 DDP처럼 덩어리째 복제하는 절충안입니다.

### 2.3 문맥/시퀀스 병렬 처리 (Context Parallelism, CP)
- **Core Question**: 초장문 텍스트(예: 1M 토큰의 책 한 권)처럼 시퀀스(Sequence) 길이 $L$이 너무 커서 병목이 발생하면?
- **Key Mechanism**:
  - 모델 레이어 중 Normalization, Residual Connection이나 일반 MLP, QKV Projection 레이어는 시퀀스를 조각내어 잘라도 각 프레임이 독립이므로 분할 병렬 연산 후 통신하는 것이 매우 쉽습니다.
  - **Attention 연산 병렬화의 어려움**: 어텐션 행렬은 쿼리(Query)가 모든 키-밸류(Key-Value)를 전부 봐야 하므로 쪼개기 어렵습니다.
    1. **Ring Attention**: 시퀀스를 특정 블록으로 나누고 GPU 간 링(Ring) 포맷으로 통신하며 Key/Value 연산을 교차로 나눕니다. 몹시 복잡하나 무한 확장이 가능합니다.
    2. **Ulysses**: 어텐션 내의 멀티-헤드(Multi-head)를 기준으로 GPU를 할당합니다. 구현이 훨씬 심플하나, 최대 활용 가능한 GPU 수가 헤드(Heads) 수에 국한된다는 한계가 있습니다.

### 2.4 파이프라인 병렬 처리 (Pipeline Parallelism, PP)
- **Core Question**: Transformer 층이 100층, 200층 등 너무 깊다면(Layer-wise)?
- **Key Mechanism**:
  - 모델의 앞쪽 레이어 군은 GPU A, 뒷단 레이어 군은 GPU B에 순차적으로 적재합니다. 경계선에서만 Activation 정보 등을 넘겨줍니다.
  - **유휴 시간(Bubble) 발생**: 직렬적인 처리 탓에 앞쪽 연산이 진행되는 동안 뒤쪽 GPU는 놀고 있습니다 (N대의 PP시 최대 효용 MFU는 $1/N$).
  - **해결책 (Micro-batches)**: 긴 대기를 막기 위해 로컬 미니배치를 더 잘게 나눈 마이크로배치 형태로 파이프라인 사이사이에 교차 투입(Interleaving)하여 GPU 유휴 시간을 최소화합니다.

### 2.5 텐서 병렬 처리 (Tensor Parallelism, TP)
- **Core Question**: 완전 연결 계층 같이 레이어 안의 거대한 단일 텐서(선형 가중치 행렬) 마저도 여러 개의 GPU로 나누어 계산해야 할 때는?
- **Key Mechanism**:
  - 행렬 $Y = X \cdot W$에서 가중치 행렬 $W$를 열(Column)이나 행(Row)으로 쪼개(Block Matrix Multiply) 각각 연산한 후 출력 $Y$를 즉시 Gather(통신 결합)합니다. 빈번하고 매우 높은 네트워크 대역폭(예: NVLink)이 필수입니다.

---

## 3. 실전 스케일링 레시피 (Scaling Recipe)
- **핵심 목표**: 시스템 연산의 효율 등급표인 **MFU (Model FLOPs Utilization)**를 대규모 환경에서도 최대화시키는 것입니다.

  1. ~1B 파라미터 / 128 GPU 이하: 단순하고 강력한 **데이터 병렬(DP/DDP)** 부터 적용합니다.
  2. 항상 GPU당 배치 사이즈는 메모리가 뻗기 전까지 **풀로 채웁니다(Max out capacity)**. 배치 크기가 클수록 메모리 트래픽 효율이 높아집니다.
  3. 활성화 메모리 절약을 위해 역전파 시퀀스에서 중간 결과를 날리는 **활성화 체크포인팅(Activation Checkpointing)**을 반드시 도입해 배치 크기를 더 확보합니다.
  4. 파라미터가 1B를 넘으면 통신 오버헤드를 줄이면서 메모리를 아끼는 **FSDP**를 우선 도입합니다 (HSDP는 256 GPU 이상의 큰 클러스터에서 고려).
  5. 규모가 1,000대 이상 클러스터이거나 파라미터 50B 이상, 시퀀스 16K를 넘어설 때 비로소 **PP, CP, TP** 등의 고급 쪼개기 병렬 전략들을 타협적으로 조합하여 사용합니다.

---
## 🚀 추가 학습 및 실습 제안 (Further Explorations)

분산 훈련은 단순히 AI 모델링을 넘어서 컴퓨터 시스템 아키텍처에 대한 이해도를 요구합니다.

1. **[구현 실습 제안] PyTorch `DDP` 장난감 모델 구현**:
   - 코랩 환경 등에서 PyTorch의 `torch.nn.parallel.DistributedDataParallel` 모듈을 불러와 심플한 MLP 모델이 각기 다른 로컬 배치에서 어떻게 그래디언트를 All-Reduce하여 평균을 내는지 실험 코드를 (가짜 GPU나 CPU 노드 환경에서) 짜보길 권장합니다.
2. **[메모리 수학 계산] Activation Checkpointing의 효용**:
   - 논문을 참조하여 일반적인 Transformer 레이어 1개를 지날 때 저장되는 활성화 값(Activations)의 메모리 크기 수식을 세워보고, 체크포인팅 도입 시 VRAM이 얼마나 확보되는지 역전파 연산량 증가 횟수와 대비하여 비교 계산 해보세요.
3. **[논문 리딩] 분산 학습 핵심 논문**:
   - *Megatron-LM (Shoeybi et al., 2019)*: 텐서 병렬화(TP)를 업계 표준으로 정착시킨 NVIDIA 논문입니다.
   - *PyTorch FSDP 논문 및 페이스북 AI 블로그 매뉴얼*: 대규모 모델이 VRAM 병목을 다루는 방법에 대해 상세한 엔지니어링 팁을 배울 수 있습니다.
