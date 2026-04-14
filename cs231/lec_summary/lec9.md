# Lecture 9: Advanced Computer Vision Tasks & Model Visualization

본 문서는 사용자의 기존 강의 노트를 분석하여 오류를 수정하고, `Core Question -> Key Mechanism -> Mathematical Foundation` 구조에 맞춰 더 깊은 개념 설명과 함께 정리한 확장본(Expanded Version)입니다.

---

## 1. 최신 트렌드: Vision Transformer (ViT) 및 구조 변형
- **Core Question**: 자연어 처리(NLP)에서 압도적인 성능을 보인 Transformer 구조를 이미지 처리에 어떻게 최적화할 것인가?
- **Key Mechanism**:
  - **ViT (Vision Transformer)**: 이미지를 작은 패치(Patch) 단위로 분할하여 NLP의 단어 토큰처럼 다룹니다. 이를 통해 CNN 없이도 글로벌 단위의 어텐션(Attention)을 처리합니다.
  - **RMSNorm (Root Mean Square Normalization)**: 기존 `LayerNorm`에서 평균 이동(mean shift) 연산을 제외하고, 분산(제곱합의 평균)만을 이용해 정규화합니다. 연산 효율성이 극대화되며 배치 정규화(Batch Normalization)와 마찬가지로 학습을 안정적으로 만들어줍니다.
  - **SwiGLU MLP**: 피드포워드 신경망(FFN)에서 전통적인 ReLU 대신 Swish 비선형 활성화 함수를 사용하는 GLU(Gated Linear Unit)를 적용하여 모델의 비선형성과 성능을 끌어올립니다.
  - **MoE (Mixture of Experts)**: 모델 전체의 파라미터는 상당히 크지만, 특정 데이터나 입력마다 적절한 '전문가(Expert)' 하위 네트워크들만 선택적으로 활성화하여 연산량을 낮게 유지하면서 확장성을 높이는 기법입니다.

## 2. 컴퓨터 비전의 핵심 과제 (Computer Vision Tasks)

### 2.1. 의미론적 분할 (Semantic Segmentation)
- **Core Question**: 이미지 내의 **모든 픽셀**이 어떤 의미의 객체(클래스)에 속하는지 픽셀 단위 파싱을 어떻게 할 것인가?
- **Key Mechanism**:
  - **접근법의 한계와 발전**: 모든 픽셀마다 개별적으로 CNN을 실행하는 것은 불가능에 가깝습니다. 대안으로 단일 레이블이 아닌 전체 픽셀 맵과 동일한 해상도의 '분할 맵'을 한 번에 출력하는 FCN(Fully Convolutional Network) 방식이 등장했습니다.
  - **다운샘플링(Downsampling)과 업샘플링(Upsampling)**:
    - 병목현상 문제: 이미지 해상도를 그대로 유지하며 깊은 컨볼루션 계층을 거치면 메모리(GPU) 소비와 계산량이 천문학적으로 커집니다.
    - 따라서 해상도를 줄이면서 시야각(Receptive Field)을 넓혀 추상적 특징을 추출하는 다운샘플링 과정과, 마지막에 결과를 본래 해상도로 복원하는 업샘플링 과정을 결합합니다.
    - 업샘플링 기법: Nearest Neighbor, Bed of Nails 방식의 `Unpooling` 기법이 있으며, 더 나아가 가중치를 역전파(Backpropagation)를 통해 학습시키는 `Learnable Upsampling (Transposed Convolution)` 방식이 있습니다.
  - **U-Net 구조**: 분할 알고리즘의 표준. 다운샘플링 과정에서 공간 정보가 손실되는 문제를 보완하기 위해, `인코더` 단계의 중간 특징맵(Feature map)을 그대로 추출해 `디코더` 단계로 복사(Skip-connection)하여 공간 정보를 유지해 정밀한 분할을 가능하게 합니다.
- **Mathematical Foundation**:
  - $\text{Loss} = \frac{1}{N} \sum_{i} \text{CrossEntropy}(y_i, \hat{y}_i)$
  - (각 픽셀 $i$를 수많은 독립적인 분류 문제로 생각하고 Loss를 통합)

### 2.2. 객체 탐지 (Object Detection)
- **Core Question**: 이미지 안에 존재하는 다수의 객체에 대하여, 그것이 "무엇"인지(Class Label)와 "어디"에 있는지(Bounding Box)를 동시에 어떻게 예측할 것인가?
- **Key Mechanism**:
  - **R-CNN 계열 (Region-based CNN)**: 슬라이딩 윈도우 방법은 검사해야 할 경계 상자의 수가 너무 많아 확장성이 떨어집니다. R-CNN 계열은 이미지 내에 객체가 있을 법한 특정 '관심 영역(Region Proposal)'을 먼저 찾고, 그 영역만을 CNN에 통과시켜 위치와 종류를 판별합니다. 이후 Fast/Faster R-CNN에서는 속도 개선을 위해 신경망이 스스로 영역을 제안(RPN)하도록 발전했습니다.
  - **YOLO (You Only Look Once)**: 매우 빠른 단일 단계 감지기(Single-stage detector)입니다. 전체 이미지를 한 번 스캔한 뒤 $S \times S$ 크기의 격자(Grid)로 나눕니다. 각 격자 셀이 직접 객체의 클래스 확률과 다수의 경계 상자 위치 정보를 한 번의 순전파로 예측합니다.
  - **DETR (DEtection TRansformer)**: 어텐션 메커니즘을 이용한 최신 기술. CNN으로 뽑아낸 이미지 토큰에 위치 인덱스(Positional Encoding)를 붙여 트랜스포머 인코더에 넣습니다. 디코더에서는 설정된 개수(예: 100개)가량의 객체 쿼리가 입력되어 셀프/크로스 어텐션을 통해 최종적으로 객체의 레이블과 바운딩 박스를 동시에 생성합니다.
- **Mathematical Foundation**:
  - 다목적 손실 함수: 분류 손실(Cross Entropy 등) + 박스 좌표에 대한 회귀 손실(L2, Smooth L1 등)을 결합하여 가중합으로 처리.
  - $\text{Loss} = L_{\text{classification}} + \lambda \cdot L_{\text{box}}$

### 2.3. 인스턴스 분할 (Instance Segmentation)
- **Core Question**: 동일한 종류의 객체(예: 사람 두 명)가 있을 때, 단순히 둘 다 '사람' 픽셀로 묶는 것이 아니라 개별 인스턴스(사람1, 사람2) 단위로 분리된 정밀한 픽셀 마스크를 어떻게 얻을 것인가?
- **Key Mechanism**:
  - **Mask R-CNN**: 객체 탐지를 수행하는 과정(Faster R-CNN 등)에, 탐지된 각 객체의 경계 상자 내부에서 픽셀 단위로 객체/배경을 분할하는 새로운 순전파 브랜치(Branch)인 마스크 예측 컨볼루션 레이어를 병렬로 추가하는 방식입니다.

## 3. 모델 시각화 및 이해 (Visualization & Understanding)
- **Core Question**: 블랙박스라 불리는 딥러닝 모델(특히 CNN)이 도대체 내부적으로 이미지의 '어느 부분'을 보고 '어떤 방식'으로 판단을 내리는지 인간이 어떻게 해석할 수 있을까?
- **Key Mechanism**:
  - **모델 계층 시각화**: 가장 단순한 선형 분류기에서는 가중치를 이미지로 재구성해 '템플릿'을 얻을 수 있습니다. 그러나 CNN의 얕은 계층은 가장자리나 색감을 학습하고 깊어질수록 추상적이라 이를 시각화하기 위해서는 여러 수학적 트릭이 필요합니다 (예: 역전파를 변형한 유도 역전파 등).
  - **두드러짐 맵 (Saliency Maps)**: 입력된 이미지의 **원래 픽셀 값 변화**가 모델의 최종 출력 스코어에 얼마나 기여하는지(민감도)를 파악합니다. 즉, 가중치가 아니라 픽셀 값에 대한 미분(Gradient)을 계산합니다.
  - **CAM & Grad-CAM**: 모델이 판단할 때 가장 두드러지게 참조한 영역의 맵(Heatmap)을 생성합니다.
    - 기존의 특징 맵(Feature map)에 글로벌 풀링을 취해 중요도를 부여(CAM)하지만, 이는 구조상 한계(FC 레이어 직전)가 있습니다.
    - 이를 개선한 `Grad-CAM`은 구조적 변경 없이 마지막 컨볼루션 계층으로 전파된 활성화 그래디언트를 사용해 중요도를 평가할 수 있어 매우 인기가 높습니다.
    - 반면, 트랜스포머 기반 비전 아키텍처는 내부적으로 '어텐션(Attention) 가중치 맵' 자체가 존재하므로 활성화 및 중요 부위를 보다 용이하게 파악할 수 있는 특징을 갖습니다.

---
## 🚀 추가 학습 및 실습 제안 (Further Explorations)

이론적 이해를 한 단계 더 높이기 위해 아래의 과제와 실습을 권장합니다.

1. **[수학 및 구현 실습] Learnable Upsampling (Transposed Conv)**: 
   - `DL_scratch/source` 체제에 맞춰 NumPy로 `Transposed Convolution`의 동작 원리(영향을 미치는 수용 영역의 가중치가 역전파되는 과정)를 간단한 1D/2D 행렬 형태로 구현해 보세요. 단순 보간법(Nearest Neighbor 등)과의 차이를 파악할 수 있습니다.
2. **[이론 심화] YOLO의 격자 메커니즘과 NMS 조명**: 
   - 객체 감지에서 겹치는 경계 상자를 제거하는 비최대 억제(Non-Maximum Suppression, NMS)의 알고리즘 원리를 공부하고 파이썬으로 구현해 보세요. (YOLO 핵심 후처리 기술 중 하나)
3. **[Backpropagation 응용] Saliency Map 직접 수식 전개**: 
   - 일반적인 오차 역전파는 가중치($W$)를 업데이트하기 위한 $\frac{\partial L}{\partial W}$를 구합니다. 하지만 Saliency Map은 모델이 생성하는 클래스 스코어($Fc$)를 입력 픽셀 이미지 행렬($X$)로 미분 $\frac{\partial Fc}{\partial X}$ 하는 구조입니다. 이 차이를 명확히 인지하고 손으로 식을 전개보는 것을 추천합니다.
4. **[추천 논문 강독]**:
   - 시간 여유가 있다면, *Vision Transformer (Dosovitskiy, 2020)* 기본 논문을 읽으며 자연어 처리 모델인 Transformer가 이미지 픽셀 패치를 어떻게 Embedding 벡터로 치환했는지 수식적으로 따라가 보세요. (CS231n의 최신 흐름 파악에 필수적입니다.)
