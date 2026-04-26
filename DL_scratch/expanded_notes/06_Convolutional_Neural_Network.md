# [Expanded] 합성곱 신경망 (Convolutional Neural Network, CNN)

이 문서는 `1_7_CNN.ipynb`의 내용을 바탕으로 작성된 심층 해설 문서입니다.

## 1. Core Question (핵심 질문)
**"주변 픽셀들 간의 공간적/기하학적 연관성이 매우 높은 이미지의 본질을 파괴하는 이전 '완전 연결(Fully Connected)' 방식의 한계를 어떻게 극복할 것인가?"**

지금까지 등장한 모든 신경망 계층(Affine 계층)은 다차원의 이미지 데이터를 `1차원 벡터로 눌러서 쫙 펴버리기` 때문에, 가로·세로에 존재하는 형상(Shape)과 픽셀 인접성에 담긴 귀중한 정보들이 완전히 무시되었습니다. **합성곱 신경망(CNN)**은 이 시각적 3차원 입체 형상을 그대로 유지한 채로 연산을 처리함으로써, 컴퓨터가 "저기 자동차 바퀴가 있다, 저기 뾰족한 고양이 귀가 있다"라는 특징을 공간 정보 손실 없이 뽑아낼 수 있게 해주는 혁명적인 아키텍처입니다.

## 2. Key Mechanism (해결 원리)

### 합성곱 계층 (Convolutional Layer)
CNN의 코어 엔진은 특징을 찾는 작은 유리창인 '필터(Filter 또는 커널)'입니다. 입력 데이터($Height \times Width$)위를 필터가 일정 간격(스트라이드, Stride)으로 미끄러지며 스캔합니다. 이때 겹친 부분의 원소끼리 곱해서 전부 더하는 **단일 곱셈 누산(Fused Multiply-Add)**이 반복 연산됩니다.
필터를 수십, 수백 개씩 사용하므로 여러 개의 특징 맵(Feature Map)이 두껍게 쌓인 3차원 데이터가 출력됩니다. 피처맵은 모서리, 질감, 색상 등 구체적 정보를 내포합니다.

### 패딩(Padding)과 풀링(Pooling)
- **패딩**: 특징 맵이 스트라이드 횟수만큼 점점 작아져서 소멸되는 것을 막고자, 원본 테두리에 0($0$, Zero-Padding)을 둘러싸 출력 공간을 보호합니다.
- **풀링 계층**: 가로 세로 방향의 공간 차원을 억눌러 압축합니다. 최대 풀링(Max Pooling)의 경우 대상 영역 내 '가장 강력한 특징(최댓값)' 하나만 남깁니다. 작은 평행 이동이나 비틀어짐에도 강인(Robust)해지는 이동 불변성(Translation Invariance) 효과를 갖습니다.

### im2col (Image to Column) 기법 최적화
CNN의 연산을 순수 `for` 루프(중첩 4번~6번 반복)로 구현하면 넘파이 연산이 한없이 느려집니다. 대신 타겟인 3차원 블록 데이터를 가로 방향 1차원 행렬(Column)로 쫙 펴서 필터와 **단 한 번의 거대 행렬 곱셈(`np.dot`)**으로 끝내버리는 트릭을 씁니다. 이를 `im2col`이라고 부릅니다.
> 💡 **코드 연계**: 이론부터 넘어와 최신 프레임워크(PyTorch) 수준에서 추상화된 전체 계층(`nn.Conv2d, nn.MaxPool2d` 조합) 및 `view`를 통한 평탄화 로직은 [1_7_CNN.ipynb:L151-L177](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_7_CNN.ipynb#L151-L177) 클래스에서 확인해 보실 수 있습니다. 아울러 완전한 백프로퍼게이션 연산 루프(`loss.backward()`)는 [1_7_CNN.ipynb:L228-L242](file:///c:/Users/vilab/csw/deep_learning/DL_scratch/source/1_7_CNN.ipynb#L228-L242)에 포함되어 있습니다!

````

````

## 3. Mathematical Foundation (수학적 기반)

### 출력 형상(Output Shape) 계산 공식
입력 이미지 크기를 $(H, W)$, 필터 크기를 $(FH, FW)$, 패딩 폭을 $P$, 스트라이드를 $S$라 할 때, 출력되는 합성곱 맵의 크기 $(OH, OW)$는 다음 수학 공식을 따릅니다. 프레임워크 사용 시 이 값이 딱 나누어 떨어져야 에러가 나지 않으므로 필수적으로 암기해야 합니다.

$$ OH = \frac{H + 2P - FH}{S} + 1 $$
$$ OW = \frac{W + 2P - FW}{S} + 1 $$

### 4차원 배치(Batch) 텐서(Tensor)의 차원 변전
$(N, C, H, W)$ 의 미니배치 데이터에 채널이 $C$이고 갯수가 $FN$인 편향 $\mathbf{B}$ 필터를 붙이면:
$$ (N, C, H, W) \circledast (FN, C, FH, FW) \,+\, \mathbf{B}(FN, 1, 1) \rightarrow (N, FN, OH, OW) $$
이 공식이 모든 현대 CNN 인공지능 프레임워크의 수학적 데이터 형상(Shape) 뼈대입니다.

### 💻 실행 가능한 파이썬 코드 스니펫

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1채널 흑백 입력 -> 16개 피처맵 출력, 커널 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        print(f"입력 원래 형상: {x.shape}")
        
        x = self.conv1(x)
        print(f"Conv 통과 후 형상 (패딩으로 공간 유지): {x.shape}")
        
        x = self.relu(x)
        x = self.pool(x)
        print(f"MaxPool 통과 후 형상 (가로세로 공간 압축): {x.shape}")
        
        # 3차원 입체 피처맵 텐서를 1차원으로 전개하여 Fully Connected 레이어용으로 평탄화
        x = x.view(x.size(0), -1) 
        print(f"Flatten 완전 연결 대기 형상: {x.shape}")
        return x

model = SimpleCNN()
# [배치(N), 채널(C), 세로(H), 가로(W)] => 임의의 이미지 데이터 1장 (28x28 해상도 흑백)
dummy_image = torch.randn(1, 1, 28, 28) 

print("--- 심플 CNN 데이터 전방 패스 테스트 ---")
output = model(dummy_image)
print("---------------------------------")
print("이론적 출력 크기 계산: OH = (28 + 2*2 - 5)/1 + 1 = 28 -> MaxPool(절반) = 14")
print(f"형상 전개: 16채널 * 14 * 14 = 총 {16*14*14} 열")
```

## 🚀 추가 학습 및 실습 제안

1. **[코딩 실습] PyTorch 출력 Shape 검토**
   - 본 노트북 구조의 `CNN` 클래스에서 처음 `in_channels=1, out_channels=16, kernel_size=5` Conv2d 에 `(1, 1, 28, 28)` 크기의 이미지를 넣고, 계산 공식에 대입해 과연 `(1, 16, 24, 24)` 의 텐서가 출력되는지 실제 코드를 브레이크포인트로 찍어 손계산과 비교해보세요.
2. **[수학적 고찰] im2col 메모리 트레이드오프 연산**
   - 앞서 해설한 `im2col` 방식은 속도는 경이롭지만 패딩 혹은 스트라이드 간격이 좁을 때(서로 필터 영역이 많이 겹칠 때) 원래 데이터보다 변환된 2차원 배열의 총 메모리 용량이 원소 단위로 몇 배나 폭발적으로 증가하는지 직접 메모리 공간 복잡도 비율을 유도해 보세요.
3. **[논문 리뷰 제안] "Gradient-based learning applied to document recognition" (LeCun et al., 1998)**
   - 오늘날 CNN 아키텍처의 기원인 전설의 'LeNet-5' 논문입니다. 은행 수표의 우편번호(수기)를 컴퓨터로 인식하기 위해 처음으로 합성곱 스트라이드와 서브샘플링(풀링) 개념을 도입한 역사적 논문을 살펴보시기 권장해 드립니다.
