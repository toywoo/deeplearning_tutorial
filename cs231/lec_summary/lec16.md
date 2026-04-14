# Lecture 16: Multi-modal Foundation Models (멀티모달 파운데이션 모델)

이 문서는 사용자의 `lec16.md` 메모 업데이트 내역(LLaVA, Flamingo 아키텍처 및 Open AI 동향)을 캡처하여, 사전에 정의한 `lecture_expansion_skill.prompt` 규칙에 따라 구조화 및 심도 있는 해설본으로 덮어쓰기 업데이트한 버전입니다.

---

## 1. 파운데이션 모델의 대통합 시대 (Foundation Models)
- **Core Question**: '고양이 분류', '영어 번역' 등 파편화된 개별 작업(Task-specific)마다 매번 전용 모델을 훈련해야 했던 비효율성을 어떻게 극복할 것인가?
- **Key Mechanism**:
  - 특정 작업군에만 국한되지 않고, 방대한 데이터를 사전 훈련(Pre-training)하여 범용적 지식과 논리를 획득한 **파운데이션 모델(Foundation Models)** 체제로 완전히 전환되었습니다.
  - GPT 같은 순수 텍스트 언어 모델을 넘어, 이제는 이미지-텍스트의 교두보를 놓은 CLIP과 CoCa를 발판 삼아 **텍스트와 이미지를 원초적으로 통합 이해하는 거대 멀티모달 모델(Gemini, GPT-4V, LLaVA 등)** 딥러닝 세계의 지배 패러다임이 되었습니다.

## 2. 텍스트와 이미지의 거리를 재다: CLIP (Contrastive Language-Image Pre-training)
- **Core Question**: 인터넷에 널려있는 막대한 양의 '이미지와 캡션 설명글' 쌍을 이용해, 신경망이 단어와 시각적 의미를 어떻게 연결 짓게(Mapping) 만들 수 있을까?
- **Key Mechanism**:
  - **대조 학습(Contrastive Learning)의 확장**: SimCLR이 이미지 간의 거리를 쟀다면, CLIP은 서로 다른 두 차원인 '이미지 임베딩 벡터'와 '텍스트 임베딩 벡터' 간의 차이를 대조 학습합니다.
  - 정답인 이미지-텍스트 쌍의 코사인 벡터 거리는 가깝게(Positive), 그 외의 틀린 조합들과의 거리는 밀쳐내도록(Negative) 강제합니다.
  - **제로샷 분류 (Zero-shot Classification)**: 훈련에 전혀 보지 못한 사물도 알아맞춥니다. 사용자가 `["강아지", "비행기", "사과"]` 라는 후보 단어를 주면, 모델이 각 단어를 텍스트 벡터로 쪼개고 현재 이미지 벡터 위치에서 가장 가까운 단어(KNN과 유사한 마스킹 기법)를 정답으로 도출해냅니다.
- **장점과 한계점**:
  - **장점**: 어휘 선택의 자유도가 무한하며, 강력한 특징 추출망으로 타 멀티모달의 핵심 눈(백본) 역할을 담당합니다.
  - **단점**: "머그컵 안에 담긴 잔디" vs "잔디밭 위에 놓인 머그컵" 같은 복잡한 공간적/관계적 매핑을 분간하기 어렵습니다.

## 3. 시력을 묘사하는 디코더 장착: CoCa (Contrastive Captioners)
- **Core Question**: CLIP은 이미지와 텍스트 쌍을 대조만 할 줄 알지, 자기가 직접 그림을 보고 어떤 상황인지 문장(캡션 생성)을 주도적으로 글짓기 할 수는 없는데 이를 어떻게 돌파할까?
- **Key Mechanism**:
  - CoCa는 일종의 CLIP 변형판으로, 이미지 인코더에서 나온 특징 맵 정보를 토대로 단어를 순차적으로 뱉어내는 **디코더(Decoder)** 를 별도로 부착한 메커니즘입니다.
  - 디코더 내부의 크로스 어텐션(Cross-Attention) 레이어를 활용해 "내가 생성할 단어가 이미지의 어떤 픽셀(특징)과 일치하는가?"를 대조해가며 훌륭한 캡셔닝을 수행합니다.

## 4. 거대 언어 모델과 시력의 융합: LLaVA & Flamingo
- **Core Question**: 이미 다음 단어 예측(Next Token Prediction)으로 극강의 논리를 갖춘 거대 언어 모델(LLM)에, '눈(렌즈)'을 달아주어 사진을 보며 대화하게 만들 방법은 무엇일까?
- **Key Mechanism**:
  - 비전 인코더(주로 CLIP의 ViT 모델)가 이미지를 분석해 만든 여러 조각 토큰들을, 일종의 "내가 모르는 새로운 자연어 단어" 인 것처럼 변환(Projection)하여 기존 언어 모델(LLM)의 입력단에 밀어넣습니다.
  - **LLaVA의 패치 토큰 살리기**: 과거엔 이미지 전체를 뜻하는 대표 토큰 달랑 하나(`[CLS]`)만 썼지만 성능이 낮았습니다. 최근엔 `[CLS]`를 제외한 나머지 모든 이미지 픽셀의 조각(Patch) 토큰정보를 버리지 않고 모델에 다 밀어넣어 이미지의 디테일과 세세한 위치를 알 수 있게 되었습니다.
  - **Flamingo의 크로스 어텐션 설계 (Deepmind)**: 
    - 플라밍고는 이미지를 텍스트 토큰과 나란히 줄세워 넣지 않고, **Perceiver Resampler**를 도입하여 방대한 이미지 정보를 고정된 수의 압축된 시각 토큰(Visual Tokens)으로 강력히 요약합니다.
    - 이후 기존 LLM의 어텐션 블록 사이에 새로운 **Gated Cross-Attention Dense** 레이어를 끼워넣어, 글을 읽어 내리는 도중에 텍스트가 위에서 압축된 시각 토큰 정보를 틈틈히 '참조'할 수 있도록 했습니다.
    - 훈련 시 이미지와 텍스트 문장이 번갈아 나오는 교차 데이터(Interleaved image-text)를 학습하여 아주 강력한 퓨샷(Few-shot) 능력을 갖게 되었습니다.

## 5. 오픈소스 진영 (Open vs Closed AI) 동향
- **Core Question**: ChatGPT(GPT-4V)나 Gemini 같은 최고 수준의 멀티모달 모델은 폐쇄형 시스템(Closed AI)인데, 개별 연구자들은 어떻게 이 격차를 좁히고 연구의 자율성에 벗어날 수 있을까?
- **Key Mechanism**:
  - Meta의 오픈소스 언어 모델인 **LLaMA** 등을 기반 뇌(Brain)로 삼고, 앞단 인코더에 오픈소스 비전 모델(OpenCLIP)을 어댑터 매커니즘으로 단순하게 잇는 기법이 폭발하며 접근성이 낮아졌습니다.
  - 그 결과 LLaVA, OpenFlamingo 등 대규모 컴퓨팅 자원 없이도 상업용 제품에 필적하는 오픈소스 파운데이션 모델 생태계가 찬란하게 구축되었습니다.

---
## 🚀 추가 학습 및 실습 제안 (Further Explorations)

1. **[수학 및 구조 파싱] Gated Cross-Attention 메커니즘**:
   - Flamingo의 핵심인 `Gated Cross-Attention` 레이어에서 어텐션 출력값에 $\text{tanh}(\alpha)$ 와 같은 학습 가능한 Gating Parameter를 왜 곱하는지 탐구해 보세요. 이 초기값을 0으로 설정해 사전 학습된 고정 언어 모델(Brain)이 망가지는 것을 어떻게 방지했는지가 관건입니다.
2. **[코드 실습] Hugging Face를 통한 LLaVA 및 OpenCLIP 추론**:
   - Hugging Face 웹에서 지원하는 `transformers` 모듈로 LLaVA 파이프라인(`llava-hf/llava-1.5-7b-hf` 등)을 가져온 후, 임의의 사진을 넣고 "이 사진 속 가장 특이한 점이 뭐야?"라고 프롬프트를 날렸을 때 토큰화가 작동하는 파이썬 스크립트를 구현해 보세요.
3. **[논문 리딩] (선택형 심화)**:
   - *Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)*: 언어 사이사이에 Gated Attention으로 시각 정보를 녹여 넣은 딥마인드의 명문 아키텍처 논문입니다.
   - *Visual Instruction Tuning / LLaVA (Liu et al., 2023)*: 복잡한 개조 없이 단순히 선형 레이어 프로젝션(Projection)만으로 비전과 언어를 연결 짓는 발상의 전환을 이끈 논문입니다.
