1. 파이썬 설치
2. 파이썬 버전과 파이토치 버전 의존성 확인하기
   * https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix 
3. 파이토치 버전에 맞는 쿠다 설치
   * https://developer.nvidia.com/cuda-toolkit-archive 
   * nvcc --version
4. 쿠다에 맞는 cuDNN 설치
   * cuDNN이란 심층 신경망을 위한 GPU 가속 라이브러리
5. 파이토치를 설치하는데 그냥 pip 명령어 입력하는 것이 아닌 아래 주소에 맞는 명령어 입력하기
   * https://pytorch.org/get-started/locally/
6. 파이토치와 파이토치_쿠다가 잘 작동되는지 확인하기
```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```