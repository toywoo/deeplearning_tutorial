# 다른 환경에서 프로젝트 설정하기

이 저장소에는 코드를 실행하기 위한 환경 설정 목록(`requirements.txt`)이 포함되어 있습니다. 다른 컴퓨터나 환경에서 이 프로젝트를 처음 실행할 때는 아래의 순서대로 진행해 주세요.

## 1. 가상 환경 생성
프로젝트 폴더로 이동한 후, 해당 폴더 내에 `.venv`라는 이름의 새로운 가상 환경을 만듭니다.

```bash
python3 -m venv .venv
```

## 2. 가상 환경 활성화
생성한 가상 환경을 활성화합니다. 운영체제에 따라 명령어가 다릅니다.

- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

- **Windows**:
  ```cmd
  .venv\Scripts\activate
  ```

## 3. 필수 패키지 설치
가상 환경이 활성화된 상태(터미널 프롬프트 앞에 `(.venv)`가 표시됨)에서, `requirements.txt`에 명시된 모든 패키지를 한 번에 설치합니다.

```bash
pip install -r requirements.txt
```

## 4. 커널 선택 (VS Code 및 IDE)
Jupyter Notebook 파일(`.ipynb`)을 계속해서 사용하시려면, 우측 상단의 커널 선택(Kernel Select) 버튼을 누르고 방금 생성한 가상 환경(`.venv`)을 노트북 실행 커널로 선택하여 코드를 실행해 주시면 됩니다.

## 5. requirements.txt 추출
pip freeze > requirements.txt