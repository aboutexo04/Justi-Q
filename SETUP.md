# Justi-Q 환경 설정 가이드

## 1. Conda 환경 생성

```bash
conda create -n justiq python=3.10 -y
conda activate justiq
```

## 2. 의존성 설치

```bash
pip install -r requirements.txt
```

## 3. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
# LLM API Key (아래 중 하나만 설정)
OPENROUTER_API_KEY=your_openrouter_api_key   # 옵션 1
# OPENAI_API_KEY=your_openai_api_key         # 옵션 2
# SOLAR_API_KEY=your_solar_api_key           # 옵션 3

# 모델 커스텀 (선택사항)
# LLM_MODEL=gpt-4o-mini

# 선택 (LangSmith 모니터링)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=justi-q
```

### API 키 발급처

**LLM (하나만 선택)**
| Provider | URL | 기본 모델 | 비고 |
|----------|-----|-----------|------|
| OpenRouter | https://openrouter.ai | Llama 3.3 70B | 무료 모델 지원 |
| OpenAI | https://platform.openai.com | gpt-4o-mini | 유료 |
| Solar | https://console.upstage.ai | solar-pro | 한국어 특화 |

**모니터링 (선택)**
- **LangSmith**: https://smith.langchain.com → Settings → API Keys

## 4. 벡터 DB 설정

### 옵션 A: 기존 DB 사용 (권장)
팀에서 공유한 `chroma_db.zip`을 프로젝트 루트에 압축 해제

### 옵션 B: 새로 인덱싱
```bash
python main.py --index
```
(M4 Mac 기준 약 10-15분 소요)

## 5. 실행

### CLI 대화형 모드
```bash
python main.py
```

### 단일 질문
```bash
python main.py --query "폭행죄 처벌 기준은?"
```

### 웹 UI (Streamlit)
```bash
streamlit run app.py
```

## 6. LangSmith 모니터링 (선택)

LangSmith를 설정하면 RAG 파이프라인의 실행 로그를 실시간으로 확인할 수 있습니다.

1. https://smith.langchain.com 에서 계정 생성
2. Settings → API Keys에서 키 발급
3. `.env` 파일에 `LANGCHAIN_API_KEY` 추가
4. Streamlit 재시작 후 질문하면 자동으로 트레이싱

트레이스 확인: https://smith.langchain.com → Tracing → `justi-q` 프로젝트

## 문제 해결

### ModuleNotFoundError
```bash
conda activate justiq
pip install -r requirements.txt
```

### ChromaDB 오류
`chroma_db/` 폴더 삭제 후 다시 인덱싱하거나 공유된 DB 사용

### LangSmith 트레이싱 안 됨
1. `.env`에 `LANGCHAIN_API_KEY` 있는지 확인
2. Streamlit 완전 종료 후 재시작
3. https://smith.langchain.com 에서 `justi-q` 프로젝트 확인
