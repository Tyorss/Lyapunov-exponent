# 외부 인증 문서 기반 RAG 한국어 챗봇

인증이 필요한 외부 API에서 문서를 가져와 한국어 RAG(Retrieval-Augmented Generation) 질의응답 챗봇을 구현한 프로젝트입니다.

## 주요 기능

- **외부 인증 문서 수집**: Bearer 토큰, API 키 등을 통한 인증된 외부 API 접근
- **다양한 문서 포맷 지원**: PDF, DOCX, TXT 등 다양한 형식의 문서 처리
- **텍스트 청킹**: 문서를 적절한 크기의 청크로 분할
- **임베딩 생성**: 한국어에 최적화된 임베딩 생성 (Sentence Transformers 또는 OpenAI)
- **벡터 검색**: FAISS를 활용한 효율적인 벡터 검색
- **GPT 응답 생성**: 관련 문서를 기반으로 한 정확한 답변 생성
- **Streamlit 웹 인터페이스**: 사용자 친화적인 채팅 인터페이스

## 시스템 구성

![시스템 구성도](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*55ggifIM-B9TVR0-Q9jH6A.png)

## 설치 방법

1. 저장소 클론
```bash
git clone <repository_url>
cd <repository_directory>
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키와 설정을 입력하세요
```

## 실행 방법

```bash
streamlit run app.py
```

웹 브라우저에서 자동으로 열리는 Streamlit 앱에 접속하여 사용할 수 있습니다.

## 모듈 구성

- `document_fetcher.py`: 외부 API 인증 및 문서 수집
- `document_processor.py`: 다양한 포맷의 문서 텍스트 추출
- `text_chunker.py`: 효율적인 텍스트 청킹 처리
- `embedding_generator.py`: 임베딩 벡터 생성
- `vector_store.py`: FAISS 벡터 인덱스 관리
- `gpt_responder.py`: GPT 기반 응답 생성
- `app.py`: Streamlit 웹 애플리케이션

## 사용 예시

1. 사이드바에서 필요한 API 키와 설정을 입력합니다.
2. "시스템 초기화" 버튼을 클릭하여 문서를 처리하고 임베딩을 생성합니다.
3. 채팅 인터페이스에서 질문을 입력하면 관련 문서에서 정보를 검색하여 답변합니다.

## 환경 변수

- `OPENAI_API_KEY`: OpenAI API 키
- `API_URL`: 문서 API 엔드포인트
- `BEARER_TOKEN`: Bearer 토큰 인증 정보
- `API_KEY`: API 키 인증 정보
- `EMBEDDING_MODEL`: 임베딩 모델 유형 (`local` 또는 `openai`)
- `LOCAL_MODEL_NAME`: 로컬 임베딩 모델명 (기본값: `jhgan/ko-sbert-sts`)

## 라이선스

[MIT License](LICENSE)

## 기여

이슈와 풀 리퀘스트는 언제나 환영합니다! 