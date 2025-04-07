import os
import streamlit as st
import logging
from typing import List, Dict, Any
import time

# 로컬 모듈 임포트
from document_fetcher import DocumentFetcher
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from gpt_responder import GPTResponder

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 페이지 기본 설정
st.set_page_config(
    page_title="외부 인증 문서 기반 RAG 한국어 챗봇",
    page_icon="📚",
    layout="wide"
)

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.embedding_generator = None
    st.session_state.responder = None
    st.session_state.documents_info = None

# 사이드바에 입력 폼 추가
with st.sidebar:
    st.title("🤖 RAG 챗봇 설정")
    
    with st.expander("API 설정", expanded=not st.session_state.initialized):
        openai_api_key = st.text_input("OpenAI API 키", type="password", 
                                      value=os.getenv("OPENAI_API_KEY", ""))
        gpt_model = st.selectbox("GPT 모델", 
                                options=["gpt-3.5-turbo", "gpt-4"], 
                                index=0)
    
    with st.expander("문서 API 설정", expanded=not st.session_state.initialized):
        api_url = st.text_input("API 엔드포인트 URL", 
                               value=os.getenv("API_URL", ""))
        
        auth_type = st.radio("인증 방식", 
                            options=["Bearer 토큰", "API 키", "없음"], 
                            index=2)
        
        if auth_type == "Bearer 토큰":
            bearer_token = st.text_input("Bearer 토큰", type="password", 
                                        value=os.getenv("BEARER_TOKEN", ""))
        elif auth_type == "API 키":
            api_key = st.text_input("API 키", type="password", 
                                   value=os.getenv("API_KEY", ""))
    
    with st.expander("청킹 설정"):
        chunk_size = st.slider("청크 크기", min_value=100, max_value=2000, value=500, step=100)
        chunk_overlap = st.slider("청크 중복", min_value=0, max_value=200, value=50, step=10)
    
    # 초기화 버튼
    init_button = st.button("시스템 초기화")

# 초기화 버튼 클릭 시 실행
if init_button or not st.session_state.initialized:
    with st.spinner("시스템을 초기화하는 중입니다..."):
        try:
            # 환경 변수 설정
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            if api_url:
                os.environ["API_URL"] = api_url
            if auth_type == "Bearer 토큰" and bearer_token:
                os.environ["BEARER_TOKEN"] = bearer_token
            elif auth_type == "API 키" and api_key:
                os.environ["API_KEY"] = api_key
            
            # 문서 가져오기
            fetcher = DocumentFetcher()
            documents = fetcher.fetch_documents()
            
            # 문서 처리
            processor = DocumentProcessor()
            processed_docs = processor.process_documents(documents)
            
            # 청크 생성
            chunker = TextChunker(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap)
            chunks = chunker.create_chunks(processed_docs)
            
            # 임베딩 생성
            embedding_generator = EmbeddingGenerator()
            embedded_chunks = embedding_generator.generate_embeddings(chunks)
            
            # 벡터 인덱스에 추가
            vector_store = VectorStore(embedding_generator.embedding_dim)
            vector_store.add_embeddings(embedded_chunks)
            
            # GPT 응답기 초기화
            responder = GPTResponder(model_name=gpt_model)
            
            # 세션 상태에 저장
            st.session_state.embedding_generator = embedding_generator
            st.session_state.vector_store = vector_store
            st.session_state.responder = responder
            st.session_state.initialized = True
            
            # 문서 정보 저장
            st.session_state.documents_info = {
                "total_docs": len(documents),
                "doc_names": [name for name, _ in documents],
                "total_chunks": len(chunks)
            }
            
            st.success(f"{len(documents)}개 문서를 처리하고 {len(chunks)}개 청크로 변환했습니다. 시스템이 준비되었습니다!")
            
        except Exception as e:
            st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
            st.session_state.initialized = False
            logger.error(f"초기화 오류: {str(e)}")

# 메인 콘텐츠 영역
st.title("📚 외부 인증 문서 기반 RAG 한국어 챗봇")

# 문서 정보 표시
if st.session_state.documents_info:
    info = st.session_state.documents_info
    st.markdown(f"""
    **처리된 문서**: {info['total_docs']}개 | **총 청크 수**: {info['total_chunks']}개
    
    **문서 목록**: {', '.join(info['doc_names'][:5])}{'...' if len(info['doc_names']) > 5 else ''}
    """)

# 메시지 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리
if st.session_state.initialized:
    user_query = st.chat_input("질문을 입력하세요")
    
    if user_query:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
        
        # 답변 생성 과정
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                # 임베딩 생성
                try:
                    # 쿼리 임베딩 생성
                    embedding_generator = st.session_state.embedding_generator
                    query_embedding = embedding_generator.get_embedding(user_query)
                    
                    # 벡터 검색
                    vector_store = st.session_state.vector_store
                    results = vector_store.search(query_embedding, top_k=3)
                    
                    if results:
                        # 맥락 기반 답변 생성
                        message_placeholder = st.empty()
                        
                        # 내용 표시 (스트리밍 없이)
                        responder = st.session_state.responder
                        response = responder.generate_response(user_query, results)
                        message_placeholder.markdown(response)
                        
                        # 메시지 기록에 추가
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "관련 문서를 찾을 수 없습니다."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                except Exception as e:
                    error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"답변 생성 오류: {str(e)}")
else:
    st.info("👈 시스템을 초기화하려면 왼쪽 사이드바에서 필요한 설정을 입력하고 '시스템 초기화' 버튼을 클릭하세요.")

# 푸터
st.markdown("---")
st.markdown("🔍 **외부 인증 문서 기반 RAG 한국어 챗봇** - 문서 기반 질의응답 시스템") 