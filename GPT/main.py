import os
import logging
import argparse
from dotenv import load_dotenv

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

def main():
    """메인 실행 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="외부 인증 문서 기반 RAG 한국어 챗봇")
    parser.add_argument("--query", "-q", type=str, help="질문 내용")
    parser.add_argument("--chunk_size", type=int, default=500, help="청크 크기")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="청크 중복 크기")
    parser.add_argument("--use_tiktoken", action="store_true", help="OpenAI tokenizer 사용")
    parser.add_argument("--embedding_model", choices=["local", "openai"], default="local", help="임베딩 모델 선택")
    parser.add_argument("--gpt_model", default="gpt-3.5-turbo", help="GPT 모델 선택")
    parser.add_argument("--top_k", type=int, default=3, help="상위 몇 개의 검색 결과를 반환할지")
    parser.add_argument("--docs_dir", type=str, help="로컬 문서 디렉토리 경로 (지정 시 API 대신 로컬 문서 사용)")
    args = parser.parse_args()
    
    # 환경 변수 로드
    load_dotenv()
    
    try:
        print("=== 외부 인증 문서 기반 RAG 한국어 챗봇 ===")
        print("초기화 중...")
        
        # 문서 가져오기 (로컬 디렉토리 또는 API)
        fetcher = DocumentFetcher(args.docs_dir)
        documents = fetcher.fetch_documents()
        
        if not documents:
            print("문서를 가져오지 못했습니다. 설정을 확인하세요.")
            return
        
        print(f"{len(documents)}개 문서를 가져왔습니다.")
        
        # 문서 처리
        processor = DocumentProcessor()
        processed_docs = processor.process_documents(documents)
        
        # 청크 생성
        chunker = TextChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_tiktoken=args.use_tiktoken
        )
        chunks = chunker.create_chunks(processed_docs)
        
        # 임베딩 생성
        embedding_generator = EmbeddingGenerator(model_type=args.embedding_model)
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        
        # 벡터 인덱스에 추가
        vector_store = VectorStore(embedding_generator.embedding_dim)
        vector_store.add_embeddings(embedded_chunks)
        
        print(f"총 {len(chunks)}개의 청크를 생성하고 임베딩했습니다.")
        
        # GPT 응답기 초기화
        responder = GPTResponder(model_name=args.gpt_model)
        
        # 대화 루프
        query = args.query
        while True:
            if not query:
                query = input("\n질문을 입력하세요 (종료하려면 'q' 또는 'exit' 입력): ")
            
            # 종료 조건
            if query.lower() in ('q', 'quit', 'exit'):
                print("프로그램을 종료합니다.")
                break
            
            # 쿼리 임베딩 생성
            query_embedding = embedding_generator.get_embedding(query)
            
            # 벡터 검색
            results = vector_store.search(query_embedding, top_k=args.top_k)
            
            if results:
                # 맥락 기반 답변 생성
                response = responder.generate_response(query, results)
                
                # 출력
                print("\n답변:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            else:
                print("관련 문서를 찾을 수 없습니다.")
            
            # 명령줄 인자로 받은 질문인 경우 한 번만 실행
            if args.query:
                break
            
            query = None  # 다음 반복에서 새 입력 받기
            
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 