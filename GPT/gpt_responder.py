import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class GPTResponder:
    """검색 결과를 기반으로 답변을 생성하는 클래스"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Args:
            model_name: 사용할 GPT 모델 이름
        """
        self.model_name = model_name
        
        # OpenAI API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        logger.info(f"GPT 응답기가 '{model_name}' 모델로 초기화되었습니다.")
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """사용자 질의와 검색 결과를 바탕으로 답변을 생성합니다.
        실제 구현을 단순화하여 임의 응답을 생성합니다.
        
        Args:
            query: 사용자 질문
            search_results: 벡터 검색 결과 리스트
            
        Returns:
            str: 생성된 답변 텍스트
        """
        if not search_results:
            return "검색 결과가 없어 답변을 생성할 수 없습니다."
        
        # 컨텍스트 준비 (검색 결과를 하나의 문자열로 연결)
        context = self._prepare_context(search_results)
        
        try:
            # 실제 구현에서는 GPT API 호출
            # 여기서는 간단한 응답 생성
            logger.info(f"'{query}' 질문에 대한 응답 생성")
            
            # 간단한 응답 템플릿
            sources = ", ".join(set(result.get("source_doc", "") for result in search_results))
            
            # 첫 번째 결과의 내용 사용
            content = search_results[0]["text"]
            if len(content) > 500:
                content = content[:500] + "..."
                
            response = f"""
검색 결과에 기반한 답변:

{content}

원하는 질문에 답변이 되셨나요? 더 자세한 내용이 필요하시면 말씀해주세요.

출처: {sources}
            """
            
            return response.strip()
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 컨텍스트 문자열로 변환합니다."""
        context_parts = []
        
        for i, result in enumerate(search_results):
            # 청크 텍스트와 출처 정보 추출
            text = result.get("text", "")
            source = result.get("source_doc", "알 수 없는 출처")
            
            # 컨텍스트에 추가
            context_parts.append(f"[문서 {i+1}] (출처: {source})\n{text}\n")
        
        return "\n".join(context_parts)

if __name__ == "__main__":
    # 간단한 테스트 코드
    from document_fetcher import DocumentFetcher
    from document_processor import DocumentProcessor
    from text_chunker import TextChunker
    from embedding_generator import EmbeddingGenerator
    from vector_store import VectorStore
    
    # 전체 파이프라인 테스트
    fetcher = DocumentFetcher()
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # 문서 처리
    documents = fetcher.fetch_documents()
    processed_docs = processor.process_documents(documents)
    chunks = chunker.create_chunks(processed_docs)
    
    # 임베딩 생성
    embedding_generator = EmbeddingGenerator()
    embedded_chunks = embedding_generator.generate_embeddings(chunks)
    
    # 벡터 인덱스에 추가
    vector_store = VectorStore(embedding_generator.embedding_dim)
    vector_store.add_embeddings(embedded_chunks)
    
    # 검색 및 응답 생성
    test_query = "RAG 시스템이란 무엇인가요?"
    query_embedding = embedding_generator.get_embedding(test_query)
    
    results = vector_store.search(query_embedding, top_k=3)
    
    # GPT 응답 생성
    responder = GPTResponder()
    response = responder.generate_response(test_query, results)
    
    # 결과 출력
    print(f"질문: {test_query}")
    print("\n답변:")
    print(response) 